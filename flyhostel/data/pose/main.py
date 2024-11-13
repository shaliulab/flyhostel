import time
import io
import shutil
import itertools
import h5py
import glob
import time
import os.path
import logging
import sqlite3
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np

from flyhostel.data.pose.pose import FilterPose
from flyhostel.utils import load_roi_width, load_metadata_prop, restore_cache, save_cache
from flyhostel.data.pose.movie_old import make_pose_movie
from flyhostel.data.pose.constants import MIN_TIME, MAX_TIME
from imgstore.interface import VideoCapture
from flyhostel.data.pose.loaders.wavelets import WaveletLoader
from flyhostel.data.pose.loaders.behavior import BehaviorLoader
from flyhostel.data.pose.landmarks import LandmarksLoader

from flyhostel.data.pose.loaders.pose import PoseLoader
from flyhostel.data.pose.loaders.centroids import load_centroid_data
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.data.pose.constants import ROI_WIDTH_MM
from flyhostel.data.pose.sleep import SleepAnnotator
from flyhostel.data.pose.loaders.centroids import flyhostel_sleep_annotation_primitive as flyhostel_sleep_annotation
from flyhostel.data.pose.loaders.centroids import to_behavpy
from flyhostel.utils.filesystem import FilesystemInterface
from motionmapperpy import setRunParameters
wavelet_downsample=setRunParameters().wavelet_downsample
pd.set_option("display.max_columns", 100)

def dunder_to_slash(experiment):
    tokens = experiment.split("_")
    return tokens[0] + "/" + tokens[1] + "/" + "_".join(tokens[2:4])


# keep only interactions where the distance between animals is max mm_max mm

from flyhostel.data.pose.sleap import draw_video_row
from flyhostel.data.deg import DEGLoader
from flyhostel.data.pose.video_crosser import CrossVideo

def make_int_or_str(values):
    out=[]
    for val in values:
        try:
            out.append(str(int(val)))
        except ValueError:
            out.append(val)
    return out


class FlyHostelLoader(CrossVideo, FilesystemInterface, SleepAnnotator, PoseLoader, WaveletLoader, BehaviorLoader, DEGLoader, FilterPose, LandmarksLoader):
    """
    Analyse microbehavior produced in the flyhostel

    experiment="FlyHostelX_XX_XX-XX-XX_XX-XX-XX"

    loader = FlyHostelLoader(experiment, n_jobs=20)
    # n_jobs simply controls how many processes to use in parallel when loading idtrackerai (centroid) data

    # loads centroid data (idtrackerai) and pose data (SLEAP)
    loader.load_data(min_time=14*3600, max_time=22*3600, time_system="zt")
    # populates loader.dt (centroid) and loader.pose (pose)

    # quantifies bouts of proboscis extension
    loader.detect_proboscis_extension(self.pose)
    # output is saved in loader.pe_df

    # output is saved in loader.dt_sleep (original framerate)
    # and loader.dt_sleep_2fps

    ## annotate interactions between flies and keep track of which body part was used
    # connect pose and centroid
    loader.integrate(self.dt, self.pose_boxcar)
    # pre-filter frames so only frames where at least two animals are at < 3 mm of each other are kept
    loader.compute_pairwise_distances(dist_max=3)

    # now on this subset, compute the interfly body pair distance
    # find the minimum distance between 2 bodyparts of different flies
    # and require it to be less than 2 mm for 3 seconds
    loader.annotate_interactions(dist_max=2, min_bout=3)

    # output is saved in
    loader.interactions_sleep


    # to load DEG human made labels (ground_truth)

    # if identity is None, all available identities in the loader are loaded
    loader.load_deg_data(identity=None)
    # now loader.deg is populated

    """

    def __init__(self, experiment, identity, *args, lq_thresh=1, roi_width_mm=ROI_WIDTH_MM, n_jobs=1, chunks=None,
                 roi_0_table=None, identity_table=None, **kwargs
        ):
        super(FlyHostelLoader, self).__init__(*args, **kwargs)

        basedir = os.environ["FLYHOSTEL_VIDEOS"] + "/" + dunder_to_slash(experiment)
        if not os.path.exists(basedir):
            dirs=glob.glob(basedir + "*")
            if len(dirs) == 1:
                basedir = dirs[0]
            else:
                raise Exception(f"{basedir} not found")
        self.basedir=basedir


        self.experiment = experiment
        self.identity=int(identity)
        self.lq_thresh = lq_thresh
        self.n_jobs=n_jobs
        self.datasetnames=[experiment + "__" + str(identity).zfill(2)]
        self.ids = self.make_ids(self.datasetnames)

        if self.identity is not None:
            pose_found=any([int(id.split("|")[1])==self.identity for id in self.ids])
            if not pose_found:
                logger.error("Pose not found")

        self.dbfile = self.load_dbfile()
        self.store_path=os.path.join(os.path.dirname(self.dbfile), "metadata.yaml")
        self.store=None
        self.store_index=None
        self.chunks=chunks
        self.stride=None
        self.roi_width = load_roi_width(self.dbfile)

        # because the arena is circular
        self.roi_height = self.roi_width

        self.framerate= int(float(load_metadata_prop(dbfile=self.dbfile, prop="framerate")))
        self.roi_width_mm=roi_width_mm
        self.px_per_mm=self.roi_width/roi_width_mm

        self.pose=None

        self.dt=None
        self.neighbors_df=None
        self.dt_sleep=None
        self.dt_with_pose=None
        self.out=None
        self.out_filtered=None
        self.dt_sleep_raw=None
        self.dt_with_pose_nns=None
        self.distances=None
        self.distances_sleep=None
        self.interactions_sleep=None
        self.rejections=None
        self.dt_sleep_2fps=None
        self.analysis_data=None


        self.load_meta_info()
        if identity_table is None:
            if self.number_of_animals==1:
                self.identity_table="IDENTITY"
            else:
                self.identity_table="IDENTITY_VAL"
        else:
            self.identity_table=identity_table

        if roi_0_table is None:
            if self.number_of_animals==1:
                self.roi_0_table="ROI_0"
            else:
                self.roi_0_table="ROI_0_VAL"
        else:
            self.roi_0_table=roi_0_table

    def load_meta_info(self):
        """
        Populate meta_info dictionary with keys:
        
        * t_after_ref: Number of seconds between start time and ZT0. Add it to an imgstore timestamp to get the ZT time
        """

        self.meta_info={}
        assert os.path.exists(self.dbfile), f"{self.dbfile} does not exist"
        with sqlite3.connect(self.dbfile) as conn:
            start_time=int(float(pd.read_sql(sql="SELECT value FROM METADATA WHERE field = 'date_time';", con=conn)["value"].values.item()))
            start_time=start_time-start_time%3600
            start_time=start_time%(24*3600)
            metadata_str=pd.read_sql(sql="SELECT value FROM METADATA WHERE field = 'ethoscope_metadata'", con=conn)["value"].values.tolist()[0]

        metadata=pd.read_csv(io.StringIO(metadata_str)).iloc[:, 1:]

        try:
            metadata_single_animal=metadata.loc[metadata["identity"]==self.identity]
            if metadata_single_animal.shape[0]==0:
                raise KeyError
        except KeyError:
            metadata_single_animal=metadata.loc[metadata["region_id"]==self.identity]
        
        if metadata_single_animal.shape[0]!=1:
            logger.error("%s rows for identity %s in %s", metadata_single_animal.shape[0], self.identity, self.dbfile)
            raise Exception


        reference_hour=(metadata_single_animal["reference_hour"]*3600).item()
        
        assert "identity" in metadata_single_animal.columns, f"identity column not found in metadata"
        metadata_single_animal["identity"]=make_int_or_str(metadata_single_animal["identity"])
        metadata_single_animal["region_id"]=make_int_or_str(metadata_single_animal["region_id"])
        metadata_single_animal["number_of_animals"]=make_int_or_str(metadata_single_animal["number_of_animals"])
        self.metadata=metadata_single_animal
        self.number_of_animals=int(self.metadata["number_of_animals"].iloc[0])

        self.meta_info={"t_after_ref": start_time-reference_hour} # seconds


    def __str__(self):
        return f"{self.experiment}__{str(self.identity).zfill(2)}"


    def list_ids(self):
        return np.unique(self.pose["id"])


    @classmethod
    def from_metadata(cls, flyhostel_number, number_of_animals, flyhostel_date, flyhostel_time, identity):
        experiment = f"FlyHostel{flyhostel_number}_{number_of_animals}X_{flyhostel_date}_{flyhostel_time}"
        loader=cls(experiment=experiment, identity=identity)
        return loader

    @classmethod
    def load_single_hostel_from_metadata(cls, *args, **kwargs):
        loader=cls.from_metadata(*args, **kwargs)
        dt=loader.load_analysis_data()
        return dt


    def get_simple_metadata(self):
        number_of_animals=int(os.path.basename(self.experiment).split("_")[1].replace("X", ""))
        flyhostel_number=int(os.path.basename(self.experiment).split("_")[0].replace("FlyHostel", ""))
        flyhostel_date, flyhostel_time=os.path.basename(self.experiment).split("_")[2:4]

        animal=f"{self.experiment}__{str(self.identity).zfill(2)}"
        id=self.make_ids(self.datasetnames)

        return pd.DataFrame({
            "identity": [self.identity],
            "number_of_animals": [number_of_animals],
            "flyhostel_number": [flyhostel_number],
            "flyhostel_date": [flyhostel_date],
            "flyhostel_time": [flyhostel_time],
            "experiment": [self.experiment],
            "animal": [animal],
            "basedir": [self.basedir],
            "id": [id],
        })


    def sleep_annotation(self, *args, source="inactive", **kwargs):
        if source=="inactive":
            return self.sleep_annotation_inactive(*args, **kwargs)
        elif source=="centroids":
            return flyhostel_sleep_annotation(*args, **kwargs)


    def compile_analysis_data(self):
        data=self.dt.copy()
        meta=data.meta.copy()
        centroid_columns=data.columns.tolist()
        data=data.loc[data["frame_number"] % wavelet_downsample == 0]
        behavior_columns=["behavior",  "score", "bout_in", "bout_out", "bout_count", "duration"]

        if self.behavior is None:
            logger.warning("Behavior not computed for %s", self)
            for column in behavior_columns:
                data[column]=np.nan
        else:
            data=data.merge(self.behavior.drop("t", axis=1, errors="ignore"), on=["id", "frame_number"], how="left")

        fields = centroid_columns + behavior_columns
        data=data[fields]
        data=to_behavpy(data, meta)
        self.analysis_data=data
        return data


    def load_analysis_data(self):
        if self.analysis_data is None:

            self.load_and_process_data(
                stride=1,
                cache="/flyhostel_data/cache",
                filters=None,
                useGPU=0,
            )
            self.compile_analysis_data()

        return self.analysis_data


    def process_data(self, stride, *args, min_time=MIN_TIME, max_time=MAX_TIME, useGPU=-1, framerate=FRAMERATE, cache=None, speed=False, sleep=False, **kwargs):
        self.filter_and_interpolate_pose(*args, min_time=min_time, max_time=max_time, stride=stride, useGPU=useGPU, framerate=framerate, cache=cache, **kwargs)

        if speed:
            logger.debug("Computing speed features on dataset of shape %s", self.pose_boxcar.shape)
            self.compute_speed(
                self.pose_boxcar, min_time=min_time, max_time=max_time,
                stride=stride,
                framerate=framerate, cache=cache, useGPU=useGPU
            )

        if sleep:
            pass

    def load_and_process_data(self, *args, min_time=MIN_TIME, max_time=MAX_TIME, stride=1, cache=None, files=None, load_behavior=True, load_deg=True, write_only=False, **kwargs):
        """
        Loads centroid, pose, deg and behavior datasets of this fly
        """
        if files is not None:
            self.datasetnames=[os.path.splitext(os.path.basename(file))[0] for file in files]
            self.ids=self.make_ids(self.datasetnames)

        before=time.time()
        self.load_data(min_time=min_time, max_time=max_time, stride=stride, cache=cache, files=files, load_behavior=load_behavior, load_deg=load_deg, write_only=write_only)
        after=time.time()
        print(f"{after-before} seconds loading data")

        if self.dt is None:
            logger.warning("No centroid data for %s__%s", self.experiment, str(self.identity).zfill(2))
        if self.pose is None:
            logger.warning("No pose data for %s__%s", self.experiment, str(self.identity).zfill(2))

        # processing happens with stride = 1 and original framerate (150)
        before=time.time()
        self.process_data(*args, min_time=min_time, max_time=max_time, stride=stride, write_only=write_only, cache=cache, **kwargs)
        after=time.time()
        print(f"{after-before} seconds processing data")

        self.apply_stride_all(stride=stride)
        self.stride=stride

    def apply_stride_all(self, stride=1):

        for df_name in ["pose", "pose_boxcar", "pose_speed", "pose_speed_boxcar"]:
            df=getattr(self, df_name)
            if df is None:
                continue
            setattr(self, df_name, self.apply_stride(df, stride=stride))


    @staticmethod
    def apply_stride(df, stride):
        out=[]
        for id, df_single_animal in df.groupby("id"):
            out.append(df_single_animal.loc[df_single_animal["frame_number"] % stride == 0])
        out=pd.concat(out, axis=0)
        return out


    def load_store_index(self, cache=None):

        if cache is not None:
            path = os.path.join(cache, f"{self.experiment}_store_index.pkl")
            ret, self.store_index = restore_cache(path)
            ret=False
            if ret:
                return
        before=time.time()
        if self.store_index is None:
            self.store=VideoCapture(self.store_path, 1)
            if self.chunks is not None:
                self.store_index=pd.concat([pd.DataFrame(self.store._index.get_chunk_metadata(chunk)) for chunk in self.chunks], axis=0)
            else:
                self.store_index=pd.DataFrame(self.store._index.get_all_metadata())
            self.store_index["frame_time"]/=1000
            self.store.release()
        after=time.time()

        if cache is not None:
            save_cache(path, self.store_index)

        logger.debug("Loading store index took %s seconds", after-before)


    def load_data(
        self, *args, identity=None, min_time=None, max_time=None, stride=1, n_jobs=1, cache=None, files=None,
        load_behavior=True, load_deg=True, write_only=False,
        **kwargs
    ):
        self.load_store_index(cache=cache)
        if identity is None:
            identities=[self.identity]
            identity=self.identity

            if self.identity is None:
                identities=[int(id.split("|")[1]) for id in self.ids]
        else:
            identities=[identity]

        logger.info("Loading centroid data")
        try:
            self.load_centroid_data(
                *args, identity=identity, min_time=min_time, max_time=max_time, n_jobs=n_jobs,
                stride=stride, verbose=False,
                cache=None,
                reference_hour=np.nan,
                identity_table=self.identity_table,
                roi_0_table=self.roi_0_table,
                **kwargs)
        except AssertionError as error:
            logger.error(error)
            logger.error(
                """Cannot load validated data!
                If your results rely on data being validated,
                you cannot use the output of this program until you fix the issue
                """
            )
            self.load_centroid_data(
                *args, identity=identity, min_time=min_time, max_time=max_time, n_jobs=n_jobs,
                stride=stride, verbose=False,
                cache=None,
                reference_hour=np.nan,
                identity_table="IDENTITY",
                roi_0_table="ROI_0",
                **kwargs)

        logger.info("Loading pose data")
        for ident in identities:
            self.load_pose_data(*args, identity=ident, min_time=min_time, max_time=max_time, verbose=False, cache=cache, files=files, write_only=write_only, **kwargs)
        logger.info("Loading DEG data")
        if load_deg:
            self.load_deg_data(*args, identity=identity, ground_truth=True, stride=stride, verbose=False, cache=cache, **kwargs)

        if load_behavior:
            logger.info("Loading behavior data")
            self.load_behavior_data(self.experiment, identity, min_time=min_time, max_time=max_time)

    def make_movie(self, ts=None, frame_numbers=None, **kwargs):
        return make_pose_movie(self.basedir, self.dt_with_pose, ts=ts, frame_numbres=frame_numbers, **kwargs)


    def load_centroid_data(self, *args, identity=None, min_time=MIN_TIME, max_time=MAX_TIME, stride=1, reference_hour=np.nan, cache=None, **kwargs):

        if cache is not None:
            logger.warning("Supplied cache will be ignored. ethoscopy cache will be used instead")

        if identity is None:
            identity=self.identity

        dt, meta_info=load_centroid_data(
            *args, experiment=self.experiment, identity=identity, reference_hour=reference_hour,
            min_time=min_time, max_time=max_time, stride=1, **kwargs
        )
        if dt is not None:
            dt.reset_index(inplace=True)
            if stride != 1:
                dt=dt.iloc[::stride]

            dt["frame_number"]=dt["frame_number"].astype(np.int32)
            dt["id"]=pd.Categorical(dt["id"])
            self.dt=dt
            self.meta_info=meta_info[0]
        else:
            logger.warning("No centroid data database found for %s %s", self.experiment, identity)


    def draw_videos(self, index):
        for i, row in index.iterrows():
            draw_video_row(self, row["identity"], i, row, output=self.experiment + "_videos")

    def get_pose_file_h5py(self, pose_name="filter_rle-jump"):
        animal=self.experiment + "__" + str(self.identity).zfill(2)
        pose_file=os.path.join(
            self.basedir, "motionmapper",
            str(self.identity).zfill(2),
            f"pose_{pose_name}",
            animal,
            animal + ".h5"
        )
        return pose_file
    
    def manage_backup_copies(self, file, fail=False):
        if validate_h5py_file(file):
            shutil.copy(file, f"{file}.backup")
            return file
        else:
            if not fail and os.path.exists(f"{file}.backup"):
                shutil.copy(f"{file}.backup", file)
                return self.manage_backup_copies(file, fail=True)
            else:
                raise OSError(f"{file} is corrupted")

def validate_h5py_file(file):
    try:
        with h5py.File(file, 'r') as f:
            print(f"File {file} integrity passed")
            pass
        return True
    except Exception as e:
        print(f"File {file} integrity check failed: {e}")
        return False
