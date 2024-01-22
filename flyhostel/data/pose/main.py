import time
import glob
import time
import pickle
import os.path
import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import codetiming

try:
    import cupy as cp
    useGPU=True
except:
    cp=None
    useGPU=False
    logger.debug("Cannot load cupy")

from flyhostel.data.interactions.centroids import (
    load_centroid_data,
)
from flyhostel.data.interactions.distance import (
    compute_distance_matrix_bodyparts
)

from flyhostel.data.pose.pose import FilterPose
from flyhostel.utils import load_roi_width, load_metadata_prop, restore_cache, save_cache
from flyhostel.data.pose.movie_old import make_pose_movie
from flyhostel.data.pose.constants import MIN_TIME, MAX_TIME
from imgstore.interface import VideoCapture
from flyhostel.data.pose.loaders.wavelets import WaveletLoader
from flyhostel.data.interactions.main import InteractionDetector
from flyhostel.data.pose.loaders.behavior import BehaviorLoader
from flyhostel.data.pose.loaders.pose import PoseLoader
from flyhostel.data.pose.constants import framerate as FRAMERATE


def dunder_to_slash(experiment):
    tokens = experiment.split("_")
    return tokens[0] + "/" + tokens[1] + "/" + "_".join(tokens[2:4])


# keep only interactions where the distance between animals is max mm_max mm
roi_width_mm=60

POSE_DATA=os.environ["POSE_DATA"]


from flyhostel.data.pose.sleap import draw_video_row
from flyhostel.data.deg import DEGLoader
from flyhostel.data.pose.video_crosser import CrossVideo


class FlyHostelLoader(CrossVideo, PoseLoader, WaveletLoader, InteractionDetector, BehaviorLoader, DEGLoader, FilterPose):
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

    def __init__(self, experiment, *args, identity=None, lq_thresh=1, roi_width_mm=roi_width_mm, n_jobs=1, pose_source="compiled", chunks=None, **kwargs):
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
        if identity is None:
            self.identity=None
        else:
            self.identity=int(identity)
        self.lq_thresh = lq_thresh
        self.n_jobs=n_jobs
        self.pose_source=pose_source
        self.datasetnames=self.load_datasetnames(source=pose_source)
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
        
        # because the arena circular
        self.roi_height = self.roi_width
        self.framerate= int(float(load_metadata_prop(dbfile=self.dbfile, prop="framerate")))
        self.roi_width_mm=roi_width_mm
        self.px_per_mm=self.roi_width/roi_width_mm

        # self.index_pandas = []
        # self.h5s_pandas = []
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
    

    def __str__(self):
        return f"{self.experiment}__{str(self.identity).zfill(2)}"


    @property
    def pose_median(self):
        return self.pose_filters["nanmedian"]


    def list_ids(self):
        return np.unique(self.pose["id"])



    def load_dbfile(self):
        dbfiles=glob.glob(self.basedir + "/FlyHostel*.db")
        assert len(dbfiles) == 1, f"{len(dbfiles)} dbfiles found in {self.basedir}: {' '.join(dbfiles)}"
        return dbfiles[0]


    def load_datasetnames(self, source):
        datasetnames = []
        if source == "processed":
            pickle_files, experiments = self.load_experiments(self.experiment + "*")
            settings={}
            for i, pickle_file in enumerate(pickle_files):
                with open(pickle_file, "rb") as handle:
                    params = pickle.load(handle)
                    settings[experiments[i]] = params
                    datasetnames.extend(params["animals"])
        elif source == "compiled":
            animals=os.listdir(POSE_DATA)
            datasetnames=sorted(list(filter(lambda animals: animals.startswith(self.experiment), animals)))
        if not datasetnames:
            logger.warning(f"No datasets starting with {self.experiment} found in {POSE_DATA}")
    
        else:
            if self.identity is not None:
                datasetnames=[datasetnames[self.identity-1]]
        return datasetnames
    


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
            # annotate sleep using centroid data
            self.process_sleep(min_time, max_time, stride, cache=cache)


    def load_and_process_data(self, *args, min_time=MIN_TIME, max_time=MAX_TIME, stride=1, cache=None, files=None, **kwargs):
        if files is not None:
            self.datasetnames=[os.path.splitext(os.path.basename(file))[0] for file in files]
            self.ids=self.make_ids(self.datasetnames)

        self.load_data(min_time=min_time, max_time=max_time, stride=stride, cache=cache, files=files)
        if self.dt is None:
            logger.warning("No centroid data for %s__%s", self.experiment, str(self.identity).zfill(2))
        if self.pose is None:
            logger.warning("No pose data for %s__%s", self.experiment, str(self.identity).zfill(2))

        # processing happens with stride = 1 and original framerate (150)
        self.process_data(*args, min_time=min_time, max_time=max_time, stride=stride, cache=cache, **kwargs)
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


    
    def process_sleep(self, min_time, max_time, stride, cache):

        if cache is not None:
            path = os.path.join(cache, f"{self.experiment}__{min_time}_{max_time}_{stride}_sleep_data.pkl")
            ret, out = restore_cache(path)
            self.dt_sleep = out
            return

        self.dt_sleep = self.annotate_sleep(self.dt)
        
        if cache is not None:
            save_cache(path, (self.dt_sleep))


    def load_store_index(self, cache=None):

        if cache is not None:
            path = os.path.join(cache, f"{self.experiment}_store_index.pkl")
            ret, self.store_index = restore_cache(path)
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


    def load_data(self, *args, identity=None, min_time=-float('inf'), max_time=+float('inf'), stride=1, n_jobs=1, cache=None, files=None, **kwargs):
        self.load_store_index(cache=cache)
        if identity is None:
            identities=[self.identity]
            identity=self.identity
            
            if self.identity is None:
                identities=[int(id.split("|")[1]) for id in self.ids]
        else:
            identities=[identity]

    
        logger.info("Loading centroid data")
        self.load_centroid_data(*args, identity=identity, min_time=min_time, max_time=max_time, n_jobs=n_jobs, stride=stride, verbose=False, cache=None, reference_hour=np.nan, **kwargs)
       
        logger.info("Loading pose data")
        for ident in identities:
            self.load_pose_data(*args, identity=ident, min_time=min_time, max_time=max_time, verbose=False, cache=cache, files=files, **kwargs)
        logger.info("Loading DEG data")
        self.load_deg_data(*args, identity=identity, ground_truth=True, stride=stride, verbose=False, cache=None, **kwargs)
        
        logger.info("Loading behavior data")
        self.load_behavior_data(self.experiment, identity, self.pose_boxcar, cache=cache)

    def load_fast(self, cache):
        self.load_centroid_data(identity=self.identity, min_time=-float('inf'), max_time=+float('inf'), stride=1, cache=None, reference_hour=np.nan)
        self.load_pose_data(identity=self.identity, min_time=-float('inf'), max_time=+float('inf'), stride=1, cache=cache)
        self.process_data(stride=1, cache=cache)
        self.load_deg_data(identity=self.identity, min_time=-float('inf'), max_time=+float('inf'), stride=1, cache=cache)
        self.load_behavior_data(self.experiment, identity=self.identity, pose=self.pose_boxcar, cache=cache)


    def load_deg_data(self, *args, min_time=-np.inf, max_time=+np.inf, stride=1, time_system="zt", ground_truth=True,  cache=None, **kwargs):
        
        if cache is not None:
            path = os.path.join(cache, f"{self.experiment}_{min_time}_{max_time}_{stride}_deg_data.pkl")
            ret, self.deg=restore_cache(path)
            if ret:
                return

        if ground_truth:
            self.load_deg_data_gt(*args, **kwargs)
        else:
            self.load_deg_data_prediction(*args, **kwargs)

        if self.deg is not None:
            self.load_store_index(cache=cache)
            # self.deg=self.annotate_time_in_dataset(self.deg, self.store_index, "frame_time", self.meta_info["t_after_ref"])
            # self.deg=self.deg.loc[
            #     (self.deg["t"] >= min_time) & (self.deg["t"] < max_time)
            # ]
            self.deg["behavior"].loc[pd.isna(self.deg["behavior"])]="unknown"
            self.deg=self.annotate_two_or_more_behavs_at_same_time(self.deg)

        if cache and self.deg is not None:
            save_cache(path, self.deg)


    @staticmethod
    def annotate_two_or_more_behavs_at_same_time(deg):
        """
        If more than 1 behavior is present in a given frame,
        create a new behavioral label by chaining said behaviors with +
        So for example, if the fly is walking and feeding at the same time,
        make it the behavior feed+walk
        """

        # Group by frame_number and id, join behaviors with '+', and reset index
        deg_group = deg.groupby(["id", "frame_number"])["behavior"].agg(lambda x: "+".join(sorted(list(set(x))))).reset_index()
        deg=deg_group.merge(
            deg.drop(["behavior"], axis=1).drop_duplicates(),
            on=["id", "frame_number"]
        )
        deg.sort_values(["id", "frame_number"],  inplace=True)
        after=time.time()
        return deg



    @staticmethod
    def annotate_time_in_dataset(dataset, index, t_column="t", t_after_ref=None):
        before=time.time()
        assert index is not None
        assert "frame_number" in dataset.columns

        if t_column in dataset.columns:
            dataset_without_t=dataset.drop(t_column, axis=1)
        else:
            dataset_without_t=dataset
        dataset=dataset_without_t.merge(index[["frame_number", t_column]], on=["frame_number"])
        if t_after_ref is not None and t_column == "frame_time":
            dataset["t"]=dataset[t_column]+t_after_ref
        after=time.time()
        logger.debug("annotate_time_in_dataset took %s seconds", after-before)
        return dataset




    def make_movie(self, ts=None, frame_numbers=None, **kwargs):
        return make_pose_movie(self.basedir, self.dt_with_pose, ts=ts, frame_numbres=frame_numbers, **kwargs)


    def compute_pairwise_distances_using_bodyparts_cpu(self, distances, pose, bodyparts, precision=100):
        """
        Compute distance between two closest bodyparts of two already close animals
        See compute_distance_matrix_bodyparts
        """

        if useGPU:
            impl = cp
        else:
            impl = np
    
        bodyparts_arr = np.array(bodyparts)
        distance_matrix, bp_pairs = compute_distance_matrix_bodyparts(distances, pose, impl, bodyparts, precision=precision)
        with codetiming.Timer(text="Computing closest body parts spent: {:.2f} seconds"):
            selected_bp_pairs = impl.argmin(distance_matrix, axis=1)
        min_distance=(distance_matrix[impl.arange(distance_matrix.shape[0]), selected_bp_pairs]/precision)

        if useGPU:
            min_distance=min_distance.get()
            bp_pairs=bp_pairs.get()
            selected_bp_pairs=selected_bp_pairs.get()
    
        pairs=np.stack([bodyparts_arr[bp_pairs[selected]] for selected in selected_bp_pairs], axis=0)
        min_distance_mm = min_distance / self.px_per_mm

        results=pd.DataFrame({"bp_distance_mm": min_distance_mm, "bp_distance": min_distance, "bp1": pairs[:, 0], "bp2": pairs[:, 1]})
        distances = pd.concat([distances.reset_index(drop=True), results], axis=1)
        distances["id"]=pd.Categorical(distances["id"])
        distances["nn"]=pd.Categorical(distances["nn"])
        distances.sort_values(["frame_number", "id", "nn"], inplace=True)
        return distances



    def load_centroid_data(self, *args, identity=None, min_time=MIN_TIME, max_time=MAX_TIME, stride=1, reference_hour=np.nan, cache=None, **kwargs):

        if cache is not None:
            logger.warning("Supplied cache will be ignored. ethoscopy cached will be used instead")

        if identity is None:
            identity=self.identity

        dt, meta_info=load_centroid_data(*args, experiment=self.experiment, identity=identity, reference_hour=reference_hour, min_time=min_time, max_time=max_time, stride=1, **kwargs)
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

    
    @staticmethod
    def make_ids(datasetnames):
        identities = [
            d[:26] +  "|" + d[-2:]
            for d in datasetnames
        ]
        return identities

    
    def draw_videos(self, index):
        for i, row in index.iterrows():
            draw_video_row(self, row["identity"], i, row, output=self.experiment + "_videos")
