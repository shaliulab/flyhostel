import time
import glob
import time
import os.path
import logging

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
from flyhostel.data.pose.loaders.pose import PoseLoader
from flyhostel.data.pose.loaders.centroids import load_centroid_data
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.constants import ROI_WIDTH_MM
from flyhostel.utils.filesystem import FilesystemInterface


def dunder_to_slash(experiment):
    tokens = experiment.split("_")
    return tokens[0] + "/" + tokens[1] + "/" + "_".join(tokens[2:4])


# keep only interactions where the distance between animals is max mm_max mm

POSE_DATA=os.environ["POSE_DATA"]


from flyhostel.data.pose.sleap import draw_video_row
from flyhostel.data.deg import DEGLoader
from flyhostel.data.pose.video_crosser import CrossVideo


class FlyHostelLoader(CrossVideo, FilesystemInterface, PoseLoader, WaveletLoader, BehaviorLoader, DEGLoader, FilterPose):
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

    def __init__(self, experiment, identity, *args, lq_thresh=1, roi_width_mm=ROI_WIDTH_MM, n_jobs=1, chunks=None, **kwargs):
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
        self.datasetnames=self.load_datasetnames()
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
    

    def __str__(self):
        return f"{self.experiment}__{str(self.identity).zfill(2)}"


    def list_ids(self):
        return np.unique(self.pose["id"])




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


    def make_movie(self, ts=None, frame_numbers=None, **kwargs):
        return make_pose_movie(self.basedir, self.dt_with_pose, ts=ts, frame_numbres=frame_numbers, **kwargs)


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

    
    def draw_videos(self, index):
        for i, row in index.iterrows():
            draw_video_row(self, row["identity"], i, row, output=self.experiment + "_videos")
