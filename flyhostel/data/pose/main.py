import time
from abc import ABC, abstractmethod
import sqlite3
import glob
import time
import pickle
import os.path
import itertools
import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import codetiming
import matplotlib.pyplot as plt
import h5py
from tqdm.auto import tqdm
try:
    import cupy as cp
    useGPU=True
except:
    cp=None
    useGPU=False
    logger.debug("Cannot load cupy")

from flyhostel.data.interactions.centroids import (
    load_centroid_data,
    flyhostel_sleep_annotation,
    time_window_length,
)
from flyhostel.data.interactions.distance import (
    compute_distance_matrix,
    compute_distance_matrix_bodyparts
)
from flyhostel.data.pose.h5py import (
    load_pose_data_compiled,
)
from flyhostel.data.pose.pose import FilterPose
from flyhostel.data.interactions.bouts import annotate_interaction_bouts, compute_bouts_, DEFAULT_STRIDE
from flyhostel.data.bodyparts import bodyparts as BODYPARTS
from flyhostel.utils import load_roi_width, load_metadata_prop, restore_cache, save_cache
from flyhostel.data.pose.movie_old import make_pose_movie
from flyhostel.data.pose.gpu_filters import filter_pose_df, PARTITION_SIZE
from flyhostel.data.pose.constants import MIN_TIME, MAX_TIME
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from imgstore.interface import VideoCapture
from flyhostel.data.pose.ethogram_utils import annotate_bout_duration, annotate_bouts
from flyhostel.data.pose.distances import compute_distance_features_pairs
from flyhostel.data.pose.wavelets import WaveletLoader
from flyhostel.data.interactions.main import InteractionDetector
bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in BODYPARTS]))
bodyparts_speed=list(itertools.chain(*[[bp + "_speed"] for bp in BODYPARTS]))
from flyhostel.data.pose.constants import framerate
from motionmapperpy import setRunParameters


def dunder_to_slash(experiment):
    tokens = experiment.split("_")
    return tokens[0] + "/" + tokens[1] + "/" + "_".join(tokens[2:4])


# keep only interactions where the distance between animals is max mm_max mm
roi_width_mm=60
dist_max_mm=4

MOTIONMAPPER_DATA=os.environ["MOTIONMAPPER_DATA"]
POSE_DATA=os.environ["POSE_DATA"]


from flyhostel.data.pose.filters import filter_pose, filter_pose_far_from_median, interpolate_pose, arr2df
from flyhostel.data.pose.sleap import draw_video_row

from flyhostel.data.interactions.mmpy_umap import UMAPLoader
from flyhostel.data.deg import DEGLoader
from flyhostel.data.pose.video_crosser import CrossVideo

class PEDetector(ABC):
    """
    Given an estimate of the pose of multiple animals, stored in a pd.DataFrame
    with columns id, frame_number and bp_x bp_y bp_is_interpolated where bp is any bp,
    detects proboscis extension bouts, defined as episodes where the proboscis acquires a non-zero distance
    from the head when looking at the top view of the fly

    How to use:

    Run detect_proboscis_extension() to access the pose dataframe and run the whole analysis
    Present filters 
    * position of proboscis canot be interpolated (rows where bp_is_interpolated is False are discarded)
    * distance from proboscis to head in any frame cannot be > 20 pixels. Such frames are discarded prior to the detection of bouts
    """

    video_chunksize=CHUNKSIZE
    video_framerate=FRAMERATE
    MAX_DISTANCE_HEAD_PROB_PIXELS=20

    def __init__(self, *args, **kwargs):

        self.experiment=None
        self.pose=None
        self.ids=None
        self.roi_width=None
        self.dt=None
        self.pe_df=None
        super(PEDetector, self).__init__(*args, **kwargs)


    def compute_prob_head_distance(self, pose, min_distance=1):

        x = pose["proboscis_x"]-pose["head_x"]
        y = pose["proboscis_y"]-pose["head_y"]
        pose["frame_idx"]=pose["frame_number"]%self.video_chunksize
        pose["chunk"]=pose["frame_number"]//self.video_chunksize

        head_prob_pose = pose[["frame_number", "chunk", "frame_idx", "id", "t", "head_x", "head_y", "proboscis_x", "proboscis_y", "head_likelihood", "proboscis_likelihood", "proboscis_is_interpolated"]]

        dist = np.sqrt((np.array([x,y])**2).sum(axis=0))
        dist_df=pd.DataFrame({"frame_number": pose["frame_number"], "distance": dist})
        dist_df=dist_df.merge(head_prob_pose, on="frame_number")
        dist_df=dist_df.loc[dist_df["distance"] > min_distance]
        dist_df["mistrack"]=False
        dist_df.loc[dist_df["distance"] > self.MAX_DISTANCE_HEAD_PROB_PIXELS, "mistrack"]=True
        dist_df=dist_df.loc[~dist_df["mistrack"]]
        dist_df=dist_df.loc[~dist_df["proboscis_is_interpolated"]]
        return dist_df
    

    def compute_bouts(self, pos_events_df, stride=DEFAULT_STRIDE):
        """
        Compute length and duration of bouts of a positive event
        
        Args:

            pos_events_df (pd.DataFrame): Dataset of instances of an event that can last an undefinite amount of time
            Each row represents one frame where the event is present
            A frame_number column must be present


        """
        assert "frame_number" in pos_events_df.columns, f"Please provide a frame_numbe column"
        encoding_df, self.stride=compute_bouts_(pos_events_df)
        if self.stride != stride:
            print(f"Stride of dataset {self.experiment} is {self.stride}")

        encoding_df["chunk"]=encoding_df["frame_number"]//self.video_chunksize
        encoding_df["frame_idx"]=encoding_df["frame_number"]%self.video_chunksize
        return encoding_df


    def analyse_bouts(self, dist_df, encoding_df):
        intervals=[]
        dynamics_df=[]
        for i, row in encoding_df.iterrows():
            fn_start = row["frame_number"]-self.stride
            fn_end = row["frame_number"] + (row["length"]+1)*self.stride
            df_block = dist_df.loc[(dist_df["frame_number"]>=fn_start) & (dist_df["frame_number"]<fn_end)]
                
            distance=df_block["distance"]
            likelihood=df_block["proboscis_likelihood"]
            n_points=df_block.shape[0]
            
            dist_ts = distance.iloc[[0, n_points//2, n_points-1]]
            lk_ts = likelihood.iloc[[0, n_points//2, n_points-1]]
            dynamics_df.append(np.concatenate([dist_ts, lk_ts]))

            max_dist = distance.max()
            mean_lk =  likelihood.mean()
            min_lk =  likelihood.min()
            max_lk =  likelihood.max()
            lk_at_max_dist = likelihood.iloc[distance.argmax()]
            
            
            interval=(fn_start, fn_end, max_dist, mean_lk, min_lk, max_lk, lk_at_max_dist)
            intervals.append(interval)
            
        intervals_df = pd.DataFrame.from_records(intervals, columns=["start", "end", "max_dist", "mean_lk", "min_lk", "max_lk", "lk_at_max_dist"])
        intervals_df["chunk"]=intervals_df["start"]//self.video_chunksize
        intervals_df["frame_idx_start"]=intervals_df["start"]%self.video_chunksize
        intervals_df["frame_idx_end"]=intervals_df["end"]%self.video_chunksize
        intervals_df["duration"]=(intervals_df["end"]-intervals_df["start"])/self.video_framerate
        intervals_df=pd.concat([intervals_df, pd.DataFrame(dynamics_df)], axis=1)
        return intervals_df


    def detect_proboscis_extension(self, pose):
        all_dfs=[]
        for id, pose_d in pose.groupby("id"):
            if not id in self.ids:
                print(f"No pose data is available for id {id}")
                continue


            dist_df = self.compute_prob_head_distance(pose_d)
            bouts_df = self.compute_bouts(dist_df)
            pe_df = self.analyse_bouts(dist_df, bouts_df)
            pe_df["id"]=id
            pe_df["experiment"]=self.experiment
            pe_df["frame_number"]=pe_df["start"]

            # where?
            dt=self.dt.loc[self.dt["id"] == id]
            centroid_data=pd.DataFrame({"frame_number": dt["frame_number"].astype(np.int64), "x": self.roi_width*dt["x"], "y": self.roi_width*dt["y"]})
            pe_df["frame_number"]=pe_df["frame_number"].astype(np.int64)

            pe_df=pd.merge_asof(pe_df, centroid_data, on="frame_number")
            all_dfs.append(pe_df)

        if len(all_dfs) == 0:
            return

        pe_df=pd.concat(all_dfs)

        # when? (it's the same for all ids so it can be done outside of the loop once for all)
        pe_df=pe_df.merge(self.pose[["id", "frame_number", "t"]], on=["id", "frame_number"])
        pe_df["chunk"]=pe_df["chunk"].astype(np.int64)
        pe_df["frame_idx_start"]=pe_df["frame_idx_start"].astype(np.int64)

        self.pe_df=pe_df
        return pe_df


class BehaviorLoader():

    def __init__(self, *args, **kwargs):
        self.behavior=None
        self.pose=None
        super(BehaviorLoader, self).__init__(*args, **kwargs)

    def load_behavior_data(self, experiment, identity, pose=None, interpolate_frames=0, cache=None):

        if cache is not None:
            path = os.path.join(cache, f"{experiment}__{str(identity).zfill(2)}_behavior.pkl")
            ret, self.behavior = restore_cache(path)
            if ret:
                return

        tokens=experiment.split("_")
        feather_path=os.path.join(
            os.environ["FLYHOSTEL_VIDEOS"],
            tokens[0],
            tokens[1],
            "_".join(tokens[2:4]),
            "motionmapper",
            str(identity).zfill(2),
            f"{experiment}__{str(identity).zfill(2)}.feather"
        )
        if os.path.exists(feather_path):

            dt=pd.read_feather(feather_path)[["id", "frame_number", "behavior", "score"]]
    
            if interpolate_frames>0:
                dt_t_complete=[]
                for i in tqdm(range(interpolate_frames), desc="filling wavelet gaps"):
                    df=dt.copy()
                    df["frame_number"]+=i
                    dt_t_complete.append(df)
                
                dt=pd.concat(dt_t_complete, axis=0)
                del dt_t_complete
                dt.sort_values(["id", "frame_number"], inplace=True)
                
            wavelet_downsample=setRunParameters().wavelet_downsample
            fps=framerate//wavelet_downsample

            if pose is None:
                logger.warning("Pose not found. Please provide a pose dataset, or refining of behavior will be disabled")
            else:
                before=time.time()
                logger.debug("Refining behavior")
                pose["chunk"]=pose["frame_number"]//CHUNKSIZE
                pose["frame_idx"]=pose["frame_number"]%CHUNKSIZE
                
                pose=compute_distance_features_pairs(pose, pairs=[("head", "proboscis"), ])
                distances=[column for column in pose.columns if "distance" in column]

                dt=pose[["id", "t", "frame_number", "chunk", "frame_idx", "identity"] + distances].merge(
                    dt[["id", "frame_number", "behavior", "score"]], how="right", on=["id", "frame_number"]
                )
                
                dt.loc[(dt["head_proboscis_distance"] < 0.01) & (dt["behavior"]=="pe_inactive"), "behavior"]="inactive"
                after=time.time()
                logger.debug("Refining behavior took %s seconds", after-before)



        else:
            logger.warning("%s does not exist", feather_path)
            return
    
        dt=annotate_bouts(dt, "behavior")
        dt=annotate_bout_duration(dt, fps=fps)
        self.behavior=dt


        if cache is not None:
            save_cache(path, self.behavior)



class FlyHostelLoader(PEDetector, CrossVideo, WaveletLoader, InteractionDetector, BehaviorLoader, DEGLoader, FilterPose):
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

    def __init__(self, experiment, *args, identity=None, lq_thresh=1, roi_width_mm=roi_width_mm, dist_max_mm=dist_max_mm, n_jobs=1, pose_source="compiled", chunks=None, **kwargs):
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
        self.dist_max_mm=dist_max_mm
        self.meta_pose={}

        # self.index_pandas = []
        # self.h5s_pandas = []
        self.pose=None
    
        self.dt=None
        self.neighbors_df=None
        self.dt_sleep=None
        self.dt_with_pose=None
        self.out=None
        self.out_filtered=None
        self.pose_and_centroid=None
        self.m3=None
        self.dt_sleep_raw=None
        self.dt_with_pose_nns=None
        self.distances=None
        self.distances_sleep=None
        self.interactions_sleep=None
        self.rejections=None
        self.dt_sleep_2fps=None
    
        self.pose_raw=None
        self.pose_jumps=None
        self.pose_filters=None
        self.pose_interpolated=None
        self.pose_speed=None
        self.pose_speed_max=None
        self.pose_annotated=None
        self.pose_speed_boxcar=None
        self.pose_boxcar_2=None
        self.pose_boxcar=None
        

        
        # for index in self.index_pandas:
        #     index["predictions"]=[e.decode().replace(".h5", ".slp") for e in index["files"]]
        #     index["video"]=[e.decode().replace(".predictions.h5", "") for e in index["files"]]


        self.meta_info=None
        self.window_size_seconds=0.5
        self.min_window_size=40
        self.min_supporting_points=3

    def __str__(self):
        return f"{self.experiment}__{str(self.identity).zfill(2)}"


    @property
    def pose_median(self):
        return self.pose_filters["nanmedian"]


    def list_ids(self):
        return np.unique(self.pose["id"])

    def export(self, pose=None, dest_folder=None, bodyparts=BODYPARTS, id=None):
        f"""
        Export the pose to an .h5 dataset compatible with motionmapper, bsoid, etc
        If the pose dataset is not specified, it will use pose_interpolated by default
        If the dest_folder is not specified, it will use {MOTIONMAPPER_DATA} by default
        """

        assert len(pose["id"].unique())==1
        bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
        node_names=[bp.encode() for bp in bodyparts]

        if pose is None:
            assert self.pose_interpolated is not None
            if id is not None:
                pose_out=self.pose_interpolated.loc[self.pose_interpolated["id"]==id]
            else:
                pose_out=self.pose_interpolated
        else:
            if id is not None:
                pose_out=pose.loc[pose["id"]==id]
            else:
                pose_out=pose

        frame_number=pose_out["frame_number"]

        id=pose_out["id"].iloc[0]
        assert len(np.unique(pose_out["id"]))==1, f"Exported dataset contains more than 1 id. Please filter it or provide an id to .export()"
        identity = int(id.split("|")[1])

        input_array=pose_out[bodyparts_xy].values
        N, m=input_array.shape
        number_of_bodyparts = m // 2
        # Reshape and transpose the array
        reshaped_array = input_array.reshape(N, number_of_bodyparts, 2).transpose(2, 1, 0)
        reshaped_array=reshaped_array[np.newaxis, :]

        assert len(frame_number) == reshaped_array.shape[3]

        files=sorted(self.meta_pose[id]["files"])
        
        key=f"{self.experiment}__{str(identity).zfill(2)}"

        if dest_folder is None:
            dest_folder=MOTIONMAPPER_DATA

        folder=f"{dest_folder}/{key}"
        os.makedirs(folder, exist_ok=True)
        filename=f"{folder}/{key}.h5"
        print(f"Saving to --> {filename}")
        f=h5py.File(filename, "w")
        f.create_dataset("tracks", data=reshaped_array)
        f.create_dataset("node_names", data=node_names)
        f.create_dataset("files", data=files)
        # f.create_dataset("frame_number", data=frame_number)
        f.close()
        return


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
    

    def load_and_process_data(self, *args, min_time=MIN_TIME, max_time=MAX_TIME, stride=1, cache=None, bodyparts=BODYPARTS, files=None, **kwargs):
        if files is not None:
            self.datasetnames=[os.path.splitext(os.path.basename(file))[0] for file in files]
            self.ids=self.make_ids(self.datasetnames)

        self.load_data(min_time=min_time, max_time=max_time, stride=stride, cache=cache, files=files)
        if self.dt is None:
            logger.warning("No centroid data for %s__%s", self.experiment, str(self.identity).zfill(2))
        if self.pose is None:
            logger.warning("No pose data for %s__%s", self.experiment, str(self.identity).zfill(2))

        # processing happens with stride = 1 and original framerate (150)
        self.process_data(*args, min_time=min_time, max_time=max_time, stride=stride, bodyparts=bodyparts, cache=cache, **kwargs)
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
    
    def process_sleep(self, min_time, max_time, stride, cache):

        if cache is not None:
            path = os.path.join(cache, f"{self.experiment}__{min_time}_{max_time}_{stride}_sleep_data.pkl")
            ret, out = restore_cache(path)
            self.dt_sleep = out
            return

        self.dt_sleep = self.annotate_sleep(self.dt)
        
        if cache is not None:
            save_cache(path, (self.dt_sleep))

    def boxcar_filter(self, pose, features, bodyparts=BODYPARTS, framerate=150):

        filtered_pose_arr, _ = filter_pose(
            "nanmean", pose, bodyparts,
            window_size=int(self.window_size_seconds*framerate),
            min_window_size=self.min_window_size,
            min_supporting_points=self.min_supporting_points,
            features=features
        )
        filtered_pose=arr2df(pose, filtered_pose_arr, bodyparts, features=features)
        return filtered_pose


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
    
    def load_pose_data(self, identity, min_time=-float("inf"), max_time=float("inf"), time_system="zt", stride=1, cache=None, verbose=False, files=None):
        
        if min_time>=max_time:
            logger.warning("Passed time interval (%s - %s) is meaningless")

        self.load_store_index(cache=cache)
        ret=False
        pose=None
    
        if cache is not None:
            path = f"{cache}/{self.experiment}__{str(identity).zfill(2)}_{stride}_pose_data.pkl"
            ret, out=restore_cache(path)

            if ret:
                (pose, meta_pose)=out

         
        if not ret:
            if self.pose_source == "processed":
                raise NotImplementedError()
                # out=load_pose_data_processed(min_time, max_time, time_system, self.datasetnames, self.ids, self.lq_thresh)
            elif self.pose_source == "compiled":
                datasets=[dataset for dataset in self.datasetnames if dataset.endswith(str(identity).zfill(2))]
                ids=[ident for ident in self.ids if ident.endswith(str(identity).zfill(2))]
                if len(datasets)==0 or len(ids)==0:
                    logger.error("identity %s not available in POSE_DATA", identity)
                out=load_pose_data_compiled(datasets, ids, self.lq_thresh, stride=stride, files=files)
            else:
                raise Exception("source must be processed or compiled")

            if out is not None:
                pose, _, index_pandas=out
                if len(pose)==0:
                    return None

                assert len(index_pandas)==1
                
                meta_pose={"files": [file.decode() for file in index_pandas[0]["files"].unique()]}

                pose2=[]
                for d in pose:
                    pose2.append(d.reset_index())
                pose=pd.concat(pose2, axis=0)
                
                # one for each animal in the experiment
                corrupt_pose=np.bitwise_or(pd.isna(pose["thorax_x"]), pd.isna(pose["frame_number"]))

                pose.loc[corrupt_pose, bodyparts_xy + [bp + "_likelihood" for bp in BODYPARTS]]=np.nan
                pose["frame_number"]=pose["frame_number"].astype(np.int32)
                pose.sort_values(["id", "frame_number"], inplace=True)
                pose.reset_index(inplace=True)
                pose["id"]=pd.Categorical(pose["id"])
                pose["identity"]=identity
                pose=self.filter_pose_by_identity(pose, identity)
                pose=self.filter_pose_by_time(pose=pose, min_time=min_time, max_time=max_time)
                if cache is not None:
                    save_cache(path, (pose, meta_pose))
                        

        if self.pose is None:
            self.pose=pose
        else:
            self.pose=pd.concat([self.pose, pose], axis=0)

        id=pose["id"].iloc[0]
        self.meta_pose[id]=meta_pose
        self.filter_pose_by_time(min_time, max_time)
        return None


    @staticmethod
    def filter_pose_by_identity(pose, identity):
        return pose.loc[pose["identity"]==identity]

            
    def filter_pose_by_time(self, min_time, max_time, pose=None):
        if pose is None:
            pose=self.pose

        if self.dt is not None and len(pose) > 0:
            
            
            # pose=self.annotate_time_in_dataset(pose, self.store_index, "frame_time", self.meta_info["t_after_ref"])
            
            if min_time > float("-inf") or max_time < float("+inf"):
                t=self.store_index["frame_time"]+self.meta_info["t_after_ref"]
                min_fn=self._store_index["frame_number"].iloc[
                    np.argmax(t>=min_time)
                ]
                max_fn=self._store_index["frame_number"].iloc[
                    -(np.argmax(t[::-1]<max_time)-1)
                ]
                pose=pose.loc[
                    (pose["t"] >= min_fn) & (pose["t"] < max_fn)
                ]
    
        return pose


    def make_movie(self, ts=None, frame_numbers=None, **kwargs):
        return make_pose_movie(self.basedir, self.dt_with_pose, ts=ts, frame_numbres=frame_numbers, **kwargs)


    def annotate_sleep(self, dt):
        """
        Annotate sleep on a downsampled timeseries (to 2 Hz)
        and bring annotation back to original framerate by interpolating
        """
        annotation_input_columns=["id", "t", "xy_dist_log10x1000", "frame_number", "x", "y", "w", "h", "phi"]
        annotation_output_columns=["id", "frame_number", "asleep", "moving", "max_velocity"]

        dt_to_annotate = dt[annotation_input_columns]
        dt_sleep = dt_to_annotate.groupby(dt_to_annotate["id"]).apply(flyhostel_sleep_annotation).reset_index()
        
        # a row full of nans is produced when no data is available for one fly in one time window
        # they need to be removed so that frame number can be integer
        dt_sleep=dt_sleep.loc[~np.isnan(dt_sleep["frame_number"])]
        dt_sleep["frame_number"]=dt_sleep["frame_number"].astype(np.int32)
                
        dt_sleep=dt_sleep[annotation_output_columns].sort_values(["frame_number", "id"])
        dt_to_merge = dt[annotation_input_columns].sort_values(["frame_number", "id"])
        
        dt_sleep=pd.merge_asof(
            dt_to_merge,
            dt_sleep,
            on="frame_number",
            by="id",
            direction="backward",
            tolerance=time_window_length*self.framerate
        )
        self.dt_sleep_2fps=dt_sleep.loc[dt_sleep["frame_number"] % (self.framerate//2) == 0]
        return dt_sleep



    def annotate_interactions(self, dist_max, min_bout):
        def keep_at_least_one_fly_sleep(dt):
            dt = dt.loc[
                dt["asleep1"] | dt["asleep2"]
            ]
            return dt
        def keep_max_one_fly_asleep(dt):
            dt.loc[
                ~(dt["asleep1"] & dt["asleep2"])
            ]
            return dt

        self.interactions_sleep=annotate_interaction_bouts(self.distances_sleep, dist_max, min_bout)
        if self.interactions_sleep is not None:
            self.rejections=keep_max_one_fly_asleep(keep_at_least_one_fly_sleep(self.interactions_sleep))
        else:
            self.rejections=None

    
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


    def merge_with_sleep_status(self, dt_sleep, distances):

        dt_sleep_to_merge=dt_sleep.reset_index()[["id", "asleep", "frame_number"]]
        distances_sleep=pd.merge_asof(
            distances,
            dt_sleep_to_merge.rename({"asleep": "asleep1"}, axis=1),
            on="frame_number", by=["id"], direction="backward", tolerance=time_window_length*self.framerate
        ).rename({"id": "id1"}, axis=1)

        distances_sleep=pd.merge_asof(
            distances_sleep.rename({"nn": "id"}, axis=1),
            dt_sleep_to_merge.rename({"asleep": "asleep2"}, axis=1),
            on="frame_number", by=["id"], direction="backward", tolerance=time_window_length*self.framerate
        ).rename({"id": "id2"}, axis=1)

        return distances_sleep


    def main(self):
        self.integrate(self.dt, self.pose_boxcar)
        self.make_movie(ts=np.arange(60000, 60300, 1))
        self.annotate_interactions(n_jobs=self.n_jobs)


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
    def load_experiments(pattern="*"):
        pickle_files = sorted(glob.glob(f"{MOTIONMAPPER_DATA}/{pattern}.pkl"))
        experiments = [os.path.splitext(os.path.basename(path))[0] for path in pickle_files]
        return pickle_files, experiments

    
    @staticmethod
    def make_ids(datasetnames):
        identities = [
            d[:26] +  "|" + d[-2:]
            for d in datasetnames
        ]
        return identities


    @staticmethod
    def compute_pose_distance(pose):
        diff=np.diff(pose[bodyparts_xy], axis=0)
        diff=np.vstack([[np.nan,] * diff.shape[1], diff])
        N=diff.shape[0]
        m=diff.shape[1]//2
        diff=diff.reshape(N, m, 2).transpose(2, 1, 0)
        
        nansum=np.nansum(diff**2, axis=0)
        distance=np.sqrt(nansum)
        bps = bodyparts_xy[::2]
        columns = [bp.replace("_x", "_speed") for bp in bps]
        for i, column in enumerate(columns):
            pose[column]=distance[i,:]
        return pose

    def compute_speed(self, pose, min_time=MIN_TIME, max_time=MAX_TIME, stride=1, framerate=FRAMERATE, cache=None, useGPU=-1):

        window_size=int(self.window_size_seconds*framerate)

        if cache is not None:
            path = f"{cache}/{self.experiment}_{min_time}_{max_time}_{stride}_pose_speed.pkl"
            ret, out = restore_cache(path)
            if ret:
                (self.pose_speed, self.pose_speed_boxcar)=out
                return

        self.pose_speed=self.compute_pose_distance(pose)

        if useGPU >= 0:
            self.pose_speed_boxcar=filter_pose_df(
                self.pose_speed, f=cp.mean, columns=bodyparts_speed,
                window_size=window_size,
                partition_size=PARTITION_SIZE,
                pad=True,
                download=True
            )
          
        else:
            self.pose_speed_boxcar=self.boxcar_filter(self.pose_speed, ["speed"], bodyparts=BODYPARTS, framerate=framerate)

        if cache is not None:
            save_cache(path, ((self.pose_speed, self.pose_speed_boxcar)))

    @staticmethod
    def annotate_pose(pose, behaviors):
        if behaviors is None:
            pose_annotated=pose.copy()
            pose_annotated["behavior"]="unknown"
        else:
            pose_annotated=pose.merge(behaviors[["frame_number", "t", "id", "behavior"]], on=["frame_number", "t", "id"], how="left")
            pose_annotated.loc[pd.isna(pose_annotated["behavior"]), "behavior"]="unknown"
        return pose_annotated

    
    def draw_videos(self, index):
        for i, row in index.iterrows():
            draw_video_row(self, row["identity"], i, row, output=self.experiment + "_videos")
