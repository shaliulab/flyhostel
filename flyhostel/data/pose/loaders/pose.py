import logging
import itertools
import os.path
from abc import abstractmethod

import h5py
import numpy as np
import pandas as pd

from flyhostel.data.pose.h5py import load_pose_data_compiled
from flyhostel.utils import restore_cache, save_cache
from flyhostel.data.pose.filters import filter_pose, arr2df
from flyhostel.data.pose.gpu_filters import filter_pose_df
from flyhostel.data.pose.constants import bodyparts as BODYPARTS
from flyhostel.data.pose.constants import MIN_TIME, MAX_TIME
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.constants import PARTITION_SIZE


logger=logging.getLogger(__name__)

bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in BODYPARTS]))
bodyparts_speed=list(itertools.chain(*[[bp + "_speed"] for bp in BODYPARTS]))
MOTIONMAPPER_DATA=os.environ["MOTIONMAPPER_DATA"]

try:
    import cupy as cp
    useGPU=True
except:
    cp=None
    useGPU=False
    logger.debug("Cannot load cupy")

class PoseLoader:

    def __init__(self, *args, **kwargs):

        self.experiment=None
        self.ids=None
        self.pose=None
        self.datasetnames=[]
        self.pose_source=None
        self.store_index=None

        self.pose_raw=None
        self.pose_jumps=None
        self.pose_filters=None
        self.pose_interpolated=None
        self.pose_speed=None
        self.pose_speed_max=None
        self.pose_annotated=None
        self.pose_speed_boxcar=None
        self.pose_boxcar=None
        self.meta_info=None


        self.window_size_seconds=0.5
        self.min_window_size=40
        self.meta_pose={}
        self.min_supporting_points=3

        super(PoseLoader, self).__init__(*args, **kwargs)
    

    @abstractmethod
    def load_store_index(self, cache):
        raise NotImplementedError()

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
            animals=[animal for animal in self.datasetnames if animal.endswith(str(identity).zfill(2))]
            ids=[ident for ident in self.ids if ident.endswith(str(identity).zfill(2))]
            if len(animals)==0 or len(ids)==0:
                logger.error("identity %s not available in POSE_DATA", identity)
            out=load_pose_data_compiled(animals, ids, self.lq_thresh, stride=stride, files=files)

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

        if len(pose) > 0:
            
            if min_time > float("-inf") or max_time < float("+inf"):
                t=self.store_index["frame_time"]+self.meta_info["t_after_ref"]
                min_fn=self.store_index["frame_number"].iloc[
                    np.argmax(t>=min_time)
                ]
                max_fn=self.store_index["frame_number"].iloc[
                    -(np.argmax(t[::-1]<max_time)-1)
                ]
                pose=pose.loc[
                    (pose["t"] >= min_fn) & (pose["t"] < max_fn)
                ]
    
        return pose


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
        assert len(np.unique(pose_out["id"]))==1, "Exported dataset contains more than 1 id. Please filter it or provide an id to .export()"
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