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
from flyhostel.data.pose.constants import get_bodyparts
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.data.pose.constants import (
    MIN_TIME,
    MAX_TIME,
    PARTITION_SIZE,
    SQUARE_WIDTH,
    SQUARE_HEIGHT
)

BODYPARTS=get_bodyparts()

logger=logging.getLogger(__name__)

bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in BODYPARTS]))
bodyparts_speed=list(itertools.chain(*[[bp + "_speed"] for bp in BODYPARTS]))

try:
    import cupy as cp
    useGPU=True
except:
    cp=None
    useGPU=False
    logger.debug("Cannot load cupy")

def project_to_absolute_coords(pose, bodypart):
    pose[f"{bodypart}_x"]+=pose["center_x"]
    pose[f"{bodypart}_y"]+=pose["center_y"]
    return pose



def times_to_frame_slice(store_index, min_time=None, max_time=None):
    """
    Convert min_time/max_time to frame numbers suitable for slicing:
      - min_time -> first frame at or after min_time (>=)
      - max_time -> first frame strictly after max_time (>).
        If none exists, returns last_frame+1.
    If a time is None, returns None so you can do full_data[min_f:max_f] unchanged.

    Assumes store_index has columns 't' and 'frame_number'.
    Returns:
      (min_frame_number, max_frame_number, t_slice)
        where t_slice are the 't' values for frames in [min_frame_number:max_frame_number]
    """
    t_vals = store_index['t'].to_numpy()
    frames = store_index['frame_number'].to_numpy()
    n = len(frames)
    last_plus_one = (frames[-1] + 1) if n else 0

    def first_frame_at_or_after(time_val):
        if time_val is None:
            return None
        idx = np.searchsorted(t_vals, time_val, side='left')  # >=
        if idx < n:
            return int(frames[idx])
        return int(last_plus_one)

    def first_frame_after(time_val):
        if time_val is None:
            return None
        idx = np.searchsorted(t_vals, time_val, side='right')  # >
        if idx < n:
            return int(frames[idx])
        return int(last_plus_one)

    min_frame_number = first_frame_at_or_after(min_time)
    max_frame_number = first_frame_after(max_time)

    # collect the times corresponding to the slice
    if min_frame_number is None:
        start_idx = 0
    else:
        start_idx = np.searchsorted(frames, min_frame_number, side="left")

    if max_frame_number is None:
        end_idx = len(frames)
    else:
        end_idx = np.searchsorted(frames, max_frame_number, side="left")

    t_slice = t_vals[start_idx:end_idx]

    return min_frame_number, max_frame_number, t_slice





class PoseLoader:

    filters=None
    dt=None

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

        # pose where coordinates are absolute
        # i.e. using original frame coord systme
        # useful when computing distances or angles between diff flies
        self.pose_absolute=None
        # pose where x and y are specified in a single column per body part,
        # as a complex number where the real part is x and the imaginary is y 
        self.pose_complex=None

        self.window_size_seconds=0.5
        self.min_window_size=40
        self.meta_pose={}
        self.min_supporting_points=3

        super(PoseLoader, self).__init__(*args, **kwargs)
    

    @abstractmethod
    def load_store_index(self, cache):
        raise NotImplementedError()
    

    def load_pose_data_v2(self, min_time, max_time, absolute=False):
        """
        Used interactively, allows filtering by time
        Eventually should be merged with load_pose_data, I just dont wanna change load_pose_data because
        it is used in the automatic pipelines
        """

        identity=self.identity
        animals=[animal for animal in self.datasetnames if animal.endswith(str(identity).zfill(2))]
        ids=[ident for ident in self.ids if ident.endswith(str(identity).zfill(2))]
        files=[(
            self.get_pose_file_h5py(pose_name="filter_rle-jump"),
            self.get_pose_file_h5py(pose_name="raw")
        )]

        self.load_store_index_v2(min_time=min_time-1, max_time=max_time+1)
        stride=1

        min_frame_number, max_frame_number, ts=times_to_frame_slice(self.store_index, min_time=min_time, max_time=max_time)
        frame_number=np.arange(min_frame_number, max_frame_number, stride)
        
        with h5py.File(files[0][0]) as file:
            first_chunk=int(os.path.basename(file["files"][0].decode()).split(".")[0])
            first_frame_number=first_chunk*CHUNKSIZE
            x0=min_frame_number-first_frame_number
            x1=max_frame_number-first_frame_number
            bodyparts=[bp.decode() for bp in file["node_names"][:]]
        
            pose=pd.concat([
                # 0, x/y, bodyparts, time
                pd.DataFrame(file["tracks"][0, 0, :, x0:x1:stride].T, columns=[f"{bp}_x" for bp in bodyparts]),
                pd.DataFrame(file["tracks"][0, 1, :, x0:x1:stride].T, columns=[f"{bp}_y" for bp in bodyparts])
            ], axis=1)

        pose.insert(0, "frame_number", frame_number)
        pose.insert(1, "id", self.ids[0])
        pose.insert(2, "t", ts)
        
        if absolute:
            self.dt=None
            self.load_centroid_data(min_time=min_time-1, max_time=max_time+1)
            pose=pose.merge(self.dt[["frame_number", "id", "x", "y"]], on=["frame_number", "id"], how="left")
            self.dt=None

            pose["center_x"]=pose["x"]*self.roi_width-SQUARE_WIDTH//2
            pose["center_y"]=pose["y"]*self.roi_width-SQUARE_HEIGHT//2
            for bodypart in BODYPARTS:
                pose=project_to_absolute_coords(pose, bodypart)

        return pose


    def load_pose_data(self, experiment=None, identity=None, min_time=None, max_time=None, time_system="zt", stride=1, cache=None, verbose=False, files=None, write_only=False):

        if experiment is None:
            experiment=self.experiment
        if identity is None:
            identity=self.identity

        if files is None:
            files=[(
                self.get_pose_file_h5py(pose_name="filter_rle-jump"),
                self.get_pose_file_h5py(pose_name="raw")
            )]
        
        if min_time is not None and max_time is not None and min_time>=max_time:
            logger.warning("Passed time interval (%s - %s) is meaningless")

        self.load_store_index(cache=cache)
        self.store_index["t"]=self.store_index["frame_time"] + self.meta_info["t_after_ref"]

        ret=False
        pose=None
        cache_path=None

        if cache is not None and min_time is None and max_time is None:
            cache_path = f"{cache}/{self.experiment}__{str(identity).zfill(2)}_{stride}_pose_data.pkl"
            if write_only:
                ret=False
            else:
                logger.debug("Cache: %s", cache_path)
                ret, out=restore_cache(cache_path)
                if not ret:
                    logger.info("%s not found or not loadable", cache_path)

                if ret:
                    (pose, meta_pose)=out

        if not ret:
            assert files is not None, f"No files passed and could not find {cache_path} in cache"
            animals=[animal for animal in self.datasetnames if animal.endswith(str(identity).zfill(2))]
            ids=[ident for ident in self.ids if ident.endswith(str(identity).zfill(2))]
            if len(animals)==0 or len(ids)==0:
                logger.error("animal with identity %s not available", identity)
            
            out=load_pose_data_compiled(animals, ids, self.lq_thresh, stride=stride, files=files, store_index=self.store_index)

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
                # this line works even if pose has no annotation on t, because only the frame_number is needed
                pose=self.filter_pose_by_time(pose=pose, min_time=min_time, max_time=max_time)
                assert pose.shape[0]>0, f"No pose data after filtering"
                if cache_path is not None:
                    save_cache(cache_path, (pose, meta_pose))
                        

        self.pose=pose
        id=pose["id"].iloc[0]
        self.meta_pose[id]=meta_pose
        self.filter_pose_by_time(min_time, max_time)
        return None


    @staticmethod
    def filter_pose_by_identity(pose, identity):
        return pose.loc[pose["identity"]==identity]
    

    def add_centroid_data_to_pose(self):
        assert self.dt is not None
        self.pose=self.pose.drop(
            ["center_x", "center_y"], axis=1, errors="ignore"
        ).merge(
            self.dt[["frame_number", "center_x", "center_y"]], on="frame_number", how="left"
        )


    def project_to_absolute_coords_all(self, bodyparts):
        """
        Project to coordinate system of the original frame, i.e. not egocentric
        """
        # there can be entries in the pose even if there are none in the dt
        # this is if the fragment was not complete in every frame AND the pose then should be all NaN or interpolated
        # (but still be present as a row in loader.pose)
        pose=self.pose.copy()
        for bodypart in bodyparts:
            pose=project_to_absolute_coords(pose, bodypart)
        
        self.pose_absolute=pose

    def generate_pose_complex(self, pose, bodyparts):
        """
        Prepare pose dataset as required by preprint feature pipeline
        """
        features=[
            "frame_number", "center_x", "center_y"
        ]

        for bodypart in bodyparts:
            features+=[f"{bodypart}_x", f"{bodypart}_y"]
        
        pose=pose[features]
        pose_complex=[]
        for bodypart in ["head", "thorax", "abdomen"]:
            loc=pose[[f"{bodypart}_x", f"{bodypart}_y"]].values
            loc=pd.DataFrame({bodypart: loc[:,0] + loc[:,1]*1j})        
            pose_complex.append(loc)

        pose_complex=pd.concat(pose_complex, axis=1)
        pose_complex.insert(0, "frame_number", pose["frame_number"])
        self.pose_complex=pose_complex

            
    def filter_pose_by_time(self, min_time, max_time, pose=None):
        if pose is None:
            pose = self.pose

        if len(pose) > 0 and min_time is not None and max_time is not None:
            t = self.store_index["frame_time"] + self.meta_info["t_after_ref"]
            frame_numbers = self.store_index["frame_number"]

            # Handle min_time
            if min_time <= t.min():
                min_fn = frame_numbers.min()
            else:
                min_fn = frame_numbers.iloc[np.argmax(t >= min_time)]

            # Handle max_time
            if max_time >= t.max():
                max_fn = frame_numbers.max() + 1  # +1 ensures exclusive filtering
            else:
                max_fn = frame_numbers.iloc[np.argmax(t >= max_time)]

            pose = pose.loc[
                (pose["frame_number"] >= min_fn) & (pose["frame_number"] < max_fn)
            ]

        return pose



    def boxcar_filter(self, pose, features, bodyparts, framerate=150):

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
                download=True,
                n_jobs=1
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
            pose_annotated=pose.merge(behaviors[["frame_number", "id", "behavior"]], on=["frame_number", "id"], how="left")
            pose_annotated.loc[pd.isna(pose_annotated["behavior"]), "behavior"]="unknown"
        return pose_annotated



    def export(self, pose=None, dest_folder=None, bodyparts=BODYPARTS, id=None):
        f"""
        Export the pose to an .h5 dataset compatible with motionmapper, bsoid, etc
        If the pose dataset is not specified, it will use pose_interpolated by default
        """
        assert dest_folder is not None

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
        # fist axis contains X-Y
        # second axis contains bodyparts
        # third axis contains time
        reshaped_array=reshaped_array[np.newaxis, :]

        assert len(frame_number) == reshaped_array.shape[3]

        files=sorted(self.meta_pose[id]["files"], key=lambda x: os.path.basename(x))
        
        key=f"{self.experiment}__{str(identity).zfill(2)}"


        folder=f"{dest_folder}/{key}"
        track_names=self.datasetnames
        os.makedirs(folder, exist_ok=True)
        filename=f"{folder}/{key}.h5"
        print(f"Saving to --> {filename}")
        f=h5py.File(filename, "w")
        f.create_dataset("tracks", data=reshaped_array)
        f.create_dataset("node_names", data=node_names)
        f.create_dataset("files", data=files)
        i=f.create_dataset("track_names", (len(track_names),), dtype="|S40")
        i[:]=np.array([e.encode() for e in track_names])

        j=f.create_dataset("filters", (len(self.filters),), dtype="|S50")
        j[:]=np.array([e.encode() for e in self.filters])

        f.close()
        return