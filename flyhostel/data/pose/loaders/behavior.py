import time
import shutil
import logging
import os.path
import pandas as pd
import h5py
from tqdm.auto import tqdm
logger=logging.getLogger(__name__)

from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.utils import restore_cache, save_cache
from flyhostel.data.pose.ethogram.utils import annotate_bout_duration, annotate_bouts
from flyhostel.data.pose.distances import add_interdistance_features
try:
    from motionmapperpy import setRunParameters
    wavelet_downsample=setRunParameters().wavelet_downsample
except ModuleNotFoundError:
    wavelet_downsample=5


def get_behavior_feather_file_path(experiment, identity):
    tokens=experiment.split("_")
    basedir=os.path.join(
        os.environ["FLYHOSTEL_VIDEOS"],
        tokens[0],
        tokens[1],
        "_".join(tokens[2:4])
    )
    
    feather_path=os.path.join(
        basedir,
        "motionmapper",
        str(identity).zfill(2),
        f"{experiment}__{str(identity).zfill(2)}.feather"
    )
    return feather_path


def interpolate_wavelets(dt, interpolate_frames=0):

    dt_t_complete=[]
    for i in tqdm(range(interpolate_frames), desc="filling wavelet gaps"):
        df=dt.copy()
        df["frame_number"]+=i
        dt_t_complete.append(df)
    
    dt=pd.concat(dt_t_complete, axis=0)
    del dt_t_complete
    dt.sort_values(["id", "frame_number"], inplace=True)
    return dt


class BehaviorLoader():

    def __init__(self, *args, **kwargs):
        self.behavior=None
        self.store_index=None
        self.pose=None
        super(BehaviorLoader, self).__init__(*args, **kwargs)

    def load_store_index(self):
        raise NotImplementedError


    def get_behavior_feather_file(self, experiment, identity, fail=False):
        file=get_behavior_feather_file_path(experiment, identity)
        assert os.path.exists(file)
        return file

            
    def load_behavior_data(self, experiment, identity, min_time=None, max_time=None):
        """
        Load the results of draw_ethogram (predict_behavior process)
        """
        feather_path=self.get_behavior_feather_file(experiment, identity)
        if os.path.exists(feather_path):
            logger.debug("Reading %s", feather_path)
            dt=pd.read_feather(feather_path)

        else:
            logger.warning("%s does not exist", feather_path)
            return
        
        if min_time is not None:
            dt=dt.loc[dt["t"]>=min_time]
        if max_time is not None:
            dt=dt.loc[dt["t"]<max_time]

        self.behavior=dt

    # def load_behavior_data(self, experiment, identity, pose=None, interpolate_frames=0, cache=None):
    #     """
    #     Load the results of draw_ethogram (predict_behavior process)
    #     Add interdistance features (proboscis-head)
    #     Annotate ZT
    #     Annotate bouts
    #     """
    #     PREDICTION_COL="prediction2"

    #     if cache is not None:
    #         path = os.path.join(cache, f"{experiment}__{str(identity).zfill(2)}_behavior.pkl")
    #         # ret, self.behavior = restore_cache(path)
    #         ret=False
    #         if ret:
    #             return
            
    #     feather_path=self.get_behavior_feather_file(experiment, identity)


    #     if os.path.exists(feather_path):

    #         logger.debug("Reading %s", feather_path)
    #         dt=pd.read_feather(feather_path)
    #         i=list(dt.columns).index("score")
    #         behaviors=list(dt.columns[i+1:])
    #         import ipdb; ipdb.set_trace()
    #         dt=dt[["id", "frame_number", PREDICTION_COL, "score"] + behaviors]
    
    #         if interpolate_frames>0:
    #             dt=interpolate_wavelets(dt, interpolate_frames=interpolate_frames)
            
    #         fps=FRAMERATE//wavelet_downsample

    #         if pose is None:
    #             logger.warning("Pose not found. Please provide a pose dataset, or refining of behavior will be disabled")
    #         else:
    #             before=time.time()
    #             logger.debug("Refining behavior")
    #             pose["chunk"]=pose["frame_number"]//CHUNKSIZE
    #             pose["frame_idx"]=pose["frame_number"]%CHUNKSIZE
    #             pose, _=add_interdistance_features(pose, [], ["head", "proboscis"], prefix="distance_")

    #             # pose=compute_distance_features_pairs(pose, pairs=[("head", "proboscis"), ])
    #             distances=[column for column in pose.columns if "distance" in column]

    #             dt=pose[["id", "t", "frame_number", "chunk", "frame_idx", "identity"] + distances].merge(
    #                 dt[["id", "frame_number", PREDICTION_COL, "score"]], how="right", on=["id", "frame_number"]
    #             )
                
    #             dt.loc[(dt["distance_head__proboscis"] < 0.01) & (dt[PREDICTION_COL]=="inactive+pe"), PREDICTION_COL]="inactive"
    #             after=time.time()
    #             logger.debug("Refining behavior took %s seconds", after-before)


    #     else:
    #         logger.warning("%s does not exist", feather_path)
    #         return
        

    #     logger.debug("Annotating bouts and their duration")
    #     dt=annotate_bouts(dt, PREDICTION_COL)
    #     dt=annotate_bout_duration(dt, fps=fps)
    #     self.behavior=dt
        
    #     logger.debug("Loading store index")
    #     self.load_store_index()
    #     self.store_index["t"]=self.store_index["frame_time"]+self.meta_info["t_after_ref"]

    #     logger.debug("Annotating time of behavior data")
    #     self.behavior=self.behavior.drop(
    #         "t", axis=1, errors="ignore"
    #     ).merge(
    #         self.store_index[["frame_number", "t"]], on="frame_number", how="left").sort_values(
    #             "frame_number"
    #     )

    #     if cache is not None:
    #         logger.debug("Saving cache to %s", path)
    #         save_cache(path, self.behavior)

