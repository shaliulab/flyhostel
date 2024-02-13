import time
import logging
import os.path
import pandas as pd
from tqdm.auto import tqdm
logger=logging.getLogger(__name__)

from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.utils import restore_cache, save_cache
from flyhostel.data.pose.ethogram_utils import annotate_bout_duration, annotate_bouts
from flyhostel.data.pose.distances import compute_distance_features_pairs
try:
    from motionmapperpy import setRunParameters
    wavelet_downsample=setRunParameters().wavelet_downsample
except ModuleNotFoundError:
    wavelet_downsample=5

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
                
            
            fps=FRAMERATE//wavelet_downsample

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

