import time
import logging
import os.path
import pandas as pd
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


def get_behavior_feather_file(experiment, identity):
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
    return feather_path



class BehaviorLoader():

    def __init__(self, *args, **kwargs):
        self.behavior=None
        self.pose=None
        super(BehaviorLoader, self).__init__(*args, **kwargs)


    @staticmethod
    def get_behavior_feather_file(experiment, identity):
        return get_behavior_feather_file(experiment, identity)


    def load_behavior_data(self, experiment, identity, pose=None, interpolate_frames=0, cache=None):

        if cache is not None:
            path = os.path.join(cache, f"{experiment}__{str(identity).zfill(2)}_behavior.pkl")
            ret, self.behavior = restore_cache(path)
            ret=False
            if ret:
                return
            
        feather_path=self.get_behavior_feather_file(experiment, identity)


        if os.path.exists(feather_path):

            dt=pd.read_feather(feather_path)
            i=list(dt.columns).index("score")
            behaviors=list(dt.columns[i+1:])
            
            dt=dt[["id", "frame_number", "behavior", "score"] + behaviors]
    
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
                pose, _=add_interdistance_features(pose, [], ["head", "proboscis"], prefix="distance_")

                # pose=compute_distance_features_pairs(pose, pairs=[("head", "proboscis"), ])
                distances=[column for column in pose.columns if "distance" in column]

                dt=pose[["id", "t", "frame_number", "chunk", "frame_idx", "identity"] + distances].merge(
                    dt[["id", "frame_number", "behavior", "score"]], how="right", on=["id", "frame_number"]
                )
                
                dt.loc[(dt["distance_head__proboscis"] < 0.01) & (dt["behavior"]=="inactive+pe"), "behavior"]="inactive"
                after=time.time()
                logger.debug("Refining behavior took %s seconds", after-before)



        else:
            logger.warning("%s does not exist", feather_path)
            return
    
        dt=annotate_bouts(dt, "behavior")
        dt=annotate_bout_duration(dt, fps=fps)
        self.behavior=dt
        
        self.store_index["t"]=self.store_index["frame_time"]+self.meta_info["t_after_ref"]

        self.behavior=self.behavior.drop(
            "t", axis=1, errors="ignore"
        ).merge(
            self.store_index[["frame_number", "t"]], on="frame_number", how="left").sort_values(
                "frame_number"
        )


        if cache is not None:
            save_cache(path, self.behavior)

