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
        self.experiment=None
        self.identity=None
        super(BehaviorLoader, self).__init__(*args, **kwargs)

    def load_store_index(self):
        raise NotImplementedError


    def get_behavior_feather_file(self, experiment, identity, fail=False):
        file=get_behavior_feather_file_path(experiment, identity)
        assert os.path.exists(file)
        return file

            
    def load_behavior_data(self, min_time=None, max_time=None):
        """
        Load a behavior timeseries computed at 30 FPS from an original centroid timeseries at 150 FPS
        Data frame is available in self.behavior

        Features:
            * id: Unique identifier for the animal
            * frame_number: Number of frames passed until the collection of the current frame
            * t: Seconds since ZT0
            * chunk, frame_idx: One chunk is made by 45k consecutive frames. frame_idx is the position within the chunk (goes up to 45k)
            * x, y: Absolute coordinates of the animal in the arena, relative to the top left corner and normalized to 1 (so the bottom right corner is 1,1)
            * food_distance: Distance to the edge of the patch of food. Negative distance = inside the patch. Unit is ROI width
            * notch_distance: Distance to the edge of the closest notch on the glass. Unit is ROI width
            * score: Score given by the RF model to the winning behavioral label
            * prediction: Winning behavioral label
            * prediction2: Behavioral label after modifications based on heuristic rules
            * rule: Which rule was applied to modify prediction
            * bout_in_pred, bout_out_pred: How many consecutive time points with the same prediction2 value until this point or from this point to the end of the bout
            * duration_pred: How many seconds does the bout last. If bout_in_pred=1, duration_pred = bout_out_pred / 30 (because fps=30)
            * centroid_speed: Distance travelled by the centroid from the previous timepoint to the current one (fps=30)
            * centroid_speed_1s: Distance travelled by the centroid in the last second, computed by adding the distance travelled in the last 150 original timepoints           
        """

        experiment=self.experiment

        identity=self.identity

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

        duplicated_locations=dt.duplicated(["id", "frame_number"]).sum()
        if duplicated_locations>0:
            logger.warning("%s duplicated rows in behavior dataset found", duplicated_locations)
            dt.drop_duplicates(["id", "frame_number"], inplace=True)
            
        self.behavior=dt