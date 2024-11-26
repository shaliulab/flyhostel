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

            
    def load_behavior_data(self, experiment=None, identity=None, min_time=None, max_time=None):
        """
        Load the results of draw_ethogram (predict_behavior process)
        """

        if experiment is None:
            experiment=self.experiment

        if identity is None:
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