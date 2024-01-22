import logging
import os.path

import pandas as pd

from flyhostel.data.interactions.main import InteractionDetector
from flyhostel.data.pose.constants import interpolate_seconds
from flyhostel.data.pose.constants import bodyparts_xy as BODYPARTS_XY
from flyhostel.data.pose.constants import bodyparts as BODYPARTS

logger=logging.getLogger(__name__)


class FlyHostelGroup(InteractionDetector):

    """
    
    How to use:

    # as dictionary of FlyHostelLoaders
    group=FlyHostelGroup(loaders, dist_max_mm, min_interaction_duration)
    # as list
    group=FlyHostelGroup.from_list(loaders, dist_max_mm, min_interaction_duration)
    
    group.find_interactions(BODYPARTS_XY)

    """

    def __init__(self, flies, *args, **kwargs):
        self.flies=flies
        
        assert len(set([fly.basedir for fly in flies.values()])) == 1
        self.basedir=flies[list(flies.keys())[0]].basedir
        self.experiment=flies[list(flies.keys())[0]].experiment
        
        self.number_of_animals=int(self.experiment.split("_")[1].replace("X", ""))

        assert len(flies)==self.number_of_animals

        for fly in flies.values():
            if fly.pose is None:
                fly.load_and_process_data(
                    stride=1,
                    cache="/flyhostel_data/cache",
                    filters=None,
                    useGPU=0
                )

        super(FlyHostelGroup, self).__init__(*args, **kwargs)


    @classmethod
    def from_list(cls, flies, *args, **kwargs):
        
        flies_dict={fly.datasetnames[0]: fly for fly in flies}
        return cls(flies=flies_dict, *args, **kwargs)


    def full_interpolation_all(self, pose="pose_boxcar",  **kwargs):
        dfs=[]
        for fly in self.flies.values():
            df=getattr(fly, pose)
            dfs.append(fly.full_interpolation(df, **kwargs))
        pose=pd.concat(dfs, axis=0).sort_values(["id", "frame_number"])
        return pose


    def load_centroid_data(self):
        dt=pd.concat([
            fly.dt for fly in self.flies.values()
        ], axis=0)

        return dt
    
    def load_pose_data(self):
        return self.full_interpolation_all("pose_boxcar", columns=BODYPARTS_XY, seconds=interpolate_seconds)
        



