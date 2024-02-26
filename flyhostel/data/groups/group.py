import logging
import joblib

import numpy as np
import pandas as pd
import cudf

from flyhostel.data.interactions.detector import InteractionDetector
from flyhostel.data.pose.constants import interpolate_seconds
from flyhostel.data.pose.constants import bodyparts_xy as BODYPARTS_XY
from flyhostel.data.pose.constants import bodyparts as BODYPARTS
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.synchrony.main import correlation_synchrony, DEFAULT_LAGS
from flyhostel.data.hostpy import load_hostel

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

    def __init__(self, flies, *args, protocol="full", min_time=12*3600, max_time=24*3600, stride=1, **kwargs):
        self.flies=flies
        for fly in flies.values():
            if protocol=="full":
                if fly.pose is None:
                    fly.load_and_process_data(
                        stride=stride,
                        cache="/flyhostel_data/cache",
                        min_time=min_time, max_time=max_time,
                        filters=None,
                        useGPU=0
                    )
                    fly.compile_analysis_data()
            elif protocol=="centroids":
                # import ipdb; ipdb.set_trace()
                logger.debug("Loading %s centroids", fly)
                fly.load_centroid_data(
                    stride=stride,
                    min_time=min_time, max_time=max_time,
                    identity_table=fly.identity_table,
                    roi_0_table=fly.roi_0_table,
                )

        self.basedir=flies[list(flies.keys())[0]].basedir

        if len(set([fly.basedir for fly in flies.values()])) == 1:
            self.experiment=flies[list(flies.keys())[0]].experiment
            self.number_of_animals=int(self.experiment.split("_")[1].replace("X", ""))

            if len(flies)!=self.number_of_animals:
                logger.warning("You have loaded flies from the same experiment, but at least one is missing")

        else:
            logger.warning("Virtual group of flies detected")
            self.experiment=None
            self.number_of_animals=None
            
        super(FlyHostelGroup, self).__init__(*args, **kwargs)

    @property
    def animals(self):
        return sorted(list(self.flies.keys()))
    
    @property
    def ids(self):
        return [self.flies[animal].ids[0] for animal in self.animals]

    def __len__(self):
        return len(self.flies)

    def compute_synchrony(self, time_window_length=1, min_time_immobile=300, source="inactive", lags=DEFAULT_LAGS):
                
        lags_seconds=np.array(lags)
        lags=(lags_seconds//time_window_length).tolist()

        dt_sleep=self.apply("sleep_annotation", source=source, min_time_immobile=min_time_immobile, time_window_length=time_window_length)
        return correlation_synchrony(dt_sleep, lags=lags)


    def apply(self, function, *args, n_jobs=1, **kwargs):
        
        out=joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(
                getattr(fly, function)
            )(
                pd.DataFrame(fly.load_analysis_data()), *args, **kwargs
            )
            for fly in self.flies.values()
            
        )        
        return pd.concat(out, axis=0)


    def get_simple_metadata(self, flies=None):
        if flies is None:
            flies=self.flies

        return pd.concat([
            fly.get_simple_metadata()
            for fly in flies.values()
        ], axis=0)

    def load_analysis_data(self, n_jobs=1):
        #TODO Check if data is already loaded

        cached_flies={fly: self.flies[fly].analysis_data for fly in self.flies}
        flies={fly: self.flies[fly] for fly in self.flies if cached_flies[fly] is None}
        cached_data=pd.concat(
            [cached_flies[fly] for fly in cached_flies if cached_flies[fly] is not None],
            axis=0
        )
        
        metadata=self.get_simple_metadata(flies)
        data=load_hostel(metadata, n_jobs=n_jobs, value="dataframe")
        data=pd.concat([cached_data, data], axis=0)
        return data


    @classmethod
    def from_metadata(cls, metadata):
        return cls(flies=load_hostel(metadata, n_jobs=1, value="object"))


    @classmethod
    def from_list(cls, flies, *args, **kwargs):
        
        flies_dict={fly.datasetnames[0]: fly for fly in flies}
        return cls(flies=flies_dict, *args, **kwargs)


    def full_interpolation_all(self, pose="pose_boxcar", **kwargs):
        dfs=[]
        for fly in self.flies.values():
            df=getattr(fly, pose).copy()
            dfs.append(fly.full_interpolation(df, **kwargs))
        pose_df=pd.concat(dfs, axis=0)
        return pose_df

    def load_centroid_data(self, framerate=30):
        skip=FRAMERATE//framerate
        dfs=[]
        for fly in self.flies.values():
            df=pd.DataFrame(fly.dt)
            df=cudf.DataFrame(df)
            df=df.loc[df["frame_number"]%skip==0]
            dfs.append(df)
        dt=cudf.concat(dfs, axis=0)
        return dt
        
    
    def load_pose_data(self, pose="pose_boxcar", framerate=30):
        skip=FRAMERATE//framerate

        dfs=[]
        for fly in self.flies.values():
            df=pd.DataFrame(getattr(fly, pose))
            df=cudf.DataFrame(df)
            df=df.loc[(df["frame_number"]%skip)==0]
            dfs.append(df)
        pose=cudf.concat(dfs, axis=0)
        return pose
