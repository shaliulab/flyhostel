import logging
import itertools
import os.path

import h5py
import numpy as np
import pandas as pd

from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.constants import get_bodyparts

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

CONTACT_THRESHOLD=2 # mm
PROXIMITY_THRESHOLD=4 # mm
DURATION_THRESHOLD=.5 # seconds
CONTACT_THRESHOLD=2 # mm
PROXIMITY_THRESHOLD=4 # mm
DURATION_THRESHOLD=.5 # seconds
from flyhostel.data.pose.constants import framerate as FRAMERATE

def index_interactions(interactions, step, framerate=FRAMERATE, contact_threshold=CONTACT_THRESHOLD, duration_threshold=DURATION_THRESHOLD, min_time_between_interactions=1, time_window_length=1):
    """
    
    Arguments:
        time_window_length (int): Number of seconds making up one behavioral bin
    """
    df=[]
    groupby=["id", "nn", "interaction"]
    interactions.sort_values(["frame_number", "id", "nn"], inplace=True)
    min_step_to_next_interaction=FRAMERATE*min_time_between_interactions
    for pair, dff in interactions.groupby(groupby[:2]):
        diff=np.diff(dff["frame_number"])
        dff["interaction"]=[0] + (np.cumsum(diff>min_step_to_next_interaction)).tolist()
        df.append(dff)
    if df:
        interactions=pd.concat(df, axis=0).reset_index(drop=True)
    else:
        raise ValueError("No interactions detected")
    
    del df

    counts=interactions.groupby(groupby).size().reset_index(name="count")
    min_distance=interactions.groupby(groupby).agg({"distance_mm": np.min}).reset_index().rename({
        "distance_mm": "distance_mm_min"
    }, axis=1)
    counts["duration"]=counts["count"]*(step/framerate)
    first_frame=interactions.groupby(groupby).first()["frame_number"].reset_index().rename({"frame_number": "first_frame"}, axis=1)

    interaction_index=interactions.merge(first_frame, on=groupby, how="inner")\
      .merge(counts, on=groupby, how="left")\
      .merge(min_distance, on=groupby, how="left")
    interaction_index["last_frame_number"]=[int(e) for e in interaction_index["first_frame"]+interaction_index["duration"]*framerate]
    interaction_index["t"]=time_window_length*(interaction_index["t"]//time_window_length)
    interaction_index["t_end"]=np.ceil(interaction_index["t"]+interaction_index["duration"])
    
    interaction_index["signif"]=False
    interaction_index.loc[
        (interaction_index["duration"]>duration_threshold) & \
        (interaction_index["distance_mm_min"]<contact_threshold),
        "signif"
    ]=True
    return interaction_index


class InteractionsLoader:
    """
    A class to load the result of the first step in the interactions pipeline
    """
    
    basedir=None
    experiment=None
    identity=None
    ids=[]
    dt=None

    def __init__(self, *args, **kwargs):
        self.interaction=None
        super(InteractionsLoader, self).__init__(*args, **kwargs)

    def load_interaction_data(
            self, experiment=None, identity=None,
            duration_threshold=DURATION_THRESHOLD,
            proximity_threshold=PROXIMITY_THRESHOLD,
            contact_threshold=CONTACT_THRESHOLD,
            framerate=50
        ):
        
        if experiment is None:
            experiment=self.experiment
        if identity is None:
            identity=self.identity
    
        neighbors_features=["id", "nn", "distance_mm", "frame_number"]
        csv_file=os.path.join(self.basedir, "interactions", experiment + "_neighbors.csv")
        if not os.path.exists(csv_file):
            return None, None
            
        neighbors=pd.read_csv(csv_file)[neighbors_features]
        
        neighbors=neighbors.loc[neighbors["id"]==self.ids[0]]
        assert neighbors.shape[0]>0, f"No flies neighboring {self.ids[0]}"

        if proximity_threshold is not None:
            neighbors=neighbors.loc[neighbors["distance_mm"]<proximity_threshold]
        
        time_index=self.index_time()
        neighbors=neighbors.merge(time_index, on="frame_number", how="inner")
        step=FRAMERATE//framerate
        assert neighbors.shape[0]>0, f"No data left for {self.ids[0]}"
        interactions=index_interactions(
            neighbors, step=step, contact_threshold=contact_threshold,
            duration_threshold=duration_threshold
        )
        interaction_signif = interactions.loc[interactions["signif"]]
        self.interaction=interaction_signif
        return None


    def index_time(self):
        time_index=self.dt[["frame_number", "t"]]
        return time_index
        