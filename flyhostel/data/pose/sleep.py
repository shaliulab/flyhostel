# emulate this R line
# dt[, asleep := sleepr:::sleep_contiguous(moving, min_valid_time = min_time_immobile, fs = 30), by=id]

# sleep_contiguous <- function(moving, fs, min_valid_time = 5*60){
#   min_len <- fs * min_valid_time
#   r_sleep <- rle(!moving)
#   valid_runs <-  r_sleep$length >= min_len
#   r_sleep$values <- valid_runs & r_sleep$value
#   inverse.rle(r_sleep)
# }

from abc import ABC
import logging

import pandas as pd
import numpy as np


from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.data.pose.constants import inactive_states
from flyhostel.data.pose.ethogram.utils import (
    annotate_bout_duration,
    annotate_bouts
)
from flyhostel.data.pose.ethogram.utils import postprocessing
from flyhostel.data.pose.ethogram.plot import bin_behavior_table_v2
from deepethogram.postprocessing import find_bout_indices
from motionmapperpy import setRunParameters
from zeitgeber.rle import encode, decode

wavelet_downsample=setRunParameters().wavelet_downsample

# duration of a behavioral unit
RESOLUTION=1
# max duration of the brief awakening in seconds
MAX_GAP=5

logger=logging.getLogger(__name__)

def sleep_contiguous(moving, min_valid_time, fs):
    
    state_mapping={True: "T", False: "F"}
    min_len = fs * min_valid_time
    not_moving = np.bitwise_not(moving)
    r_sleep = encode([state_mapping[v] for v in not_moving])
    length = np.array([tupl[1] for tupl in r_sleep])
    values = np.array([tupl[0]=="T" for tupl in r_sleep])
    valid_runs = length > min_valid_time
    values = np.bitwise_and(values, valid_runs)
    values = [state_mapping[v] for v in values]

    code = zip(values, length)
    sleep_sequence=np.array(list(decode(code)))=="T"
    return sleep_sequence


# this removes the rows where centroid data is available
# but the pose estimation was not run
# typically chunks 20-50 are sometimes available only in centroid form


def find_brief_awakenings_all(dt, *args, **kwargs):
    out=[]
    for id, dt_single_animal in dt.groupby("id"):
        out.append(
            find_brief_awakenings(dt_single_animal, *args, **kwargs)
        )
    out=pd.concat(out, axis=0)
    return out



def sleep_annotation_inactive(
        data, min_time_immobile=300, time_window_length=1,
        min_time_immobile_2=None, max_time_mobile=None, velocity_correction_coef=None, mask=None,
    ):
    
    dt, _, _=bin_behavior_table_v2(data, time_window_length=time_window_length, x_var="t")
    dt=postprocessing(dt, time_window_length)
    logger.debug("Asserting inactive state")
    # import ipdb; ipdb.set_trace()
    behavior_data=np.array(["all_inactive" if "inactive" in behavior or behavior == "background" else "active" for behavior in dt["behavior"]])
    logger.debug("Computing moving state")
    dt["moving"]=np.bitwise_not(behavior_data=="all_inactive")
    logger.debug("Computing asleep state")
    if min_time_immobile_2 is None:
        dt["asleep"]=sleep_contiguous(dt["moving"], min_valid_time=min_time_immobile, fs=1/time_window_length)
    else:
        # import ipdb; ipdb.set_trace()
        dt["nm"]=sleep_contiguous(dt["moving"], min_valid_time=min_time_immobile_2, fs=1/time_window_length)
        real_movement_bout=sleep_contiguous(dt["nm"], min_valid_time=max_time_mobile, fs=1/time_window_length)
        dt["nm2"]=dt["nm"].copy()
        dt.loc[
            np.bitwise_and(~real_movement_bout, ~dt["nm"]),
            "nm2"
        ]=True
        dt["asleep"]=sleep_contiguous(~dt["nm2"], min_valid_time=min_time_immobile, fs=1/time_window_length)

    # import ipdb; ipdb.set_trace()
    logger.debug("sleep_annotation_inactive Done")
    del dt["id"]
    return dt

class SleepAnnotator(ABC):
    """
    Extend a FlyHostelLoader with the ability to annotate sleep and detect brief awakenings
    """

    @staticmethod
    def find_brief_awakenings(dt, *args, **kwargs):
        return find_brief_awakenings(dt, *args, **kwargs)
    

    def sleep_annotation_inactive(self, *args, **kwargs):
        return sleep_annotation_inactive(*args, **kwargs)
        

def find_brief_awakenings(dt, min_significant_bout=60, max_gap=5, time_window_length=1):

    max_gap_length=max_gap//time_window_length

    assert len(dt["id"].unique())==1, f"Please pass datasets of single flies to find_brief_awakenings or use find_brief_awakenings_all"

    dt=pd.DataFrame(dt[["id", "t", "behavior", "frame_number"]])

    dt=most_common_behavior_vectorized(dt.copy(), time_window_length)
    dt["frame_idx"]=dt["frame_number"]%CHUNKSIZE
    dt["chunk"]=dt["frame_number"]//CHUNKSIZE
    dt.loc[dt["behavior"].isin(inactive_states), "behavior"]="all_inactive"


    dt=annotate_bouts(dt, "behavior")
    dt=annotate_bout_duration(dt, fps=1)

    significant_inactive_bout=((dt["behavior"]=="all_inactive") & (dt["duration"]>=min_significant_bout))

    out=[]
    for gap_length in range(1, max_gap_length):
        out.extend(
            find_bout_indices(
                predictions_trace=significant_inactive_bout,
                bout_length=gap_length,
                positive=False,
            )
        )
    out=sorted(out)
    brief_awakenings=dt.iloc[out]

    return brief_awakenings
