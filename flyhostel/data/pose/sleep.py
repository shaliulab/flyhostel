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

import pandas as pd
import numpy as np

from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.ethogram_utils import (
    annotate_bout_duration,
    annotate_bouts
)
from flyhostel.data.pose.ethogram_utils import most_common_behavior_vectorized
from deepethogram.postprocessing import find_bout_indices
from motionmapperpy import setRunParameters
from zeitgeber.rle import encode, decode

wavelet_downsample=setRunParameters().wavelet_downsample

# duration of a behavioral unit
RESOLUTION=1
# max duration of the brief awakening in seconds
MAX_GAP=5
inactive_states=["inactive", "pe_inactive", "micromovement", "inactive+micromovement"]




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



class SleepAnnotator(ABC):
    """
    Extend a FlyHostelLoader with the ability to annotate sleep and detect brief awakenings
    """

    @staticmethod
    def find_brief_awakenings(dt, *args, **kwargs):
        return find_brief_awakenings(dt, *args, **kwargs)
    

    def sleep_annotation_inactive(self, data, min_time_immobile=300, time_window_length=1):
        
        dt=most_common_behavior_vectorized(data, time_window_length)
        
        behavior=dt["behavior"].copy()
        behavior.loc[behavior.isin(inactive_states)]="all_inactive"
        
        dt["moving"]=np.bitwise_not(behavior=="all_inactive")
        dt["asleep"]=sleep_contiguous(dt["moving"], min_valid_time=min_time_immobile, fs=1/time_window_length)
        return dt
        

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
