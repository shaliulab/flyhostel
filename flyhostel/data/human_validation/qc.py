import logging
import collections
import joblib
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
# from memory_profiler import profile
from flyhostel.utils import prepare_batches
from .constants import BATCH_SIZE, LOGFILE
from .qcs import (
    yolov7_qc,
    all_found_qc,
    all_id_expected_qc,
    first_frame_idx_qc,
    inter_qc,
)

logger=logging.getLogger(__name__)

# @profile
def all_qc_batch(all_windows, **kwargs):
    out=[]
    while len(all_windows)>0:
        consecutive_windows=all_windows.pop(0)
        out.append(all_qc(**consecutive_windows, **kwargs))

    qc=pd.DataFrame(out)
    return qc


# @profile
def all_qc(i, number_of_animals, behavior_window, chunksize, window_before=None, window_after=None, logfile=None):
    """
    For every group of windows, verify:

        * That yolov7 was not used (potential AI mistake) in behavior_window
        * That all animals are found in behavior_window
        * That behavior_window is not in the first or last frame of a chunk
        * That there was no fragment change (potential errors) between the three windows
    """
    frame_number=int(behavior_window[0, frame_number_idx])

    yolov7_pass=yolov7_qc(behavior_window)
    all_found_pass=all_found_qc(behavior_window, number_of_animals)
    all_id_expected_pass=all_id_expected_qc(behavior_window, number_of_animals)
    first_frame_idx_pass=first_frame_idx_qc(behavior_window, chunksize)
    last_frame_idx_pass=last_frame_idx_qc(behavior_window, chunksize)

    if i == 0:
        inter_qc_pass=True
    else:
        inter_qc_pass=inter_qc(window_before=window_before, window=behavior_window, window_after=window_after)

    if i % 50000 == 0 and logfile is not None:
        with open(logfile, "w") as handle:
            handle.write(f"Last window: {i}\nLast frame number {frame_number}\n")

    qc = (
        True and
        # require yolov7 is not used
        yolov7_pass and
        # require all flies are found / segmented (even if the identity is not assigned)
        all_found_pass and
        # require fragments dont change
        inter_qc_pass and
        # require it is not first frame in chunk unless all animals have an identity
        (first_frame_idx_pass or all_id_expected_pass) and
        # require it is not last frame in chunk unless all animals have an identity
        (first_frame_idx_pass or all_id_expected_pass)
    )

    result={
        "frame_number": frame_number,
        "yolov7_qc": yolov7_pass,
        "all_found_qc": all_found_pass,
        "all_id_expected_qc": all_id_expected_pass,
        "first_frame_idx_qc": first_frame_idx_pass,
        "last_frame_idx_qc": last_frame_idx_pass,
        "inter_qc_qc": inter_qc_pass,
        "qc": qc
    }

    return result


def annotate_nan_frames(df):
    return df


def generate_consecutive_windows(df):

    n_windows=df[["chunk", "frame_number"]].drop_duplicates().shape[0]
    # 0 local_identity
    # 1 identity
    # 2 chunk
    # 3 fragment
    # 4 modified
    # 5 frame_number
    # 6 x
    # 7 y
    FEATURES=["local_identity", "identity", "chunk", "fragment", "modified", "frame_number", "x", "y"]

    all_windows=df[FEATURES].groupby([
        "chunk", "frame_number"
    ]).__iter__()
    logger.debug("Generating %s windows", n_windows)

    output=[]
    _, window_after = next(all_windows)
    window_after=window_after.values.astype(np.int32)
    window_group=collections.deque([None, None, window_after])
    has_finished=False
    pb=tqdm(total=n_windows)
    i=0

    while not has_finished:
        try:
            _, window_after = next(all_windows)
            window_after=window_after.values.astype(np.int32)
            window_group.append(window_after)
        except StopIteration:
            has_finished=True
            window_group.append(None)

        window_group.popleft()
        output.append({
            "i": i,
            "window_before": window_group[0],
            "behavior_window": window_group[1],
            "window_after": window_group[2],
        })
        pb.update(1)
        i+=1
    return output
    

def analyze_experiment(df, number_of_animals, min_frame_number, max_frame_number, chunksize, n_jobs=1):
    """
    Quality control (QC) of segmentation and identification (idtrackerai+yolov7) pipeline

    Arguments

        df (pd.DataFrame): One row per animal and bin, with columns
            local_identity
            identity
            chunk
            fragment
            modified
            frame_number
            x
            y
        min_frame_number (int): Start QC from this frame
        max_frame_number (int): End QC at this frame
        chunksize (int): Number of frames in a flyhostel chunk
        For 150 FPS = 45000 in 5 minutes


    Works by producing windows of behavior
    A window consists of rows that come from the same frame
    This function 
    """

    logger.debug("Sorting data chronologically")
    df.sort_values(["chunk", "frame_number"], inplace=True)


    consecutive_windows=generate_consecutive_windows(df)
    batches=prepare_batches(consecutive_windows, BATCH_SIZE, n_jobs=n_jobs)
    
    logger.debug("Running QC using %s jobs in %s batches of size %s. Saving log to %s", n_jobs, len(batches), BATCH_SIZE, LOGFILE)
    # debug

    qc=joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            all_qc_batch
        )(
            batches[j],
            number_of_animals=number_of_animals,
            chunksize=chunksize,
            logfile=LOGFILE
        )
        for j in range(len(batches))
    )
    qc=pd.concat(qc, axis=0)
    return qc
