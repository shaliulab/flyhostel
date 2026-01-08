import logging
import collections
import joblib
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
# from memory_profiler import profile

logger=logging.getLogger(__name__)
# 0 local_identity
# 1 identity
# 2 chunk
# 3 fragment
# 4 modified
# 5 frame_number
# 6 x
# 7 y

identity_idx=0
chunk_idx=2
fragment_idx=3
modified_idx=4
frame_number_idx=5
tests=["yolov7_qc", "all_found_qc", "all_id_expected_qc", "first_frame_idx_qc", "inter_qc"]

BATCH_SIZE=20000

def intra_qc(window, number_of_animals):
    """
    Return True if all of these conditions are met, and False otherwise
        All local identities are non zero
        modified is set to 0 (meaning YOLOv7 has not processed the frame)
        The number of detected animals matches the expected number of animals
    """
    return yolov7_qc(window) and all_found_qc(window, number_of_animals) and all_id_expected_qc(window, number_of_animals) and nms_qc(window)

def nms_qc(window):
    """
    Return True if two objects have the same centroid (non maximal suppresion failure)
    """

    x_y=window[:, 6:7]
    u, c = np.unique(x_y, return_counts=True, axis=0)
    duplicates=np.sum(c > 1)
    return not duplicates

def yolov7_qc(window):
    """
    Return True if YOLOv7 did not process any frame in the group
    """
    return (window[:, modified_idx]==0).all()


def all_found_qc(window, number_of_animals):
    """
    Return True if the number of found objects in all frames of the group
    matches the number of animals
    """
    return window.shape[0]==number_of_animals


def all_id_expected_qc(window, number_of_animals, idx=identity_idx):
    """
    Return True only if the local identities found are the ones expected
    from the number of animals
    So if there are three animals, the local identities available shoould be 1 2 and 3
    """
    if number_of_animals > 1:
        identities=range(1, number_of_animals+1)
    else:
        identities=[0]
    labels = labels=[i for i in identities]
    return (sorted(window[:, idx])==labels)


def inter_qc(window_before, window, window_after):
    """
    Return False if the fragment identifiers in the two windows are not the same
    or if the next window is in another chunk
    In other words, flag a fragment change or new chunks
    """

    is_different = not set(window[:, fragment_idx]).issubset(window_before[:, fragment_idx])
    if window_after is None:
        is_end_of_chunk=False
    else:
        is_end_of_chunk = window[0, chunk_idx] < window_after[0, chunk_idx]

    return not is_different and not is_end_of_chunk


# @profile
def all_qc_batch(all_kwargs):
    out=[]
    while len(all_kwargs)>0:
        kwargs=all_kwargs.pop(0)
        out.append(all_qc(**kwargs))
        del kwargs

    qc=pd.DataFrame.from_records(out, columns=["chunk", "frame_number"] + tests)

    return qc


def first_frame_idx_qc(window, chunksize):
    return window[0, frame_number_idx] % chunksize != 0


# @profile
def all_qc(i, number_of_animals, behavior_window, chunksize, window_before=None, window_after=None, logfile=None, ):
    """
    For every group of windows, verify:

        * That yolov7 was not used (potential AI mistake) in behavior_window
        * That all animals are found in behavior_window
        * That all animals have the expected identities (1, 2, 3, ...) in behavior_window
        * That behavior_window is not in the first frame of a chunk
        * That there was no fragment change (potential errors) between the three windows
    """

    frame_number=int(behavior_window[0, frame_number_idx])
    chunk=int(behavior_window[0, chunk_idx])

    yolov7_pass=yolov7_qc(behavior_window)
    all_found_pass=all_found_qc(behavior_window, number_of_animals)
    all_id_expected_pass=all_id_expected_qc(behavior_window, number_of_animals)
    first_frame_idx_pass=first_frame_idx_qc(behavior_window, chunksize)

    if i == 0:
        inter_qc_pass=True
    else:
        inter_qc_pass=inter_qc(window_before=window_before, window=behavior_window, window_after=window_after)

    if i % 50000 == 0 and logfile is not None:
        with open(logfile, "w") as handle:
            handle.write(f"Last window: {i}\nLast frame number {frame_number}\n")

    return (chunk, frame_number, yolov7_pass, all_found_pass, all_id_expected_pass, first_frame_idx_pass, inter_qc_pass)


def annotate_nan_frames(df):
    return df



def analyze_video(df, number_of_animals, min_frame_number, max_frame_number, chunksize, n_jobs=1):
    """
    Quality control (QC) of idtrackerai+yolov7 results

    Arguments

        df (pd.DataFrame): One row per animal and bin, with columns
            chunk
            frame_number
            identity
            fragment
    """

    logger.debug("Sorting data chronologically")
    df.sort_values(["chunk", "frame_number"], inplace=True)
    logger.debug("Setting index of data")

    n_windows=df[["chunk", "frame_number"]].drop_duplicates().shape[0]
    # 0 local_identity
    # 1 identity
    # 2 chunk
    # 3 fragment
    # 4 modified
    # 5 frame_number
    # 6 x
    # 7 y
    all_windows=df[["local_identity", "identity", "chunk", "fragment", "modified", "frame_number", "x", "y"]].groupby([
        "chunk", "frame_number"
    ]).__iter__()
    logger.debug("Generating %s windows", n_windows)

    kwargs=[]
    _, window_after = next(all_windows)
    window_after=window_after.values.astype(np.int32)
    window_group=collections.deque([None, None, window_after])
    has_finished=False
    pb=tqdm(total=n_windows)
    logfile="qc.log"
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
        kwargs.append({
            "i": i,
            "number_of_animals": number_of_animals,
            "logfile": logfile,
            "window_before": window_group[0],
            "behavior_window": window_group[1],
            "window_after": window_group[2],
            "chunksize": chunksize,
        })
        pb.update(1)
        i+=1

    batches=[]

    if n_jobs>=1:
        n_batches=n_jobs
    else:
        n_cpus=joblib.cpu_count()
        n_batches=n_cpus+n_jobs

    n_batches=n_windows//BATCH_SIZE + 1

    for j in range(n_batches):
        batches.append(
            kwargs[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        )
    logger.debug("Running QC using %s jobs in %s batches of size %s. Saving log to %s", n_jobs, len(batches), BATCH_SIZE, logfile)
    # debug

    qc=joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            all_qc_batch
        )(
            batches[j]
        )
        for j in range(len(batches))
    )
    qc=pd.concat(qc, axis=0)


    extra_check=np.bitwise_and(
        qc["inter_qc"],
        np.bitwise_or(
            qc["first_frame_idx_qc"],
            qc["all_id_expected_qc"]
        )
    )

    qc["qc"]=np.bitwise_and(
        np.bitwise_and(
            qc["yolov7_qc"],
            qc["all_found_qc"],
        ), extra_check
    )

    return qc, tests