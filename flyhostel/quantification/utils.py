import logging

from tqdm import tqdm
import joblib
import pandas as pd
import numpy as np

from flyhostel.constants import N_JOBS
from flyhostel.quantification.constants import FLYHOSTEL_ID

logger = logging.getLogger(__name__)

def bin_apply(dt, y, analysis_params, keep_cols=[]):

    logger.debug(
        f"Binning data every {analysis_params.summary_time_window/60} minutes"
    )

    dt_original = dt.copy()

    dt["t_bin"] = (
        np.floor(dt["t"] / analysis_params.summary_time_window)
        * analysis_params.summary_time_window
    )

    dt = dt.groupby([FLYHOSTEL_ID, "t_bin"])
    dt_binned = getattr(dt, analysis_params.sumary_FUN)()

    dt_binned = dt_binned.reset_index()[[FLYHOSTEL_ID, "t_bin", y]]
    dt_binned.columns = [FLYHOSTEL_ID, "t", y]

    if keep_cols:
        keep_cols.append(FLYHOSTEL_ID)
        keep_data = dt_original[keep_cols]
        keep_data = keep_data.sort_values(by=keep_cols).drop_duplicates(FLYHOSTEL_ID)
        dt_binned=pd.merge(dt_binned, keep_data, on=FLYHOSTEL_ID)

      
    return dt_binned


def init_data_frame():
    return pd.DataFrame(
        {
            "velocity": np.array([], np.float64),
            "frame_number": np.array([], np.int64),
            "t": np.array([], np.float64),
            "t_round": np.array([], np.int64),
            "fly_no": np.array([], np.int64),
            "L": [],
        }
    )


def tidy_dataset(velocity, chunk_metadata, analysis_params):
    """
    Given a velocity array of shape nframesx2,
    produce a dataframe with a structure similar to the ethoscope
    """

    frame_number, frame_time = chunk_metadata
    assert len(velocity) == len(frame_number)

    data = pd.DataFrame(
        {"velocity": velocity, "frame_number": frame_number}
    )

    # its better to use the index instead of the frame number
    # in case the first frame_number is not 0
    data["frame_time"] = [frame_time[i] for i, _ in enumerate(data["frame_number"])]
    data["t"] = data["frame_time"]
    data["t"] /= 1000  # to seconds
    data["t"] += analysis_params.offset
    data["L"] = ["T" if e else "F" for e in ((data["t"] / 3600) % 24) < 12]
    data["t_round"] = (
        np.floor(data["t"] / analysis_params.time_window_length)
        * analysis_params.time_window_length
    )

    data.drop("frame_time", axis=1, inplace=True)
    return data


def annotate_id(data, experiment):

    assert "fly_no" in data.columns
    
    indices = [str(int(e)).zfill(2) for e in data['fly_no']]
    data[FLYHOSTEL_ID] = [f"{experiment}|{e}" for e in indices]
    return data

def tidy_dataset_all(velocities, experiment_name, **kwargs):
    """
    Tidy a dataset made up by several animals
    Each animal gets a new slot on the 1th axis (2 dimension) of velocities

    Args:
        * velocities (np.ndarray): of shape nframesxnanimalsx2
    
    Return:
        * dt_raw (pd.DataFrame): Frame by frame data for all flies
        * dat (pd.DataFrame): max-agregated data for all flies, with frequency given by time_window_length
    """

    logger.debug("Tidying dataset")

    n_animals = velocities.shape[1]
    dt_raw = init_data_frame()
    output = []

    if N_JOBS == 1:
        for i in tqdm(range(n_animals), desc="Generating dataset for animal "):
            output.append(tidy_dataset(velocities[:, i], **kwargs))

    else:
        output = joblib.Parallel(n_jobs=N_JOBS)(
            joblib.delayed(tidy_dataset)(velocities[:, i], **kwargs)
            for i in range(n_animals)
        )

    for i in range(n_animals):
        d = output[i]
        d["fly_no"] = [
            i+1,
        ] * d.shape[0]
        dt_raw = pd.concat([dt_raw, d])
        
    dt_raw = annotate_id(dt_raw, experiment_name)
    data = (
        dt_raw.groupby([FLYHOSTEL_ID, "t_round"])
        .max()
        .reset_index()[[FLYHOSTEL_ID, "velocity", "fly_no", "t_round", "L"]]
    )

    dt_raw.drop("t_round", inplace=True, axis=1)

    return dt_raw, data

def make_suffix(analysis_params):
    suffix = f"{str(analysis_params.velocity_correction_coef * 1000).zfill(5)}"
    f"-{analysis_params.min_time_immobile}"
    f"-{analysis_params.time_window_length}"
    f"-{analysis_params.summary_time_window}"
    return suffix