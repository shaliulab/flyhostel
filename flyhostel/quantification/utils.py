import logging

from tqdm import tqdm
import joblib
import pandas as pd
import numpy as np

from flyhostel.constants import N_JOBS

logger = logging.getLogger(__name__)

def bin_apply(dt, y, analysis_params):

    logger.debug(
        f"Binning data every {analysis_params.summary_time_window/60} minutes"
    )

    dt["t_bin"] = (
        np.floor(dt["t"] / analysis_params.summary_time_window)
        * analysis_params.summary_time_window
    )

    dt = dt.groupby(["id", "t_bin"])
    dt_binned = getattr(dt, analysis_params.sumary_FUN)()

    dt_binned = dt_binned.reset_index()[["id", "t_bin", y]]
    dt_binned.columns = ["id", "t", y]
    return dt_binned


def init_data_frame():
    return pd.DataFrame(
        {
            "velocity": [],
            "frame_number": [],
            "t": [],
            "t_round": [],
            "id": [],
            "L": [],
        }
    )


def tidy_dataset(velocity, chunk_metadata, analysis_params):

    frame_number, frame_time = chunk_metadata
    assert len(velocity) == len(frame_number[1:-1])

    data = pd.DataFrame(
        {"velocity": velocity, "frame_number": frame_number[1:-1]}
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


def tidy_dataset_all(velocities, **kwargs):
    logger.debug("Tidying dataset")

    n_animals = velocities.shape[1]
    data = init_data_frame()
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
        d["id"] = [
            i,
        ] * d.shape[0]
        data = pd.concat([data, d])

    data = (
        data.groupby(["id", "t_round"])
        .max()
        .reset_index()[["velocity", "id", "t_round", "L"]]
    )

    return data

def make_suffix(analysis_params):
    suffix = f"{str(analysis_params.velocity_correction_coef * 1000).zfill(5)}"
    f"-{analysis_params.min_time_immobile}"
    f"-{analysis_params.time_window_length}"
    f"-{analysis_params.summary_time_window}"
    return suffix