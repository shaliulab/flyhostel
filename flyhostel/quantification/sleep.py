import itertools
import logging
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import zeitgeber # https://github.com/shaliulab/zeitgeber
from flyhostel.constants import N_JOBS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def sleep_annotation(data, analysis_params):
    """
    Arguments:

        * data (pd.DataFrame): data frame with columns velocity, t_round
        * analysis_params (namedtuple): tuple with elements min_time_immobile, time_window_length, velocity_correction_coef
    """

    # work on an explicit copy of data
    # to avoid warning described in
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy

    dt_sleep = data.copy()
    
    dt_sleep["moving"] = (data["velocity"] > analysis_params.velocity_correction_coef)
    dt_sleep["window_number"] = np.arange(data.shape[0])
    rle = zeitgeber.rle.encode((~dt_sleep["moving"]).tolist())
    dt_sleep["asleep"] = list(
        itertools.chain(
            *[
                # its asleep if
                # 1. it is not moving (x[0])
                # 2. if the length of the not moving state (x[1]) is >= the ratio between
                # min_time_immobile and time_window_length (i.e. the minimum number of windows)
                [
                    x[0]
                    and x[1]
                    >= (
                        analysis_params.min_time_immobile
                        / analysis_params.time_window_length
                    ),
                ]
                * x[1]
                for x in rle
            ]
        )
    )


    dt_sleep["t"] = data["t_round"].tolist()
    dt_sleep.drop("t_round", axis=1, inplace=True)
    return dt_sleep


def sleep_annotation_all(data, **kwargs):

    logger.debug(f"Annotating sleep behavior")

    if N_JOBS == 1:
        data_annot = []
        for i in tqdm(
            np.unique(data["id"]), desc="Quantifying sleep on animal"
        ):
            data_annot.append(
                sleep_annotation(data.loc[data["id"] == i], **kwargs)
            )
    else:
        data_annot = joblib.Parallel(n_jobs=N_JOBS)(
            joblib.delayed(sleep_annotation)(
                data.loc[data["id"] == i], **kwargs
            )
            for i in np.unique(data["id"])
        )
    
    logger.debug("Done")
    dt = pd.concat(data_annot)
    return dt
