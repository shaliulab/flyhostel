import itertools
import logging
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import zeitgeber # https://github.com/shaliulab/zeitgeber
from flyhostel.constants import N_JOBS
from flyhostel.quantification.constants import FLYHOSTEL_ID

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
    rle = zeitgeber.rle.encode((dt_sleep["moving"]).tolist())

    dt_sleep["brief_awakening"] =  list(
        itertools.chain(
            *[
                # its asleep if
                # 1. it is not moving (x[0])
                # 2. if the length of the not moving state (x[1]) is >= the ratio between
                # min_time_immobile and time_window_length (i.e. the minimum number of windows)
                [
                    moving
                    and duration
                    <= (
                        analysis_params.max_brief_awakening
                        / analysis_params.time_window_length
                    ),
                ]
                * duration
                for moving, duration in rle
            ]
        )
    )


    dt_sleep["moving_and_not_in_brief_awakening"] = np.bitwise_and(dt_sleep["moving"], ~np.bitwise_and(dt_sleep["brief_awakening"], dt_sleep["moving"]))

    # Activate this line to enable brief awakenings
    rle = zeitgeber.rle.encode((~dt_sleep["moving_and_not_in_brief_awakening"]).tolist()) 
    # Activate this line to disable brief awakenings
    # rle = zeitgeber.rle.encode((~dt_sleep["moving"]).tolist()) 


    dt_sleep["asleep"] = list(
        itertools.chain(
            *[
                # its asleep if
                # 1. it is not moving (x[0])
                # 2. if the length of the not moving state (x[1]) is >= the ratio between
                # min_time_immobile and time_window_length (i.e. the minimum number of windows)
                [
                    not_moving == True
                    and duration
                    >= (
                        analysis_params.min_time_immobile
                        / analysis_params.time_window_length
                    ),
                ]
                * duration
                for not_moving, duration in rle
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
        for id in tqdm(
            np.unique(data[FLYHOSTEL_ID]), desc="Quantifying sleep on animal"
        ):
            data_annot.append(
                sleep_annotation(data.loc[data[FLYHOSTEL_ID] == id], **kwargs)
            )
    else:
        data_annot = joblib.Parallel(n_jobs=N_JOBS)(
            joblib.delayed(sleep_annotation)(
                data.loc[data[FLYHOSTEL_ID] == id], **kwargs
            )
            for id in np.unique(data[FLYHOSTEL_ID])
        )
    
    logger.debug("Done")
    dt = pd.concat(data_annot)
    return dt
