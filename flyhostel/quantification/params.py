import datetime
import logging

# mutable namedtuples
# https://pypi.org/project/recordtype/
from recordtype import recordtype

import yaml

logger = logging.getLogger(__name__)

from flyhostel.constants import *


AnalysisParams = recordtype(
    "AnalysisParams",
    [
        "time_window_length",
        "velocity_correction_coef",
        "min_time_immobile",
        "summary_time_window",
        "sumary_FUN",
        "max_brief_awakening",
        "reference_hour",
        "offset",
    ],
)


PlottingParams = recordtype(
    "PlottingParams",
    [
        "chunk_index",
        "experiment_name",
        "ld_annotation",
        "number_of_animals",
        "ethogram_frequency"
    ],
)

ANALYSIS_PARAMS="analysis_params.yaml"


def get_analysis_params(store_metadata):

    date_format = "%Y-%m-%dT%H:%M:%S.%f"

    store_datetime = datetime.datetime.strptime(
        store_metadata["created_utc"], date_format
    )
    store_hour_start = (
        store_datetime.hour
        + store_datetime.minute / 60
        + store_datetime.second / 3600
    )

    try:
        with open(ANALYSIS_PARAMS, "r") as filehandle:
            data = yaml.load(filehandle, yaml.SafeLoader)
    except:
        logger.warning(f"No {ANALYSIS_PARAMS} detected. Using defaults")
        data = {}

    time_window_length = data.get("TIME_WINDOW_LENGTH", DEFAULT_TIME_WINDOW_LENGTH)
    velocity_correction_coef = data.get("VELOCITY_CORRECTION_COEF", DEFAULT_VELOCITY_CORRECTION_COEF) # cm / second
    min_time_immobile = data.get("MIN_TIME_IMMOBILE", DEFAULT_MIN_TIME_IMMOBILE)
    summary_time_window = data.get("SUMMARY_TIME_WINDOW", DEFAULT_SUMMARY_TIME_WINDOW)
    reference_hour = data.get("REFERENCE_HOUR", DEFAULT_REFERENCE_HOUR)
    max_brief_awakening = data.get("MAX_BRIEF_AWAKENING", DEFAULT_MAX_BRIEF_AWAKENING)
    offset = store_hour_start - reference_hour
    offset *= 3600
    summary_FUN = data.get("summary_FUN", "mean")

    params = AnalysisParams(
        time_window_length,
        velocity_correction_coef,
        min_time_immobile,
        summary_time_window,
        summary_FUN,
        max_brief_awakening,
        reference_hour,
        offset,
    )

    logger.info(params)

    return params

def load_params(store_metadata):

    ## Define plotting and analyze params
    analysis_params = get_analysis_params(store_metadata)

    chunks_per_hour = 3600 / ETHOGRAM_FREQUENCY
    chunk_index = {
        chunk: round(
            analysis_params.offset / 3600 + chunk * 1 / chunks_per_hour, 1
        )
        for chunk in store_metadata["chunks"]
    }
    plotting_params = PlottingParams(
        chunk_index=chunk_index, experiment_name=None,
        ld_annotation=None,
        number_of_animals=None,
        ethogram_frequency=ETHOGRAM_FREQUENCY
    )

    return analysis_params, plotting_params
