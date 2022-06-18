import argparse
import os
import os.path
import logging
import datetime

from flyhostel.plotting.environment import make_environment_plot
from flyhostel.quantification.imgstore import read_store_metadata
import numpy as np

logging.getLogger("flyhostel.sensors.io.plotting").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)


# TODO Move to another module
import glob
import pandas as pd
import pandas.api.types as ptypes

from imgstore.util import motif_extra_data_json_to_df
##



# TODO move from here to another module
def get_extra_data(store_path, ignore_corrupt_chunks=False):

    paths = sorted(
        glob.glob(
            os.path.join(
                store_path,
                "*.extra.json"
            )
        )
    )

    dfs = []
    for path in paths:
        try:
            dfs.append(motif_extra_data_json_to_df(None, path))
        except Exception as error:
            if ignore_corrupt_chunks:
                logger.warning(error)
            else:
                raise error
    
    extra_data = pd.concat(dfs, axis=0, ignore_index=True)

    extra_data["light"] = np.float64(extra_data["light"].values)


    assert ptypes.is_numeric_dtype(extra_data["light"]), "light is not numeric"
    assert ptypes.is_numeric_dtype(extra_data["temperature"]), "temperature is not numeric"
    assert ptypes.is_numeric_dtype(extra_data["humidity"]), "humidity is not numeric"
    assert ptypes.is_numeric_dtype(extra_data["time"]), "time is not numeric"
    assert ptypes.is_numeric_dtype(extra_data["frame_time"]), "frame_time is not numeric"
    assert ptypes.is_numeric_dtype(extra_data["frame_index"]), "frame_index is not numeric"
    return extra_data

    
def read_data(store_path):
    """
    Open imgstore and load environmental data to Python
    """

    imgstore_logger = logging.getLogger("imgstore")
    imgstore_logger.setLevel(logging.ERROR)
    imgstore_logger.setLevel(logging.WARNING)
    data = get_extra_data(store_path, ignore_corrupt_chunks=True)
    return data


def clean_data(data):
    """
    Remove missing data
    """
    data = data.loc[~np.isnan(data["humidity"])]
    data = data.loc[~np.isnan(data["temperature"])]
    data.reset_index(inplace=True)
    return data


def compute_zt0_offset(start_time, reference_hour):
    """
    Return ms passed from zt0 to experiment start
    """
    start_time = datetime.datetime.strptime(
        start_time,
        "%Y-%m-%dT%H:%M:%S.%f"
    )

    hour = start_time.hour + 1 # TODO Fix this timezone correction
    minute = start_time.minute
    second = start_time.second

    start_time  = hour * 3600 + minute * 60 + second

    zt0 = reference_hour * 3600
    offset = start_time - zt0
    offset_ms = offset * 1000
    return offset_ms

def discretize_light(data, threshold=None):
    if threshold is None:
        threshold = data["light"].mean()
        logger.info("Light threshold: ", threshold)
    
    data["L"] = [str(e)[0] for e in data["light"] > threshold]
    return data



def annotate_phase(data):
    data["L"] = [str(e)[0] for e in data["ZT"] / 3600000 % 24 < 12]
    return data



def load_data(store_path, reference_hour, threshold=None):

    # read data
    data = read_data(store_path)
    store_metadata = read_store_metadata(store_path)
    start_time = store_metadata["created_utc"]

    # clean
    data = clean_data(data)

    # create ZT column
    offset_ms = compute_zt0_offset(start_time, reference_hour)
    data["ZT"] = data["frame_time"] + offset_ms
    # annotate phase
    # data = discretize_light(data, threshold=threshold)
    data = annotate_phase(data)

    #
    data["t"] = data["ZT"] / 1000 # to seconds

    return data


def plot_data(root, data, **kwargs):
    

    make_environment_plot(
        root=root,
        data=data,
        **kwargs,
    )

def save_data(dest, data):
    data.to_csv(dest)


