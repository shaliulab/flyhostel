import argparse
import os
import os.path
import logging
import re

from flyhostel.sensors.io.plotting import make_environmental_plot
import zeitgeber

import numpy as np
import imgstore

logging.getLogger("flyhostel.sensors.io.plotting").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument(
        "--experiment-folder", "--input", dest="input", type=str, required=True
    )
    ap.add_argument(
        "--output", dest="output", type=str, default="."
    )
    ap.add_argument(
        "--reference_hour",
        "--zt0",
        dest="reference_hour",
        type=float,
        required=True,
    )

    ap.add_argument(
        "--light-threshold",
        dest="light_threshold",
        type=int,
        default=None,
    )
    return ap


def read_data(store_path):
    """
    Open imgstore and load environmental data to Python
    """

    imgstore_logger = logging.getLogger("imgstore")
    imgstore_logger.setLevel(logging.ERROR)
    store = imgstore.new_for_filename(store_path)
    imgstore_logger.setLevel(logging.WARNING)
    data = store.get_extra_data(ignore_corrupt_chunks=True)
    return store, data


def discretize_light(data, threshold=None):
    if threshold is None:
        threshold = data["light"].mean()
        logger.info("Light threshold: ", threshold)
    
    data["L"] = [str(e)[0] for e in data["light"] > threshold]
    return data


def clean_data(data):
    """
    Remove missing data
    """
    data = data.loc[~np.isnan(data["humidity"])]
    data = data.loc[~np.isnan(data["temperature"])]
    data.reset_index(inplace=True)
    return data


def compute_zt0_offset(store, reference_hour):
    """
    Return ms passed from zt0 to experiment start
    """
    experiment_name = os.path.basename(os.path.dirname(store.full_path))
    datetime_str = re.match(
        "^([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}).*",
        experiment_name,
    ).group(1)

    start_time = zeitgeber.seconds_since_midnight(datetime_str)

    zt0 = reference_hour * 3600
    offset = start_time - zt0
    offset_ms = offset * 1000
    return offset_ms


def load_data(store_path, reference_hour, threshold=None):

    # read data
    store, data = read_data(store_path)

    # clean
    data = clean_data(data)

    # create ZT column
    offset_ms = compute_zt0_offset(store, reference_hour)
    data["ZT"] = data["frame_time"] + offset_ms
    # annotate phase
    data = discretize_light(data, threshold=threshold)

    #
    data["t"] = data["ZT"] / 1000 # to seconds

    return store, data


def plot_data(root, data, **kwargs):
    

    make_environmental_plot(
        root=root,
        data=data,
        **kwargs,
    )

def save_data(dest, data):
    data.to_csv(dest)


def main(args=None, ap=None):

    if args is None:
        if ap is None:
            ap = get_parser()

        args = ap.parse_args()

    store, data = load_data(
        store_path=args.input,
        reference_hour=args.reference_hour,
        threshold=args.light_threshold,
    )

    os.makedirs(args.output, exist_ok=True)

    experiment_date = os.path.basename(os.path.dirname(store.full_path))
    dest=os.path.join(
        args.output,
        f"{experiment_date}_environment-log.csv"
    )

    save_data(dest, data)
    root=os.path.join(
        args.output,
        f"{experiment_date}"
    )
    plot_data(root, data, title=experiment_date)

if __name__ == "__main__":
    main()
