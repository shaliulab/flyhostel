import argparse
import os.path
import itertools
import datetime
import logging
# from collections import namedtuple
from recordtype import recordtype
import functools
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import yaml
import zeitgeber # https://github.com/shaliulab/zeitgeber

from flyhostel.sensors.io.plotting import geom_ld_annotations
from flyhostel.quantification.trajectorytools import load_trajectories
from flyhostel.quantification.imgstore import read_store_metadata
from flyhostel.utils import add_suffix
from flyhostel.constants import *


PlottingParams = recordtype(
    "PlottingParams",
    [
        "chunk_index",
        "experiment_name",
        "ld_annotation",
        "number_of_animals",
    ],
)
AnalysisParams = recordtype(
    "AnalysisParams",
    [
        "time_window_length",
        "velocity_correction_coef",
        "min_time_immobile",
        "summary_time_window",
        "sumary_FUN",
        "reference_hour",
        "offset",
    ],
)

#N_JOBS = -2
N_JOBS = 1
FREQ = 300

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument(
        "--imgstore-folder", dest="imgstore_folder", required=True, type=str
    )
    ap.add_argument(
        "--analysis-folder", dest="analysis_folder", default=None, type=str
    )

    ap.add_argument("--interval", nargs="+", type=int, required=False, default=None)

    ap.add_argument("--output", dest="output", default=None, type=str)
    ap.add_argument(
        "--ld-annotation",
        dest="ld_annotation",
        action="store_true",
        default=True,
    )
    ap.add_argument(
        "--no-ld-annotation",
        dest="ld_annotation",
        action="store_false",
        default=True,
    )
    return ap


def sleep_annotation(data, analysis_params):

    data["moving"] = (
        data["velocity"] > analysis_params.velocity_correction_coef
    )
    data["window_number"] = np.arange(data.shape[0])
    rle = zeitgeber.rle.encode((~data["moving"]).tolist())
    data["asleep"] = list(
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

    data["t"] = data["t_round"]
    data.drop("t_round", axis=1, inplace=True)
    return data


def sleep_annotation_all(data, **kwargs):

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

    dt_sleep = pd.concat(data_annot)
    return dt_sleep


def bin_apply(dt, analysis_params):
    dt["t_bin"] = (
        np.floor(dt["t"] / analysis_params.summary_time_window)
        * analysis_params.summary_time_window
    )

    dt = dt.groupby(["id", "t_bin"])
    dt_binned = getattr(dt, analysis_params.sumary_FUN)()
    dt_binned["L"] = [
        "F" if e else "T" for e in (dt_binned["t"] / 3600) % 24 > 12
    ]
    dt_binned = dt_binned.reset_index()[["id", "t_bin", "asleep", "L"]]
    dt_binned.columns = ["id", "t", "asleep", "L"]
    return dt_binned


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
        with open("./analysis_params.yaml", "r") as filehandle:
            data = yaml.load(filehandle, yaml.SafeLoader)
    except:
        logger.warning("No analysis_params.yaml detected. Using defaults")
        data = {}

    time_window_length = data.get("TIME_WINDOW_LENGTH", DEFAULT_TIME_WINDOW_LENGTH)
    velocity_correction_coef = data.get("VELOCITY_CORRECTION_COEF", DEFAULT_VELOCITY_CORRECTION_COEF) # cm / second
    min_time_immobile = data.get("MIN_TIME_IMMOBILE", DEFAULT_MIN_TIME_IMMOBILE)
    summary_time_window = data.get("SUMMARY_TIME_WINDOW", DEFAULT_SUMMARY_TIME_WINDOW)
    reference_hour = data.get("REFERENCE_HOUR", DEFAULT_REFERENCE_HOUR)
    offset = store_hour_start - reference_hour
    offset *= 3600
    summary_FUN = data.get("summary_FUN", "mean")

    params = AnalysisParams(
        time_window_length,
        velocity_correction_coef,
        min_time_immobile,
        summary_time_window,
        summary_FUN,
        reference_hour,
        offset,
    )
    return params


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


def sleep_plot(dt, plotting_params):

    for column in ["L", "asleep", "t", "id"]:
        assert column in dt.columns, f"{column} is not available"

    idents = dt["id"].unique()
    fig = plt.figure(1, figsize=(10, 7), dpi=90, facecolor="white")
    ncol = 1
    nrow = len(idents)
    axes = []
    Y_RANGE = (0, 1)

    for i, ident in enumerate(idents):
        int_str = f"{nrow}{ncol}{i+1}"
        axes.append(fig.add_subplot(int(int_str)))

        # take one fly
        dt_one_fly = dt.loc[dt["id"] == ident].reset_index()

        if plotting_params.ld_annotation:
            # plot the phases (LD)
            geom_ld_annotations(dt_one_fly, axes[i], yrange=Y_RANGE)

        # plot the data
        axes[i].plot(
            dt_one_fly["t"] / 3600, dt_one_fly["asleep"], linewidth=1, c="blue"
        )

        axes[i].set_ylim(Y_RANGE)
        axes[i].set_yticks([0, .5, 1])
        axes[i].set_yticklabels(["0", "50", "100"])
        axes[i].set_xlabel("ZT")

    # fig.subplots_adjust(bottom=0.0, right=0.8, top=1.0)
    fig.suptitle(f"Fly Hostel - {plotting_params.experiment_name}")
    return fig


def tidy_dataset(velocity, chunk_metadata, analysis_params):

    frame_number, frame_time = chunk_metadata

    # wrapping around dataframe() to avoid
    # annoying warning
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


def f(hex_code):
    h = hex_code.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def make_waffle_array(timeseries, ncols, scale=False):

    if scale:
        max_val = timeseries.max()
        timeseries *= 255 / max_val

    if len(timeseries.shape) == 1:
        timeseries = timeseries.reshape((*timeseries.shape, 1))
        nchannels = 1
    else:
        nchannels = timeseries.shape[1]

    nrows = int(timeseries.shape[0] / ncols)
    timeseries = timeseries[: nrows * ncols]
    timeseries = timeseries.reshape((nrows, ncols, nchannels))

    return timeseries


def waffle_plot(
    fig, i, timeseries, plotting_params, ncols, freq=300, **kwargs
):

    int_str = f"1{plotting_params.number_of_animals}{int(i)+1}"
    ax = fig.add_subplot(int(int_str))

    timeseries = make_waffle_array(
        timeseries=timeseries,
        ncols=ncols,
        **kwargs
    )

    if ax is None:
        ax = plt

    nrows, ncols = timeseries.shape[:2]
    pos = list(range(0, 1+int(nrows / 6) * 6, 3600 // freq))

    ticks = []
    positions = []
    for p in pos:
        if p in plotting_params.chunk_index:
            ticks.append(plotting_params.chunk_index[p])
            positions.append(p)

    if ticks is not None:
        ax.set_yticks(positions, ticks)
        ax.set_xticks([0, ncols-1], [0, freq])
        ax.set_xlabel("Time in chunk (s)")

    ax.imshow(timeseries)

    if i == 0:
        ax.set_ylabel("ZT")

    return ax


def waffle_plot_all(data, analysis_params, plotting_params):

    fig = plt.figure(2, figsize=(12, 7), dpi=90, facecolor="white")
    plt.axis("off")
    plt.title(plotting_params.experiment_name)

    ncols = FREQ // analysis_params.time_window_length
    colors = {"T": ["#F9A825", "#FFF59D"], "F": ["#4527A0", "#CE93D8"]}
    colors = {k: [f(v) for v in colors[k]] for k in colors}

    for i in np.unique(data["id"]):
        timeseries = prepare_data_for_waffle_plot(
            data, i, analysis_params, colors=colors
        )

        ax = waffle_plot(
            fig,
            i,
            timeseries,
            plotting_params=plotting_params,
            ncols=ncols,
            scale=False,
            freq=FREQ,
        )

    return fig


def prepare_data_for_waffle_plot(data, i, analysis_params, colors):

    timeseries = data.loc[data["id"] == i]["velocity"].values
    timeseries_phase = data.loc[data["id"] == i]["L"].values
    timeseries = timeseries < analysis_params.velocity_correction_coef
    timeseries = timeseries * 1
    timeseries = np.array(
        [
            colors[timeseries_phase[i]][timeseries[i]]
            for i in range(len(timeseries))
        ]
    )
    return timeseries

@functools.lru_cache(maxsize=100, typed=False)
def read_data(imgstore_folder, analysis_folder, interval=None):

    # Load trajectories
    status, chunks, tr = load_trajectories(analysis_folder, interval=interval)

    # Load metadata
    store_metadata, chunk_metadata = read_store_metadata(
        imgstore_folder, chunk_numbers=chunks
    )
    pixels_per_cm = store_metadata["pixels_per_cm"]
    tr.new_length_unit(pixels_per_cm, "cm")

    return tr, chunks, store_metadata, chunk_metadata


def load_params(store_metadata):

    ## Define plotting and analyze params
    analysis_params = get_analysis_params(store_metadata)

    chunks_per_hour = 3600 / FREQ
    chunk_index = {
        chunk: round(
            analysis_params.offset / 3600 + chunk * 1 / chunks_per_hour, 1
        )
        for chunk in store_metadata["chunks"]
    }
    plotting_params = PlottingParams(
        chunk_index=chunk_index, experiment_name=None,
        ld_annotation=None,
        number_of_animals=None
    )

    return analysis_params, plotting_params


def process_data(tr, analysis_params, chunk_metadata):
    ## Process dataset
    logger.info("Computing velocity")
    velocities = np.abs(tr.v).sum(axis=2)

    logger.info("Tidying dataset")
    data = tidy_dataset_all(
        velocities,
        chunk_metadata=chunk_metadata,
        analysis_params=analysis_params,
    )

    logger.info(f"Annotating sleep behavior")
    dt_sleep = sleep_annotation_all(data, analysis_params=analysis_params)

    logger.info(
        f"Binning data every {analysis_params.summary_time_window/60} minutes"
    )

    dt_binned = bin_apply(dt_sleep, analysis_params)

    return data, dt_sleep, dt_binned


def plot_data(data, dt_binned, analysis_params, plotting_params, suffix=""):
    """
    Plot a sleep trace plot and a waffle plot
    """

    ## Save and plot results
    logger.info("Building plot")
    fig1 = sleep_plot(
        dt_binned,
        plotting_params=plotting_params
    )
    plot1 = (add_suffix(plotting_params.experiment_name + "-facet.png", suffix), fig1)
    fig2 = waffle_plot_all(data, analysis_params, plotting_params)
    plot2 = (add_suffix(plotting_params.experiment_name + "-waffle.png", suffix), fig2)

    return plot1, plot2


def save_results(data, dt_sleep, dt_binned, plot1, plot2, output, suffix=""):

    logger.info("Saving results ...")
    os.makedirs(output, exist_ok=True)
    data.to_csv(os.path.join(output, add_suffix(RAW_DATA, suffix)))
    dt_sleep.to_csv(os.path.join(output, add_suffix(ANNOTATED_DATA, suffix)))
    dt_binned.to_csv(os.path.join(output, add_suffix(BINNED_DATA, suffix)))

    path2, fig2 = plot2
    path2 = os.path.join(output, path2)
    logger.info(f"Saving plot to {path2}")
    fig2.savefig(path2, transparent=False)

    path1, fig1 = plot1
    path1 = os.path.join(output, path1)
    logger.info(f"Saving plot to {path1}")
    fig1.savefig(path1, transparent=False)


def make_suffix(analysis_params):
    suffix = f"{str(analysis_params.velocity_correction_coef * 1000).zfill(5)}"
    f"-{analysis_params.min_time_immobile}"
    f"-{analysis_params.time_window_length}"
    f"-{analysis_params.summary_time_window}"
    return suffix


def main(args=None, ap=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()


    if args.analysis_folder is None:
        args.analysis_folder = os.path.join(
            args.imgstore_folder,
            "idtrackerai"
        )
    if args.output is None:
        args.output = os.path.join(
            args.imgstore_folder,
            "output"
        )

    experiment_name = os.path.basename(args.imgstore_folder.rstrip("/"))

    tr, chunks, store_metadata, chunk_metadata = read_data(args.imgstore_folder, args.analysis_folder, tuple(args.interval))
    analysis_params, plotting_params = load_params(store_metadata)
    suffix = make_suffix(analysis_params)
    plotting_params.number_of_animals = tr.s.shape[1]
    plotting_params.experiment_name = experiment_name
    plotting_params.ld_annotation = args.ld_annotation

    data, dt_sleep, dt_binned = process_data(tr, analysis_params, chunk_metadata)
    plot1, plot2 = plot_data(data, dt_binned, analysis_params, plotting_params, suffix=suffix)
    save_results(data, dt_sleep, dt_binned, plot1, plot2, args.output, suffix=suffix)

    return 0


if __name__ == "__main__":
    # args = argparse.Namespace(experiment_folder = "/Dropbox/FlySleepLab_Dropbox/Data/flyhostel_data/videos/2021-11-27_12-02-38/")
    # main(args=args)
    main()
