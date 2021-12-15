import argparse
import os.path
import itertools
import datetime
import logging
from collections import namedtuple

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import zeitgeber  # https://github.com/shaliulab/zeitgeber

from flyhostel.sensors.io.plotting import geom_ld_annotations
from flyhostel.quantification.trajectorytools import load_trajectories
from flyhostel.quantification.imgstore import read_store_metadata


PlottingParams = namedtuple(
    "PlottingParams",
    [
        "chunk_index",
        "experiment_name",
    ],
)
AnalysisParams = namedtuple(
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

N_JOBS = -2
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
        "--analysis-folder", dest="analysis_folder", required=True, type=str
    )


    ap.add_argument("--output", dest="output", required=True, type=str)
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

    time_window_length = 10
    velocity_correction_coef = 2
    min_time_immobile = 300
    summary_time_window = 30 * 60
    reference_hour = 6
    offset = store_hour_start - reference_hour
    offset *= 3600
    summary_FUN = "mean"

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


def sleep_plot(dt, plotting_params, ld_annotation=True):

    for column in ["L", "asleep", "t", "id"]:
        assert column in dt.columns, f"{column} is not available"

    idents = dt["id"].unique()
    fig = plt.figure(1, figsize=(10, 7), dpi=90, facecolor="white")
    ncol = 1
    nrow = len(idents)
    axes = []

    for i, ident in enumerate(idents):
        int_str = f"{nrow}{ncol}{i+1}"
        ax = fig.add_subplot(int(int_str))
        axes.append(ax)

        # take one fly
        dt_one_fly = dt.loc[dt["id"] == ident].reset_index()

        # plot the data
        ax.plot(
            dt_one_fly["t"] / 3600, dt_one_fly["asleep"], linewidth=1, c="blue"
        )
        ax.set_ylim([0, 1])

        if ld_annotation:
            # plot the phases (LD)
            geom_ld_annotations(dt_one_fly, ax, yrange=[0, 1])

    fig.subplots_adjust(bottom=0.0, right=0.8, top=1.0)
    fig.suptitle(f"Fly Hostel - {plotting_params.experiment_name}")
    return fig


def tidy_dataset(velocity, chunk_metadata, analysis_params):

    frame_number, frame_time = chunk_metadata

    # wrapping around dataframe() to avoid
    # annoying warning
    data = pd.DataFrame(
        {"velocity": velocity, "frame_number": frame_number[1:-1]}
    )

    data["frame_time"] = [frame_time[i] for i in data["frame_number"]]
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


def waffle_plot(
    fig, i, timeseries, plotting_params, ncols, freq=300, scale=False
):

    int_str = f"16{int(i)+1}"
    ax = fig.add_subplot(int(int_str))

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
    if ax is None:
        ax = plt

    pos = list(range(0, int(nrows / 100) * 100, 3600 // freq))
    ticks = [plotting_params.chunk_index[p] for p in pos]

    if ticks is not None:
        ax.set_yticks(pos, ticks)

    ax.imshow(timeseries)

    if i == 0:
        ax.set_ylabel("ZT")

    return ax


def waffle_plot_all(data, analysis_params, plotting_params):

    fig = plt.figure(2, figsize=(10, 7), dpi=90, facecolor="white")
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


def main(args=None, ap=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    if args.imgstore_folder == ".":
        input = os.getcwd()
    else:
        input = args.imgstore_folder

    experiment_name = os.path.basename(input.rstrip("/"))

    # Load trajectories
    status, chunks, tr = load_trajectories(args.analysis_folder)
    # Load metadata
    store_metadata, chunk_metadata = read_store_metadata(
        input, chunk_numbers=chunks
    )

    ## Define plotting and analyze params
    analysis_params = get_analysis_params(store_metadata)

    chunks_per_hour = 3600 / FREQ
    chunk_index = {
        chunk: round(
            analysis_params.offset / 3600 + chunk * 1 / chunks_per_hour, 1
        )
        for chunk in chunks
    }
    plotting_params = PlottingParams(
        chunk_index=chunk_index, experiment_name=experiment_name
    )


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

    ## Save and plot results

    logger.info("Building plot")
    fig = sleep_plot(
        dt_binned,
        plotting_params=plotting_params,
        ld_annotation=args.ld_annotation,
    )
    fig2 = waffle_plot_all(data, analysis_params, plotting_params)

    logger.info("Saving results ...")
    os.makedirs(args.output, exist_ok=True)
    data.to_csv(os.path.join(args.output, "data.csv"))
    dt_sleep.to_csv(os.path.join(args.output, "dt_sleep.csv"))
    dt_binned.to_csv(os.path.join(args.output, "dt_binned.csv"))

    path = os.path.join(args.output, experiment_name + "-waffle" + ".png")
    logger.info(f"Saving plot to {path}")
    fig2.savefig(path, transparent=False)

    path = os.path.join(args.output, experiment_name + "-facet" + ".png")
    logger.info(f"Saving plot to {path}")
    fig.savefig(path, transparent=False)

    return 0


if __name__ == "__main__":
    # args = argparse.Namespace(experiment_folder = "/Dropbox/FlySleepLab_Dropbox/Data/flyhostel_data/videos/2021-11-27_12-02-38/")
    # main(args=args)
    main()
