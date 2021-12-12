import argparse
import os.path
import itertools
import datetime
from typing_extensions import Required
import warnings
import logging
from pprint import pprint
from collections import namedtuple

AnalysisParams = namedtuple("AnalysisParams", [
    "time_window_length",
    "velocity_correction_coef",
    "min_time_immobile",
    "summary_time_window",
    "sumary_FUN",
    "reference_hour",
    "offset"
])

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


from zeitgeber import rle # https://github.com/shaliulab/zeitgeber

from flyhostel.sensors.io.plotting import geom_ld_annotations
from flyhostel.quantification.trajectorytools import load_trajectories
from flyhostel.quantification.imgstore import read_store_metadata

def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-folder", "--input", dest="input", required=True, type=str)
    
    return ap

def sleep_annotation(data, analysis_params):
    data["moving"] = data["velocity"] > analysis_params.velocity_correction_coef
    data_annot = []
    for i in np.unique(data["id"]):
        d=data.loc[data["id"] == i,:]
        d["window_number"] = np.arange(d.shape[0])
        rle=rle.decompose((~d["moving"]).tolist())
        d["asleep"] =  list(
            itertools.chain(*[
                # its asleep if
                # 1. it is not moving (x[0])
                # 2. if the length of the not moving state (x[1]) is >= the ratio between
                # min_time_immobile and time_window_length (i.e. the minimum number of windows)
                [x[0] and x[1] >= (analysis_params.min_time_immobile / analysis_params.time_window_length), ]
                * x[1] for x in rle
            ])
        )

        d["t"] = d["t_round"]
        d.drop("t_round", axis=1, inplace=True)
        data_annot.append(d)
    
    dt_sleep = pd.concat(data_annot)
    return dt_sleep


def bin_apply(dt, analysis_params):
    dt["t_bin"] = np.floor(
        dt["t"] / analysis_params.summary_time_window
    ) * analysis_params.summary_time_window

    dt=dt.groupby(["id", "t_bin"])
    dt_binned = getattr(dt, analysis_params.sumary_FUN)()
    dt_binned=dt_binned.reset_index()[["id", "t_bin", "asleep"]]
    dt_binned.columns = ["id", "t", "asleep"]
    return dt_binned


def get_analysis_params(store_metadata):

    date_format = "%Y-%m-%dT%H:%M:%S.%f"

    store_datetime = datetime.datetime.strptime(store_metadata["created_utc"], date_format)
    store_hour_start = store_datetime.hour + store_datetime.minute/60 + store_datetime.second / 3600
    
    time_window_length=10
    velocity_correction_coef=2
    min_time_immobile=300
    summary_time_window=30*60
    reference_hour = 6
    offset = store_hour_start - reference_hour
    offset *= 3600
    summary_FUN = "mean"


    params = AnalysisParams(time_window_length, velocity_correction_coef, min_time_immobile, summary_time_window, summary_FUN, reference_hour, offset)
    return params


def init_data_frame():
    return pd.DataFrame({
            "velocity": [],
            "frame_number": [],
            "t": [],
            "t_round": [],
            "id": [],
            "L": [],
        })


def sleep_plot(dt):
    idents = dt["id"].unique()
    fig = plt.figure(1, figsize=(10,7), dpi=90, facecolor="white")
    ncol=1
    nrow=len(idents)
    axes = []

    for i, ident in enumerate(idents):
        int_str = f"{nrow}{ncol}{i+1}"
        ax = fig.add_subplot(int(int_str))
        axes.append(ax)

        # take one fly     
        dt_one_fly = dt.loc[dt["id"] == ident]
        
        # plot the data
        ax.plot(dt_one_fly["t"] / 3600, dt_one_fly["asleep"], linewidth=1, c="blue")
        ax.set_ylim([0, 1])

        # plot the phases (LD)
        
        geom_ld_annotations(dt_one_fly, ax, yrange=[0,1])
    # fig.subplots_adjust(bottom=0.0, right=0.8, top=1.0)
    return fig


def main(ap=None, args=None):


    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()


    # load trajectories
    status, tr = load_trajectories(args.input)

    # load metadata
    store_metadata, chunk_metadata = read_store_metadata(args.input)    
    analysis_params = get_analysis_params(store_metadata)
    frame_number = list(itertools.chain(*[m["frame_number"] for m in chunk_metadata.values()]))
    frame_time = list(itertools.chain(*[m["frame_time"] for m in chunk_metadata.values()]))

    # Let assume that 50 pixels in the video frame are 1 cm.
    tr.new_length_unit(250, 'cm')

    # Since we have the frames per second stored int the tr.params dictionary we will use them to 
    tr.new_time_unit(tr.params['frame_rate'], 's')
    
    velocities = np.abs(tr.v).sum(axis=2)

    # initialized data_frame
    data=init_data_frame()

    for i in tqdm(range(velocities.shape[1])):
        d = pd.DataFrame({
            "velocity": velocities[:,i],
            "frame_number": frame_number[1:-1]
        })
        
        d["frame_time"] = [frame_time[i] for i in d["frame_number"]]
        d["t"] = d["frame_time"]
        d["t"] /= 1000 # to seconds
        d["t"] += analysis_params.offset
        d["L"] = ["T" if e else "F" for e in ((d["t"] / 3600) % 24) < 12]
        d["t_round"] = np.floor(d["t"]/analysis_params.time_window_length)*analysis_params.time_window_length
        d["id"] = [i,] * d.shape[0]
        d.drop("frame_time",axis=1, inplace=True)

        data=pd.concat([data, d])
        

    data=data.groupby(["id", "t_round"]).max().reset_index()[["velocity", "id", "t_round", "L"]]
    data.to_csv("data.csv")
    dt_sleep = sleep_annotation(data, analysis_params)
    dt_sleep.to_csv("dt_sleep.csv")

    dt_binned = bin_apply(dt_sleep, analysis_params)

    fig = sleep_plot(dt_binned)


    experiment_name = os.path.basename(args.input.rstrip("/"))
    fig.suptitle(f"Fly Hostel - {experiment_name}")
    fig.savefig(experiment_name + "-facet" + ".png", transparent=False)

  

if __name__ == "__main__":
    # args = argparse.Namespace(experiment_folder = "/Dropbox/FlySleepLab_Dropbox/Data/flyhostel_data/videos/2021-11-27_12-02-38/")
    # main(args=args)
    main()
