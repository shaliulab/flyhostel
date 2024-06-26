import os.path
import logging
import shutil
import numpy as np

from flyhostel.data.idtrackerai import read_data
from flyhostel.plotting.synchrony import synchrony_plot
from flyhostel.plotting.sleep import sleep_plot
from flyhostel.plotting.velocity import velocity_plot

from flyhostel.plotting.ethogram import ethogram_plot
from flyhostel.constants import (
    BINNED,
    COLORS,
)
from flyhostel.quantification.constants import TRAJECTORIES_SOURCE
import yaml

from flyhostel.quantification.ethogram import prepare_data_for_ethogram_plot
from flyhostel.quantification.sleep import sleep_annotation_all


from flyhostel.quantification.utils import tidy_dataset_all, bin_apply, make_suffix, annotate_id
from flyhostel.quantification.params import ANALYSIS_PARAMS, load_params
from flyhostel.quantification.data_view import DataView

logger = logging.getLogger(__name__)


def process_data(positions, velocities, chunk_metadata, analysis_params, plotting_params):
    dt_raw, data = tidy_dataset_all(
        velocities,
        chunk_metadata=chunk_metadata,
        analysis_params=analysis_params,
        experiment_name=plotting_params.experiment_name
    )
    positions=positions.reshape((-1, 2))
    dt_raw["x"] = positions[:, 0]
    dt_raw["y"] = positions[:, 1]

    dt_sleep = sleep_annotation_all(data, analysis_params=analysis_params)
    dt_binned = bin_apply(dt_sleep, "asleep", analysis_params, keep_cols=["fly_no"])

    dt_binned["L"] = [
        "F" if e else "T" for e in (dt_binned["t"] / 3600) % 24 > 12
    ]
    dt_ethogram = prepare_data_for_ethogram_plot(data, analysis_params)
     
    return dt_raw, dt_sleep, dt_binned, dt_ethogram


from .parser import get_parser

def main(args=None, ap=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    output = args.output


    if args.interval is None:
        trajectories_source = os.path.join(args.imgstore_folder, f"{TRAJECTORIES_SOURCE}.yml")
        if os.path.exists(trajectories_source):
            with open(trajectories_source, "r") as filehandle:
                trajectories_source = yaml.load(filehandle, yaml.SafeLoader)
                trajectories = list(trajectories_source.values())
                trajectories = [int(t.replace(".npy", "")) for t in trajectories]
                interval = [min(trajectories), max(trajectories)]
        else:
            interval = None
       
    else:
        interval = args.interval

    os.makedirs(output, exist_ok=True)
    shutil.copyfile(ANALYSIS_PARAMS, os.path.join(output, ANALYSIS_PARAMS))

    experiment_name = os.path.basename(
        os.path.realpath(os.path.basename(args.imgstore_folder.rstrip("/")))
    )

    tr, velocities, chunks, store_metadata, chunk_metadata = read_data(args.imgstore_folder, interval, source=args.source, interpolate_nans=args.interpolate_nans, n_jobs=args.n_jobs)

    if getattr(tr, "concatenation", None) is not None: 
        np.savetxt(os.path.join(output, f"{os.path.basename(os.path.realpath(args.imgstore_folder))}_concatenation.csv"), tr.concatenation, fmt="%1.0f")
    np.save(os.path.join(output, f"{os.path.basename(os.path.realpath(args.imgstore_folder))}_trajectories.npy"), tr._s)
    np.save(os.path.join(output, f"{os.path.basename(os.path.realpath(args.imgstore_folder))}_frames.npy"), chunk_metadata[0])
    np.save(os.path.join(output, f"{os.path.basename(os.path.realpath(args.imgstore_folder))}_timestamps.npy"), chunk_metadata[1])
    #####
    noa = velocities.shape[1]

    analysis_params, plotting_params = load_params(store_metadata)
    suffix = make_suffix(analysis_params)
    plotting_params.number_of_animals = noa
    plotting_params.experiment_name = experiment_name
    plotting_params.ld_annotation = args.ld_annotation
    
    
    dt_raw, dt_sleep, dt_binned, dt_ethogram = process_data(tr.s, velocities, chunk_metadata, analysis_params, plotting_params)
    
    # make and save plots and data
    fig1 = sleep_plot(dt_binned,plotting_params=plotting_params)
    sleep_view = DataView(experiment_name, BINNED, dt_binned, fig1)
    sleep_view.save(output, suffix="")

    fig2 = ethogram_plot(
        dt_ethogram, analysis_params, plotting_params, 
        colors=COLORS,
        ncols=plotting_params.ethogram_frequency // analysis_params.time_window_length
    )
    
    ethogram_view = DataView(experiment_name, "ethogram", dt_ethogram, fig2)
    ethogram_view.save(output, suffix="")

    fig3 = synchrony_plot(dt_binned, plotting_params=plotting_params, y="asleep")
    synchrony_view = DataView(experiment_name, "synchrony", dt_binned, fig3)
    synchrony_view.save(output, suffix="")

    fig4 = velocity_plot(dt_raw, plotting_params=plotting_params)
    raw_view = DataView(experiment_name, "raw", dt_raw, fig4)
    raw_view.save(output, suffix="")
    
    annotation_view = DataView(experiment_name, "annotation",dt_sleep, None)
    annotation_view.save(output, suffix="")
    return 0
