import dataclasses
import os.path
import logging

from flyhostel.utils import add_suffix
from flyhostel.data.idtrackerai import read_data
from flyhostel.plotting.synchrony import synchrony_plot
from flyhostel.plotting.sleep import sleep_plot
from flyhostel.plotting.ethogram import ethogram_plot
from flyhostel.constants import (
    RAW,
    ANNOTATED,
    BINNED,
    COLORS,
)
from flyhostel.quantification.constants import FLYHOSTEL_ID, TRAJECTORIES_SOURCE
import yaml

from .ethogram import prepare_data_for_ethogram_plot
from .sleep import sleep_annotation_all


from .parser import get_parser
from .utils import tidy_dataset_all, bin_apply, make_suffix, annotate_id
from .params import load_params

logger = logging.getLogger(__name__)


class DataView:

    def __init__(self, experiment, name, data, fig=None):
        self.experiment=experiment
        self.name = name
        self.data = data
        self.fig = fig

        self._csv_path = os.path.join(f"{experiment}_{name}.csv")
        self._fig_path = os.path.join(f"{experiment}_{name}.png")

    def save(self, output, suffix):

        columns = self.data.columns.tolist()
        columns.pop(columns.index(FLYHOSTEL_ID))
        columns.insert(0, FLYHOSTEL_ID)

        data = self.data[columns]

        data.to_csv(
            add_suffix(os.path.join(output, self._csv_path), suffix)
        )

        if self.fig is not None:
            self.fig.savefig(
                add_suffix(os.path.join(output, self._fig_path), suffix),
                transparent=False
            )
            self.fig.clear()


def process_data(velocities, chunk_metadata, analysis_params, plotting_params):
    dt_raw, data = tidy_dataset_all(
        velocities,
        chunk_metadata=chunk_metadata,
        analysis_params=analysis_params,
        experiment_name=plotting_params.experiment_name
    )

    dt_sleep = sleep_annotation_all(data, analysis_params=analysis_params)
    dt_binned = bin_apply(dt_sleep, "asleep", analysis_params, keep_cols=["fly_no"])

    dt_binned["L"] = [
        "F" if e else "T" for e in (dt_binned["t"] / 3600) % 24 > 12
    ]
    dt_ethogram = prepare_data_for_ethogram_plot(data, analysis_params)        
    return dt_raw, dt_sleep, dt_binned, dt_ethogram


def main(args=None, ap=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    if args.output is None:
        output = os.path.join(
            args.imgstore_folder,
            "output"
        )
    else:
        output = args.output


    if args.interval is None:
        trajectories_source = os.path.join(args.imgstore_folder, f"{TRAJECTORIES_SOURCE}.yml")
        with open(trajectories_source, "r") as filehandle:
            trajectories_source = yaml.load(filehandle, yaml.SafeLoader)
            trajectories = list(trajectories_source.values())
            trajectories = [int(t.replace(".npy", "")) for t in trajectories]
            interval = [min(trajectories), max(trajectories)]
       
    else:
        interval = args.interval

    os.makedirs(output, exist_ok=True)

    experiment_name = os.path.basename(
        os.path.realpath(os.path.basename(args.imgstore_folder.rstrip("/")))
    )
    tr, velocities, chunks, store_metadata, chunk_metadata = read_data(args.imgstore_folder, tuple(interval))


    # TODO: Format this into a clean function or something
    import numpy as np
    np.save(os.path.join(output, f"{os.path.basename(os.path.realpath(args.imgstore_folder))}_trajectories.npy"), tr._s)
    np.save(os.path.join(output, f"{os.path.basename(os.path.realpath(args.imgstore_folder))}_timestamps.npy"), chunk_metadata[1])

    noa = velocities.shape[1]
    analysis_params, plotting_params = load_params(store_metadata)
    suffix = make_suffix(analysis_params)
    plotting_params.number_of_animals = noa
    plotting_params.experiment_name = experiment_name
    plotting_params.ld_annotation = args.ld_annotation

    dt_raw, dt_sleep, dt_binned, dt_ethogram = process_data(velocities, chunk_metadata, analysis_params, plotting_params)

    # make and save plots and data
    fig1 = sleep_plot(dt_binned,plotting_params=plotting_params)
    sleep_view = DataView(experiment_name, BINNED, dt_binned, fig1)
    sleep_view.save(output, suffix=suffix)

    fig2 = ethogram_plot(
        dt_ethogram, analysis_params, plotting_params, 
        colors=COLORS,
        ncols=plotting_params.ethogram_frequency // analysis_params.time_window_length
    )
    
    ethogram_view = DataView(experiment_name, "ethogram", dt_ethogram, fig2)
    ethogram_view.save(output, suffix=suffix)

    fig3 = synchrony_plot(dt_binned, plotting_params=plotting_params, y="asleep")
    synchrony_view = DataView(experiment_name, "synchrony", dt_binned, fig3)
    synchrony_view.save(output, suffix=suffix)

    raw_view = DataView(experiment_name, "raw", dt_raw, None)
    raw_view.save(output, suffix="raw")
    
    annotation_view = DataView(experiment_name, "annotation",dt_sleep, None)
    annotation_view.save(output, suffix="annotation")
    return 0


if __name__ == "__main__":
    # args = argparse.Namespace(experiment_folder = "/Dropbox/FlySleepLab_Dropbox/Data/flyhostel_data/videos/2021-11-27_12-02-38/")
    # main(args=args)
    main()
