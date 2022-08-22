import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from flyhostel.quantification.constants import FLYHOSTEL_ID

from .core import geom_ld_annotation 

def bin(dt, y, time_window_length=10, stat="max"):

    dt["t_round"] = np.floor(dt["t"] / time_window_length) * time_window_length
    dt_stat=dt.groupby("t_round").aggregate({y: getattr(np, stat)}).rename(columns={y: f"{stat}_{y}"})

    dt = pd.merge_asof(
        dt_stat,
        dt,
        left_on="t_round",
        right_on="t_round",
        direction="forward",
        tolerance=10
    )
    return dt

def velocity_plot(dt, plotting_params):

    ids = dt[FLYHOSTEL_ID].unique()
    fig = plt.figure(4, figsize=(10, 7), dpi=90, facecolor="white")
    ncol = 1
    nrow = len(ids)
    axes = []
    Y_RANGE=[0, 10]

    for i, id in enumerate(ids):
        int_str = f"{nrow}{ncol}{i+1}"
        axes.append(fig.add_subplot(int(int_str)))

        dt_one_fly = bin(
            dt.loc[dt[FLYHOSTEL_ID] == id].reset_index(),
            "velocity",
            10,
            "max"
        )

        # plot the data
        axes[i].plot(
            dt_one_fly["t"] / 3600, dt_one_fly["velocity"], linewidth=1, c="blue"
        )

        axes[i].set_xlabel("ZT")
        axes[i].set_ylim(Y_RANGE)

    # fig.subplots_adjust(bottom=0.0, right=0.8, top=1.0)
    fig.suptitle(f"Fly Hostel - {plotting_params.experiment_name}")
    return fig
