import matplotlib.pyplot as plt
import pandas as pd

from flyhostel.quantification.constants import FLYHOSTEL_ID
from .core import geom_ld_annotation 

def synchrony_plot(dt_binned, y, plotting_params):

    for column in ["L", y, "t", FLYHOSTEL_ID]:
        assert column in dt_binned.columns, f"{column} is not available"


    dt_binned["diff"]=dt_binned.groupby(FLYHOSTEL_ID)[[y]].diff()
    # drop the first bin, for which no diff with the previous bin can be computed 
    dt_binned.dropna(inplace=True)
    sync_dt = pd.DataFrame(1 / (1 + dt_binned.groupby("t")["diff"].std())).rename(columns={"diff": "sync"})
    sync_dt.reset_index(inplace=True)
    sync_dt["L"] = [
        "F" if e else "T" for e in (sync_dt["t"] / 3600) % 24 > 12
    ]

    fig = plt.figure(3, figsize=(10, 7), dpi=90, facecolor="white")
    Y_RANGE=(0,1)

    ax=fig.add_subplot(int("111"))

    if plotting_params.ld_annotation:
        # plot the phases (LD)
        ax = geom_ld_annotation(sync_dt, ax, yrange=Y_RANGE)

    ax.plot(sync_dt["t"] / 3600, sync_dt["sync"])
    ax.set_ylim(Y_RANGE)
    ax.set_yticks([0, .5, 1])
    ax.set_yticklabels(["0", "50", "100"])
    ax.set_xlabel("ZT")
    return fig
    

