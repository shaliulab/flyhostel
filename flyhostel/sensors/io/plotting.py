import numpy as np
from shapely.geometry import Polygon
from descartes import PolygonPatch  

import zeitgeber
import matplotlib.pyplot as plt

def geom_ld_annotations(data, ax, yrange=[0, 100], xtick_freq=6):
    """
    data must have:
    
    Column L with either T or F (str)
    Column t with ZT in hours (float)
    
    """

    values, pos = zeitgeber.rle.decompose(data["L"].values.tolist())
    print(values)
    print(pos)
    zts = []
    for i in pos:
        zts.append(round(data.loc[i]["t"], 2))

    max_t = data["t"].tail().values[-1]
    min_t = zts[0]              
    y_max  = yrange[1]
    color = {"F": (0, 0, 0), "T": (1, 1, 1)}
    for i in range(len(zts)):
        if (i + 1) == len(zts):
            ring_mixed = Polygon([(zts[i], 0), (max_t, 0), (max_t, y_max), (zts[i], y_max)])        
        else:
            ring_mixed = Polygon([(zts[i], 0), (zts[i+1], 0), (zts[i+1], y_max), (zts[i], y_max)])
        ring_patch = PolygonPatch(ring_mixed, facecolor=color[values[i]], alpha=0.2, edgecolor=(0,0,0))
        ax.add_patch(ring_patch)

    xrange = [int(np.floor(min_t)), int(np.ceil(max_t))]
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    xticks = np.arange(xtick_freq* (int(min_t) // xtick_freq), xtick_freq*(int(max_t + xtick_freq) // xtick_freq), xtick_freq)
    ax.set_xticks(xticks)

    ax.set_xlabel("ZT")
    
    return ax

def geom_env_data(data, ax):
    
    ax2 = ax.twinx()
    ax2.set_ylabel("Temp ºC")
    ax.set_ylabel("% Hum")

    # ax.set_xticks(list(range(*xrange)) + [xrange[-1]])
    # ax.set_yticks(list(range(*yrange)) + [yrange[-1]])
    # ax.set_aspect(1)
    ax.scatter(data["t"], data["humidity"], s=.1)
    ax2.scatter(data["t"], data["temperature"], c="red", s=.1)
    return ax, ax2


def make_environmental_plot(data, dest, title=""):
    fig = plt.figure(1, figsize=(5,5), dpi=90)
    ax = fig.add_subplot(111)
    ax = geom_ld_annotations(data, ax)
    geom_env_data(data, ax)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(dest)
