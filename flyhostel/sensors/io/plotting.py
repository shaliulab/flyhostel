import logging

import numpy as np
from shapely.geometry import Polygon
from descartes import PolygonPatch

import zeitgeber
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)


def geom_ld_annotations(*args, **kwargs):
    logger.warning("geom_ld_annotations is deprecated. Please use geom_ld_annotation")

    return geom_ld_annotation(*args, **kwargs)


def geom_ld_annotation(data, ax, yrange=(0, 100), xtick_freq=6):
    """
    data must have:

    Column L with either T or F (str)
    Column t with ZT in hours (float)

    """

    light_states, pos = zeitgeber.rle.decompose(data["L"].values.tolist())
    transitions = []
    for i in pos:
        transitions.append(round(data.loc[i]["t"] / 3600, 2))

    max_t = data["t"].tail().values[-1] / 3600
    min_t = transitions[0]
    
    y_min = yrange[0]
    y_max = yrange[1]
    
    color = {"F": (0, 0, 0), "T": (1, 1, 1)}

    for i in range(len(transitions)):
        if (i + 1) == len(transitions):
            polygon = Polygon(
                [
                    (transitions[i], y_min),
                    (max_t, y_min),
                    (max_t, y_max),
                    (transitions[i], y_max),
                ]
            )
        else:
            polygon = Polygon(
                [
                    (transitions[i], y_min),
                    (transitions[i + 1], y_min),
                    (transitions[i + 1], y_max),
                    (transitions[i], y_max),
                ]
            )
        polygon_path = PolygonPatch(
            polygon,
            facecolor=color[light_states[i]],
            alpha=0.2,
            edgecolor=(0, 0, 0),
        )
        ax.add_patch(polygon_path)

    xrange = [int(np.floor(min_t)), int(np.ceil(max_t))]
    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    xticks = np.arange(
        xtick_freq * (int(min_t) // xtick_freq),
        xtick_freq * (int(max_t + xtick_freq) // xtick_freq),
        xtick_freq,
    )
    ax.set_xticks(xticks)


    return ax


def geom_env_data(data, ax):

    ax2 = ax.twinx()
    ax2.set_ylabel("Temp ºC")
    ax.set_ylabel("% Hum")

    # ax.set_xticks(list(range(*xrange)) + [xrange[-1]])
    # ax.set_yticks(list(range(*yrange)) + [yrange[-1]])
    # ax.set_aspect(1)
    ax.scatter(data["t"], data["humidity"], s=0.1)
    ax2.scatter(data["t"], data["temperature"], c="red", s=0.1)
    return ax, ax2


def make_environmental_plot(data, dest, title=""):
    fig = plt.figure(1, figsize=(5, 5), dpi=90)
    ax = fig.add_subplot(111)
    ax = geom_ld_annotations(data, ax)
    geom_env_data(data, ax)
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(dest)
