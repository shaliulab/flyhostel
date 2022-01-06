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

def set_xticks(ax, xtick_freq, min_t, max_t):
    xticks = np.arange(
        xtick_freq * (int(min_t) // xtick_freq),
        xtick_freq * (int(max_t + xtick_freq) // xtick_freq),
        xtick_freq,
    )
    ax.set_xticks(xticks)
    return ax, xticks


def geom_ld_annotation(data, ax, yrange=(0, 100), xtick_freq=6):
    """
    data must have:

    Column L with either T or F (str)
    Column t with ZT in hours (float)

    """

    data=data.copy() # working with a copy! :D
    data["t"] /= 3600 # to hours


    light_states, positions = zeitgeber.rle.decompose(data["L"].values.tolist())
    transitions = []
    for i in positions:
        transitions.append(round(data.loc[i]["t"], 2))


    max_t = data["t"].tail().values[-1]
    min_t = transitions[0]
    y_min = yrange[0]
    y_max = yrange[1]

    ax, xticks = set_xticks(ax, xtick_freq, min_t, max_t)
    xrange = [min(xticks), max(xticks)]
    print(xrange)

    # background color (default)
    polygon = Polygon(
        [
            (xrange[0], y_min),
            (xrange[1], y_min),
            (xrange[1], y_max),
            (xrange[0], y_max),
        ]
    )
   
    polygon_path = PolygonPatch(
        polygon,
        facecolor=(0, 0, 0),
        edgecolor=(0, 0, 0),
    )

    ax.add_patch(polygon_path)
    ###   

    
    color = {"F": (0.5, 0.5, 0.5), "T": (1, 1, 1)}
    logger.debug(f"Positions set to {positions}")
    logger.debug(f"Transitions set to {transitions}")

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
            edgecolor=(0, 0, 0),
        )
        ax.add_patch(polygon_path)

    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    return ax


def geom_env_data(data, ax):

    ax2 = ax.twinx()
    ax2.set_ylabel("Temp ºC")
    ax.set_ylabel("% Hum")

    # ax.set_xticks(list(range(*xrange)) + [xrange[-1]])
    # ax.set_yticks(list(range(*yrange)) + [yrange[-1]])
    # ax.set_aspect(1)
    ax.scatter(data["t"] / 3600, data["humidity"], s=0.1)
    ax2.scatter(data["t"] / 3600, data["temperature"], c="red", s=0.1)
    return ax, ax2


def make_environmental_plot(root, data, title=""):
    fig = plt.figure(1, figsize=(5, 5), dpi=90)
    ax = fig.add_subplot(111)
    ax = geom_ld_annotations(data, ax, yrange=(0, 100))
    geom_env_data(data, ax)
    ax.set_title(title)
    plt.tight_layout()
    dest = root + "_environment-log.png"
    fig.savefig(dest)
    plt.close(fig)

    light_log_data = data.copy()
    light_log_data["t"] /= 3600

    fig = plt.figure(1, figsize=(5, 5), dpi=90)
    ax = fig.add_subplot(111)
    ax = geom_ld_annotations(data, ax, yrange=(0, light_log_data["light"].max()))
    ax.scatter(light_log_data["t"], light_log_data["light"])
    ax = set_xticks(ax, 6, light_log_data["t"].min(), light_log_data["t"].max())
    dest = root + "_light-log.png"
    fig.savefig(dest)
    plt.close(fig)
