import logging
import math
import warnings

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

import zeitgeber

try:
    matplotlib.use('TkAgg')
except Exception:
    warnings.warn("matplotlib cannot use TkAgg backend")

logger = logging.getLogger(__name__)


def geom_ld_annotations(*args, **kwargs):
    logger.warning("geom_ld_annotations is deprecated. Please use geom_ld_annotation")

    return geom_ld_annotation(*args, **kwargs)

def set_xticks(ax, xtick_freq, min_t, max_t):
    xticks = np.arange(
        xtick_freq * (int(min_t) // xtick_freq),
        xtick_freq * (math.ceil(max_t + xtick_freq) / xtick_freq),
        xtick_freq,
    )
    ax.set_xticks(xticks)
    return ax, xticks


def geom_ld_annotation(data, ax, yrange=(0, 100), xtick_freq=6):
    """
    Render a background shade in a plot with time along the X axis
    to represent the D/L phases

    Arguments:
        * data (pd.DataFrame). must have:
          * Column L with either T or F (str)
          * Column t with ZT in seconds (float)

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

    # background color (default)
    polygon = plt.Polygon(
        [
            (xrange[0], y_min),
            (xrange[1], y_min),
            (xrange[1], y_max),
            (xrange[0], y_max),
        ], color=(0.86,0.86,0.86)
    )

    ax.add_patch(polygon)

    palette = {"F": (0.5, 0.5, 0.5), "T": (1, 1, 1)}
    logger.debug(f"Positions set to {positions}")
    logger.debug(f"Transitions set to {transitions}")

    for i in range(len(transitions)):
        if (i + 1) == len(transitions):
            polygon = plt.Polygon(
                [
                    (transitions[i], y_min),
                    (max_t, y_min),
                    (max_t, y_max),
                    (transitions[i], y_max),
                ], color=palette[light_states[i]]
            )
        else:
            polygon = plt.Polygon(
                [
                    (transitions[i], y_min),
                    (transitions[i + 1], y_min),
                    (transitions[i + 1], y_max),
                    (transitions[i], y_max),
                ], color=palette[light_states[i]]
            )

        ax.add_patch(polygon)

    ax.set_xlim(*xrange)
    ax.set_ylim(*yrange)
    return ax

