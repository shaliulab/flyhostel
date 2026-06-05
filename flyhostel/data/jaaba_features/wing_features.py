"""
Wing features.

JAABA's wing features assume two per-frame quantities:
    wing_anglel  : angle of the LEFT wing relative to the body axis (typically negative)
    wing_angler  : angle of the RIGHT wing relative to the body axis (typically positive)
and optionally per-wing lengths and areas. By JAABA's sign convention, a
symmetric extension yields wing_anglel < 0 and wing_angler > 0. The "max" wing
angle uses the negated left value so that both wings can be compared as
non-negative quantities.

Inputs are FlyTrack objects with wing_anglel / wing_angler / etc. populated.
Each function returns (data, units). data has length fly.nframes (or one less
for derivatives).
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

from .trx import FlyTrack


def _require(arr: Optional[np.ndarray], name: str, fly: FlyTrack) -> np.ndarray:
    if arr is None:
        raise ValueError(
            f"FlyTrack id={fly.fly_id} does not have {name}; "
            "wing features require wing_anglel/wing_angler (and lengths/areas)."
        )
    return arr


# -------- per-wing angles ----------------------------------------------------


def max_wing_angle(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Larger of the two wing angles (signed: left is negated to be positive)."""
    l = _require(fly.wing_anglel, "wing_anglel", fly)
    r = _require(fly.wing_angler, "wing_angler", fly)
    return np.maximum(-l, r), "rad"


def min_wing_angle(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Smaller of the two wing angles (using the negated left convention)."""
    l = _require(fly.wing_anglel, "wing_anglel", fly)
    r = _require(fly.wing_angler, "wing_angler", fly)
    return np.minimum(-l, r), "rad"


def mean_wing_angle(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Average wing angle (using the negated left convention).

    compute_mean_wing_angle.m: (-wing_anglel + wing_angler) / 2
    """
    l = _require(fly.wing_anglel, "wing_anglel", fly)
    r = _require(fly.wing_angler, "wing_angler", fly)
    return 0.5 * (-l + r), "rad"


def wing_angle_diff(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Signed difference: wing_angler - wing_anglel (positive if right > left)."""
    l = _require(fly.wing_anglel, "wing_anglel", fly)
    r = _require(fly.wing_angler, "wing_angler", fly)
    return r - l, "rad"


def wing_angle_imbalance(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """|wing_angler + wing_anglel|. Symmetric extension gives zero."""
    l = _require(fly.wing_anglel, "wing_anglel", fly)
    r = _require(fly.wing_angler, "wing_angler", fly)
    return np.abs(r + l), "rad"


# -------- per-wing lengths and areas ----------------------------------------


def max_wing_length(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    l = _require(fly.wing_lengthl_mm, "wing_lengthl_mm", fly)
    r = _require(fly.wing_lengthr_mm, "wing_lengthr_mm", fly)
    return np.maximum(l, r), "mm"


def min_wing_length(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    l = _require(fly.wing_lengthl_mm, "wing_lengthl_mm", fly)
    r = _require(fly.wing_lengthr_mm, "wing_lengthr_mm", fly)
    return np.minimum(l, r), "mm"


def mean_wing_length(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    l = _require(fly.wing_lengthl_mm, "wing_lengthl_mm", fly)
    r = _require(fly.wing_lengthr_mm, "wing_lengthr_mm", fly)
    return 0.5 * (l + r), "mm"


def max_wing_area(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    l = _require(fly.wing_areal_mm, "wing_areal_mm", fly)
    r = _require(fly.wing_arear_mm, "wing_arear_mm", fly)
    return np.maximum(l, r), "mm^2"


def min_wing_area(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    l = _require(fly.wing_areal_mm, "wing_areal_mm", fly)
    r = _require(fly.wing_arear_mm, "wing_arear_mm", fly)
    return np.minimum(l, r), "mm^2"


def mean_wing_area(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    l = _require(fly.wing_areal_mm, "wing_areal_mm", fly)
    r = _require(fly.wing_arear_mm, "wing_arear_mm", fly)
    return 0.5 * (l + r), "mm^2"


# -------- "inmost" vs "outmost" (smaller / larger) --------------------------


def length_inmost_wing(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Length of the smaller wing."""
    return min_wing_length(fly)


def length_outmost_wing(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Length of the larger wing."""
    return max_wing_length(fly)


def area_inmost_wing(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    return min_wing_area(fly)


def area_outmost_wing(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    return max_wing_area(fly)


def angle_biggest_wing(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Wing angle of whichever wing is currently larger by area."""
    l_area = _require(fly.wing_areal_mm, "wing_areal_mm", fly)
    r_area = _require(fly.wing_arear_mm, "wing_arear_mm", fly)
    l_ang = _require(fly.wing_anglel, "wing_anglel", fly)
    r_ang = _require(fly.wing_angler, "wing_angler", fly)
    pick_right = r_area >= l_area
    return np.where(pick_right, r_ang, -l_ang), "rad"


def angle_smallest_wing(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Wing angle of whichever wing is currently smaller by area."""
    l_area = _require(fly.wing_areal_mm, "wing_areal_mm", fly)
    r_area = _require(fly.wing_arear_mm, "wing_arear_mm", fly)
    l_ang = _require(fly.wing_anglel, "wing_anglel", fly)
    r_ang = _require(fly.wing_angler, "wing_angler", fly)
    pick_right = r_area < l_area
    return np.where(pick_right, r_ang, -l_ang), "rad"


# -------- derivatives ---------------------------------------------------------


def dwing_angle_diff(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    d, _ = wing_angle_diff(fly)
    if len(d) < 2:
        return np.zeros(0), "rad/s"
    return np.diff(d) / fly.dt, "rad/s"


def dmean_wing_angle(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    m, _ = mean_wing_angle(fly)
    if len(m) < 2:
        return np.zeros(0), "rad/s"
    return np.diff(m) / fly.dt, "rad/s"


def dmax_wing_angle(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    m, _ = max_wing_angle(fly)
    if len(m) < 2:
        return np.zeros(0), "rad/s"
    return np.diff(m) / fly.dt, "rad/s"


def dmin_wing_angle(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    m, _ = min_wing_angle(fly)
    if len(m) < 2:
        return np.zeros(0), "rad/s"
    return np.diff(m) / fly.dt, "rad/s"


def dmax_wing_length(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    m, _ = max_wing_length(fly)
    if len(m) < 2:
        return np.zeros(0), "mm/s"
    return np.diff(m) / fly.dt, "mm/s"


def dmin_wing_length(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    m, _ = min_wing_length(fly)
    if len(m) < 2:
        return np.zeros(0), "mm/s"
    return np.diff(m) / fly.dt, "mm/s"


def dmean_wing_area(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    m, _ = mean_wing_area(fly)
    if len(m) < 2:
        return np.zeros(0), "mm^2/s"
    return np.diff(m) / fly.dt, "mm^2/s"
