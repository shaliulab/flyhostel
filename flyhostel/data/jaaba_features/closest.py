"""
"Closest fly" features.

JAABA's compute_closestfly_* functions compute, at each frame, which other fly
in the same ROI is closest under some distance metric (centroid, nose-to-ell,
ell-to-ell, etc.). They cache both the closest-fly id and the distance to it.

This module exposes:
    closestfly(trx, fly1_id, metric)  -> (closest_ids, min_distance)
    nflies_close(trx, fly1_id, threshold_mm) -> count_per_frame
    closest_fly_angles(...)  -> angle-on-closest-fly features

You pick the distance metric by passing one of the pair functions from
pair_features.py.
"""

from __future__ import annotations
from typing import Callable, Tuple, List
import numpy as np

from .trx import Trx, FlyTrack
from .pair_features import (
    dcenter_pair,
    dnose2tail_pair,
    dnose2center_pair,
    dnose2ell_pair,
    dell2nose_pair,
    dell2ell_pair,
    anglefrom1to2_pair,
    anglesub_pair,
)


PairFunc = Callable[[FlyTrack, FlyTrack], np.ndarray]


def closest_fly(trx: Trx, fly1_id: int, metric: PairFunc = dcenter_pair
                ) -> Tuple[np.ndarray, np.ndarray]:
    """For each frame in fly1's track, find the closest other fly under ``metric``.

    Only flies in the same ROI as fly1 are considered. Returns:
        closest_ids : int array of length fly1.nframes; -1 where no fly exists
                      or where the distance is NaN
        mindist     : float array of length fly1.nframes; NaN where no other fly
    """
    fly1 = trx[fly1_id]
    other_ids = [fid for fid in trx.flies_in_roi(fly1.roi) if fid != fly1_id]
    if not other_ids:
        return (np.full(fly1.nframes, -1, dtype=int),
                np.full(fly1.nframes, np.nan))

    # stack distances: shape (n_others, nframes)
    dist_stack = np.full((len(other_ids), fly1.nframes), np.inf)
    for k, fid in enumerate(other_ids):
        d = metric(fly1, trx[fid])
        # turn NaN into inf so it can't win the min
        d = np.where(np.isnan(d), np.inf, d)
        dist_stack[k] = d

    idx = np.argmin(dist_stack, axis=0)
    mindist = dist_stack[idx, np.arange(fly1.nframes)]
    # mark frames where no valid distance was found
    closest_ids = np.array([other_ids[i] for i in idx], dtype=int)
    no_other = ~np.isfinite(mindist)
    closest_ids[no_other] = -1
    mindist = np.where(no_other, np.nan, mindist)
    return closest_ids, mindist


def nflies_close(trx: Trx, fly1_id: int, threshold_mm: float,
                 metric: PairFunc = dcenter_pair) -> np.ndarray:
    """Count of other flies within ``threshold_mm`` of fly1 at each frame.

    Length: fly1.nframes.
    """
    fly1 = trx[fly1_id]
    other_ids = [fid for fid in trx.flies_in_roi(fly1.roi) if fid != fly1_id]
    if not other_ids:
        return np.zeros(fly1.nframes, dtype=int)
    count = np.zeros(fly1.nframes, dtype=int)
    for fid in other_ids:
        d = metric(fly1, trx[fid])
        count += (d < threshold_mm).astype(int)  # NaN comparisons -> False
    return count


def angle_on_closest_fly(trx: Trx, fly1_id: int, metric: PairFunc = dcenter_pair
                         ) -> np.ndarray:
    """Angle (in fly1's body frame) to the closest fly under ``metric``.

    Returns length fly1.nframes; NaN at frames with no closest fly.
    """
    closest_ids, _ = closest_fly(trx, fly1_id, metric)
    fly1 = trx[fly1_id]
    out = np.full(fly1.nframes, np.nan)
    # process by groups of frames sharing the same closest fly to vectorize
    unique = set(closest_ids.tolist())
    unique.discard(-1)
    for fid in unique:
        mask = closest_ids == fid
        ang = anglefrom1to2_pair(fly1, trx[fid])
        out[mask] = ang[mask]
    return out


def anglesub_on_closest_fly(trx: Trx, fly1_id: int,
                            metric: PairFunc = dcenter_pair) -> np.ndarray:
    """Angle subtended by the closest fly in fly1's visual field."""
    closest_ids, _ = closest_fly(trx, fly1_id, metric)
    fly1 = trx[fly1_id]
    out = np.full(fly1.nframes, np.nan)
    unique = set(closest_ids.tolist())
    unique.discard(-1)
    for fid in unique:
        mask = closest_ids == fid
        a = anglesub_pair(fly1, trx[fid], fov=trx.fov)
        out[mask] = a[mask]
    return out


# Convenience aliases so users can write a single line:


def closestfly_center(trx: Trx, fly1_id: int):
    return closest_fly(trx, fly1_id, dcenter_pair)


def closestfly_nose2ell(trx: Trx, fly1_id: int):
    return closest_fly(trx, fly1_id, dnose2ell_pair)


def closestfly_ell2nose(trx: Trx, fly1_id: int):
    return closest_fly(trx, fly1_id, dell2nose_pair)


def closestfly_ell2ell(trx: Trx, fly1_id: int):
    return closest_fly(trx, fly1_id, dell2ell_pair)


def closestfly_nose2tail(trx: Trx, fly1_id: int):
    return closest_fly(trx, fly1_id, dnose2tail_pair)


def closestfly_anglesub(trx: Trx, fly1_id: int):
    return closest_fly(trx, fly1_id, anglesub_pair)
