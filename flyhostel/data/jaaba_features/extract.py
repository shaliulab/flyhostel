"""
Batch feature extraction.

Convenience layer that computes a configurable set of JAABA per-frame features
for every fly in a Trx, and assembles them into a pandas DataFrame (one row
per fly per frame) ready to feed to a classifier (JAABA-style boosting,
scikit-learn, XGBoost, etc.).

For 1.69M frames x 2 flies (your FlyHostel dataset) the resulting DataFrame
will be ~3.4M rows; with ~25 features that's about 100M floats, ~800MB. If
that's too much, stream by chunking on the time axis.
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd

from .trx import FlyTrack, Trx
from . import single_fly as sf
from . import wing_features as wf
from . import pair_features as pf
from . import closest as cf


# Map feature names to (kind, callable). 'kind' specifies the calling signature:
#   "single"     - f(fly) -> (data, units)
#   "single_arena" - f(fly, arena_center) -> (data, units)
#   "wing"       - f(fly) -> (data, units), requires wing_anglel/r
#   "pair_closest" - takes (trx, fly_id, metric) -> series in fly1's frame
SINGLE_FEATURES = {
    "ecc": sf.ecc,
    "area_mm": sf.area_mm,
    "phi": sf.phi,
    "velmag_ctr": sf.velmag_ctr,
    "velmag_nose": sf.velmag_nose,
    "velmag_tail": sf.velmag_tail,
    "accmag": sf.accmag,
    "dtheta": sf.dtheta,
    "d2theta": sf.d2theta,
    "da": sf.da,
    "db": sf.db,
    "darea": sf.darea,
    "decc": sf.decc,
    "du_ctr": sf.du_ctr,
    "dv_ctr": sf.dv_ctr,
    "yaw": sf.yaw,
}

WING_FEATURES = {
    "max_wing_angle": wf.max_wing_angle,
    "min_wing_angle": wf.min_wing_angle,
    "mean_wing_angle": wf.mean_wing_angle,
    "wing_angle_diff": wf.wing_angle_diff,
    "wing_angle_imbalance": wf.wing_angle_imbalance,
}

# Single-fly-with-arena features
ARENA_FEATURES = {
    "arena_r": sf.arena_r,
    "dist2wall": sf.dist2wall,
    "angle2wall": sf.angle2wall,
    "dangle2wall": sf.dangle2wall,
}


def _align_to_nframes(data: np.ndarray, nframes: int) -> np.ndarray:
    """Pad a feature array to nframes by appending NaN.

    JAABA stores per-frame features as length nframes. Derivative features are
    length nframes-1 (or -2); we right-pad with NaN.
    """
    if len(data) == nframes:
        return data
    if len(data) < nframes:
        pad = np.full(nframes - len(data), np.nan)
        return np.concatenate([data, pad])
    return data[:nframes]


def extract_features(
    trx: Trx,
    feature_names: Optional[Iterable[str]] = None,
    include_pair: bool = True,
    include_closest: bool = True,
    include_wing: bool = True,
    include_arena: bool = False,
    closest_metric: str = "center",
    other_fly_id: Optional[int] = None,
) -> pd.DataFrame:
    """Compute features for every (fly, frame) and return a long-format DataFrame.

    Columns: ``fly_id``, ``frame``, plus one column per feature.

    Parameters
    ----------
    trx : Trx
    feature_names : iterable of str, optional
        If given, only these features are computed. Names are the keys in
        SINGLE_FEATURES / WING_FEATURES / ARENA_FEATURES, plus
        ``"dcenter"``, ``"dnose2tail"``, ``"dnose2center"``,
        ``"dnose2ell"``, ``"dell2ell"``, ``"anglesub"``,
        ``"anglefrom1to2"``, ``"magveldiff"`` for pairwise,
        and ``"closest_fly_dist"``, ``"angle_on_closest"``,
        ``"anglesub_on_closest"`` for closest-fly features.
    include_pair, include_closest, include_wing, include_arena : bool
        Toggles for whole categories.
    closest_metric : str
        Distance metric for closest-fly features. One of "center",
        "nose2ell", "ell2nose", "ell2ell", "nose2tail", "anglesub".
    other_fly_id : int, optional
        If set, pair features are computed against this specific fly.
        Useful in your 1M:1F (or 2M:2F restricted to known pairs) setup where
        the "other" fly is a known target rather than the closest one. If
        None, pair features against each pair (i, j) are appended as columns
        named e.g. "dcenter__vs1" for fly j=1.
    """
    metric_map = {
        "center": pf.dcenter_pair,
        "nose2ell": pf.dnose2ell_pair,
        "ell2nose": pf.dell2nose_pair,
        "ell2ell": pf.dell2ell_pair,
        "nose2tail": pf.dnose2tail_pair,
        "anglesub": pf.anglesub_pair,
    }
    if closest_metric not in metric_map:
        raise ValueError(f"closest_metric must be one of {list(metric_map)}")

    frames_per_fly: Dict[int, np.ndarray] = {}
    columns: Dict[int, Dict[str, np.ndarray]] = {f.fly_id: {} for f in trx.flies}

    for fly in trx.flies:
        # absolute frame indices for this fly
        frames = np.arange(fly.firstframe, fly.endframe + 1)
        frames_per_fly[fly.fly_id] = frames
        nf = fly.nframes

        # single-fly features
        for name, func in SINGLE_FEATURES.items():
            if feature_names is not None and name not in feature_names:
                continue
            data, _ = func(fly)
            columns[fly.fly_id][name] = _align_to_nframes(data, nf)

        # wing
        if include_wing and fly.wing_anglel is not None and fly.wing_angler is not None:
            for name, func in WING_FEATURES.items():
                if feature_names is not None and name not in feature_names:
                    continue
                data, _ = func(fly)
                columns[fly.fly_id][name] = _align_to_nframes(data, nf)

        # arena
        if include_arena and trx.arena_radius_mm is not None:
            for name, func in ARENA_FEATURES.items():
                if feature_names is not None and name not in feature_names:
                    continue
                if name == "dist2wall":
                    data, _ = func(fly, trx.arena_radius_mm, trx.arena_center_mm)
                elif name in ("arena_r", "angle2wall", "dangle2wall"):
                    data, _ = func(fly, trx.arena_center_mm)
                else:
                    data, _ = func(fly)
                columns[fly.fly_id][name] = _align_to_nframes(data, nf)

        # pairwise / closest-fly
        if include_closest:
            closest_ids, mind = cf.closest_fly(trx, fly.fly_id, metric_map[closest_metric])
            columns[fly.fly_id]["closest_fly_dist"] = mind
            columns[fly.fly_id]["closest_fly_id"] = closest_ids.astype(float)
            ang = cf.angle_on_closest_fly(trx, fly.fly_id, metric_map[closest_metric])
            columns[fly.fly_id]["angle_on_closest"] = ang
            asub = cf.anglesub_on_closest_fly(trx, fly.fly_id, metric_map[closest_metric])
            columns[fly.fly_id]["anglesub_on_closest"] = asub

        if include_pair:
            if other_fly_id is not None:
                if other_fly_id in trx.fly_ids and other_fly_id != fly.fly_id:
                    other = trx[other_fly_id]
                    columns[fly.fly_id]["dcenter"] = pf.dcenter_pair(fly, other)
                    columns[fly.fly_id]["dnose2tail"] = pf.dnose2tail_pair(fly, other)
                    columns[fly.fly_id]["dnose2center"] = pf.dnose2center_pair(fly, other)
                    columns[fly.fly_id]["anglefrom1to2"] = pf.anglefrom1to2_pair(fly, other)
                    columns[fly.fly_id]["anglesub"] = pf.anglesub_pair(fly, other, fov=trx.fov)
                    columns[fly.fly_id]["magveldiff"] = pf.magveldiff_pair(fly, other)
            else:
                # all-pairs: one set of pair features per other fly
                for other in trx.flies:
                    if other.fly_id == fly.fly_id:
                        continue
                    suffix = f"__vs{other.fly_id}"
                    columns[fly.fly_id][f"dcenter{suffix}"] = pf.dcenter_pair(fly, other)
                    columns[fly.fly_id][f"anglefrom1to2{suffix}"] = pf.anglefrom1to2_pair(fly, other)
                    columns[fly.fly_id][f"anglesub{suffix}"] = pf.anglesub_pair(fly, other, fov=trx.fov)

    # assemble as a single long DataFrame
    dfs = []
    for fly in trx.flies:
        df = pd.DataFrame(columns[fly.fly_id])
        df.insert(0, "frame", frames_per_fly[fly.fly_id])
        df.insert(0, "fly_id", fly.fly_id)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
