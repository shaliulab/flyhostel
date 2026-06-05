"""
movement/xarray -> FlyTrack adapter.

Converts a ``movement``-style xarray Dataset (with ``pose_tracks`` and
``confidence`` variables and dimensions time, individuals, keypoints, space)
into a JAABA-style :class:`Trx` of :class:`FlyTrack` objects.

The expected dataset layout:

    Dimensions:
        time:        T frames
        individuals: N flies
        keypoints:   K keypoints (must include head, thorax, and one of
                     [abdomen, tail]; optionally [lW, rW])
        space:       2 (x, y)

    Data variables:
        pose_tracks (time, individuals, keypoints, space)  -- pixel coords
        confidence  (time, individuals, keypoints)          -- per-kp scores

    Attributes:
        fps : float
        source_software : 'SLEAP' (or similar)

If your keypoints are named differently (the user has ``'head', 'thorax',
'lW', 'rW'`` plus a few we haven't seen), pass the actual names via the
``keypoint_map`` argument.
"""

from __future__ import annotations
from typing import Dict, Optional, Sequence
import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None  # type: ignore

from .trx import FlyTrack, Trx


# default keypoint name aliases - tries each in order until one is found
_DEFAULT_KEYPOINT_ALIASES = {
    "head": ["head", "nose", "Head", "h"],
    "thorax": ["thorax", "Thorax", "th", "body", "center"],
    "abdomen": ["abdomen", "tail", "Abdomen", "Tail", "abd", "tip", "ab"],
    "wing_l": ["lW", "wingL", "wing_l", "wing_l_tip", "leftWing", "lw"],
    "wing_r": ["rW", "wingR", "wing_r", "wing_r_tip", "rightWing", "rw"],
}


def _resolve_keypoint(available: Sequence[str], aliases: Sequence[str],
                      required: bool, label: str) -> Optional[str]:
    avail_lower = {n.lower(): n for n in available}
    for alias in aliases:
        if alias in available:
            return alias
        if alias.lower() in avail_lower:
            return avail_lower[alias.lower()]
    if required:
        raise KeyError(
            f"No keypoint matching {label} found. Looked for {aliases}; "
            f"available: {list(available)}"
        )
    return None


def _signed_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angle from vector a to vector b, both shape (N, 2). +ccw."""
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    return np.arctan2(cross, dot)


def movement_to_trx(
    ds: "xr.Dataset",
    px_per_mm: float,
    *,
    keypoint_map: Optional[Dict[str, str]] = None,
    confidence_threshold: Optional[float] = None,
    smooth_window: Optional[int] = None,
    arena_radius_mm: Optional[float] = None,
    arena_center_mm: tuple = (0.0, 0.0),
    b_to_a_ratio: float = 0.35,
    flip_wing_sign: bool = False,
) -> Trx:
    """Build a Trx from a movement/xarray pose dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Movement-style dataset with ``pose_tracks`` (time, individuals,
        keypoints, space) and (optional) ``confidence``.
    px_per_mm : float
        Pixel-to-mm conversion factor (e.g., chamber diameter in pixels /
        chamber diameter in mm).
    keypoint_map : dict, optional
        Mapping from canonical name to actual keypoint name in your dataset.
        E.g., ``{"head": "head", "thorax": "thorax", "abdomen": "tail",
                 "wing_l": "lW", "wing_r": "rW"}``. If omitted, the adapter
        will try common aliases.
    confidence_threshold : float, optional
        If given, any frame where any of (head, thorax, abdomen) has confidence
        below this is NaN-masked.
    smooth_window : int, optional
        If given, apply a moving-average smoothing of this window length to
        the x/y coordinates of each keypoint per fly. Useful when SLEAP is
        jittery.
    arena_radius_mm, arena_center_mm : optional arena geometry.
    b_to_a_ratio : float
        Semi-minor axis is set to ``b_to_a_ratio * a_mm`` where a_mm is half
        the head-to-abdomen distance.
    flip_wing_sign : bool, default False
        Set True ONLY if your wing labels (``lW``, ``rW``) follow image-side
        (left/right of the camera frame) rather than anatomical-side (left/
        right of the fly's body, sign-invariant to camera orientation).
        With anatomically-labeled wings (the standard SLEAP / FlyHostel
        convention), the raw signed body-to-wing angle in image coordinates
        is already sign-correct for JAABA (anatomical-left wing -> negative
        angle when spread, anatomical-right wing -> positive angle when
        spread), so leave this False.

    Returns
    -------
    Trx
        A Trx with one FlyTrack per individual that has at least one valid
        frame. Each FlyTrack's frame indices are absolute (relative to the
        first frame in ``ds``).
    """
    if xr is None:
        raise ImportError("xarray is not installed; run `pip install xarray`.")

    # extract metadata
    fps = float(ds.attrs.get("fps", 30.0))
    dt = 1.0 / fps

    available_kp = list(ds.coords["keypoints"].values)
    if keypoint_map is None:
        keypoint_map = {
            "head": _resolve_keypoint(available_kp, _DEFAULT_KEYPOINT_ALIASES["head"], True, "head"),
            "thorax": _resolve_keypoint(available_kp, _DEFAULT_KEYPOINT_ALIASES["thorax"], True, "thorax"),
            "abdomen": _resolve_keypoint(available_kp, _DEFAULT_KEYPOINT_ALIASES["abdomen"], True, "abdomen"),
            "wing_l": _resolve_keypoint(available_kp, _DEFAULT_KEYPOINT_ALIASES["wing_l"], False, "wing_l"),
            "wing_r": _resolve_keypoint(available_kp, _DEFAULT_KEYPOINT_ALIASES["wing_r"], False, "wing_r"),
        }

    head_name = keypoint_map["head"]
    thorax_name = keypoint_map["thorax"]
    abdomen_name = keypoint_map["abdomen"]
    wing_l_name = keypoint_map.get("wing_l")
    wing_r_name = keypoint_map.get("wing_r")

    individuals = list(ds.coords["individuals"].values)
    n_frames = ds.sizes["time"]

    flies = []
    for f_idx, ind_name in enumerate(individuals):
        # pull this individual's keypoint trajectories: shape (T, K, 2)
        kp_arr = ds["pose_tracks"].isel(individuals=f_idx).values
        if "confidence" in ds:
            conf_arr = ds["confidence"].isel(individuals=f_idx).values  # (T, K)
        else:
            conf_arr = None

        # index of each canonical keypoint
        def _idx(name: Optional[str]) -> Optional[int]:
            if name is None:
                return None
            return available_kp.index(name)

        i_head = _idx(head_name)
        i_thorax = _idx(thorax_name)
        i_abd = _idx(abdomen_name)
        i_wl = _idx(wing_l_name)
        i_wr = _idx(wing_r_name)

        head_xy = kp_arr[:, i_head, :].astype(float)
        thorax_xy = kp_arr[:, i_thorax, :].astype(float)
        abd_xy = kp_arr[:, i_abd, :].astype(float)

        # optional confidence-based NaN masking
        if confidence_threshold is not None and conf_arr is not None:
            for kp_idx, xy in [(i_head, head_xy), (i_thorax, thorax_xy),
                               (i_abd, abd_xy)]:
                bad = conf_arr[:, kp_idx] < confidence_threshold
                xy[bad] = np.nan

        # optional smoothing
        if smooth_window is not None and smooth_window > 1:
            head_xy = _smooth_xy(head_xy, smooth_window)
            thorax_xy = _smooth_xy(thorax_xy, smooth_window)
            abd_xy = _smooth_xy(abd_xy, smooth_window)

        # define valid frames as those where thorax is finite
        valid = np.isfinite(thorax_xy[:, 0]) & np.isfinite(thorax_xy[:, 1])
        if not valid.any():
            continue
        first = int(np.argmax(valid))
        last = n_frames - 1 - int(np.argmax(valid[::-1]))
        nframes_fly = last - first + 1
        sl = slice(first, last + 1)

        # body coordinates (pixels -> mm)
        x_mm = thorax_xy[sl, 0] / px_per_mm
        y_mm = thorax_xy[sl, 1] / px_per_mm

        # body orientation: angle of (head - thorax)
        dx = head_xy[sl, 0] - thorax_xy[sl, 0]
        dy = head_xy[sl, 1] - thorax_xy[sl, 1]
        theta = np.arctan2(dy, dx)

        # semi-major axis: half head-to-abdomen
        bl_px = np.hypot(head_xy[sl, 0] - abd_xy[sl, 0],
                         head_xy[sl, 1] - abd_xy[sl, 1])
        a_mm = (bl_px / 2.0) / px_per_mm
        b_mm = b_to_a_ratio * a_mm

        # wings
        wl_ang = None
        wr_ang = None
        if i_wl is not None:
            wl_xy = kp_arr[sl, i_wl, :].astype(float)
            if confidence_threshold is not None and conf_arr is not None:
                bad = conf_arr[sl, i_wl] < confidence_threshold
                wl_xy[bad] = np.nan
            body_dir = np.column_stack([dx, dy])
            wing_dir = np.column_stack([
                wl_xy[:, 0] - thorax_xy[sl, 0],
                wl_xy[:, 1] - thorax_xy[sl, 1],
            ])
            wl_ang = _signed_angle(body_dir, wing_dir)
            if flip_wing_sign:
                wl_ang = -wl_ang
        if i_wr is not None:
            wr_xy = kp_arr[sl, i_wr, :].astype(float)
            if confidence_threshold is not None and conf_arr is not None:
                bad = conf_arr[sl, i_wr] < confidence_threshold
                wr_xy[bad] = np.nan
            body_dir = np.column_stack([dx, dy])
            wing_dir = np.column_stack([
                wr_xy[:, 0] - thorax_xy[sl, 0],
                wr_xy[:, 1] - thorax_xy[sl, 1],
            ])
            wr_ang = _signed_angle(body_dir, wing_dir)
            if flip_wing_sign:
                wr_ang = -wr_ang

        # fly_id: use the index in the dataset (so two-fly chambers get 0 and 1)
        flies.append(FlyTrack(
            fly_id=f_idx,
            firstframe=first,
            nframes=nframes_fly,
            dt=dt,
            x_mm=x_mm,
            y_mm=y_mm,
            a_mm=a_mm,
            b_mm=b_mm,
            theta_mm=theta,
            wing_anglel=wl_ang,
            wing_angler=wr_ang,
        ))

    return Trx(
        flies=flies,
        arena_radius_mm=arena_radius_mm,
        arena_center_mm=arena_center_mm,
    )


def _smooth_xy(xy: np.ndarray, window: int) -> np.ndarray:
    """Moving-average smoothing of an (N, 2) coordinate array, NaN-safe.

    Uses a centered window; edges are partially-padded so output is same length.
    """
    if window < 2:
        return xy
    out = np.empty_like(xy)
    for col in range(xy.shape[1]):
        x = xy[:, col]
        # mask NaNs with zeros, keep weight as 0 at NaN positions
        mask = np.isfinite(x).astype(float)
        x0 = np.where(np.isfinite(x), x, 0.0)
        kernel = np.ones(window) / window
        num = np.convolve(x0, kernel, mode="same")
        den = np.convolve(mask, kernel, mode="same")
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, col] = np.where(den > 0, num / den, np.nan)
    return out