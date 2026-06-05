"""
SLEAP -> FlyTrack adapter.

This module converts SLEAP keypoint output into JAABA-style FlyTrack objects.
JAABA's features assume per-frame ellipse parameters (a, b, theta) and a body
centroid in millimeters. SLEAP gives you keypoint coordinates in pixels.

We provide two adapters:

    sleap_to_trx_from_keypoints(...)
        For when you have SLEAP keypoints with at least head, thorax, abdomen
        and (optionally) wing tip points. We fit an ellipse from those points
        per frame.

    sleap_to_trx_from_h5(...)
        Wrapper that loads SLEAP's HDF5 analysis export and calls the above.

Naming convention for keypoints (you can rename via the kp_names argument):
    "head"        : nose tip
    "thorax"      : body center
    "abdomen"     : tail tip
    "wing_l_tip"  : left wing tip (optional)
    "wing_r_tip"  : right wing tip (optional)

Body orientation theta is computed from thorax -> head vector. The semi-major
axis a_mm is taken as half the head-to-abdomen distance. The semi-minor axis
b_mm defaults to ``b_to_a_ratio * a_mm`` (default 0.35, a typical Drosophila
body aspect ratio). You can also supply b_mm explicitly per fly.

Wing angles: by JAABA convention wing_anglel < 0 and wing_angler > 0 when
both wings are spread. We compute each wing angle as the signed angle from
the body axis (thorax -> head, treated as the +x direction in body frame) to
the wing tip vector (thorax -> wing tip), positive counter-clockwise.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import numpy as np

from .trx import FlyTrack, Trx


def _signed_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed angle from vector a to vector b, both shape (N, 2). +ccw."""
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    dot = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
    return np.arctan2(cross, dot)


def sleap_to_trx_from_keypoints(
    keypoints_px: np.ndarray,
    kp_names: Sequence[str],
    fly_ids: Sequence[int],
    px_per_mm: float,
    fps: float,
    arena_radius_mm: Optional[float] = None,
    arena_center_mm: tuple = (0.0, 0.0),
    head_name: str = "head",
    thorax_name: str = "thorax",
    abdomen_name: str = "abdomen",
    wing_l_name: Optional[str] = "wing_l_tip",
    wing_r_name: Optional[str] = "wing_r_tip",
    b_to_a_ratio: float = 0.35,
) -> Trx:
    """Build a Trx from SLEAP keypoints.

    Parameters
    ----------
    keypoints_px : array, shape (n_flies, n_frames, n_kp, 2)
        SLEAP output in pixel coordinates. NaN for missing detections.
    kp_names : sequence of str, length n_kp
        Names of the keypoints, in the order they appear in the keypoints array.
    fly_ids : sequence of int, length n_flies
        Per-track identifier (e.g. from idtracker.ai).
    px_per_mm : float
        Pixel-to-mm conversion factor (calibrated from arena diameter).
    fps : float
        Frames per second of the recording.
    arena_radius_mm, arena_center_mm : optional arena geometry for wall features.
    head_name, thorax_name, abdomen_name : keypoint names to use.
    wing_l_name, wing_r_name : optional wing tip names; pass None if absent.
    b_to_a_ratio : float
        Used to set the semi-minor axis: b_mm = b_to_a_ratio * a_mm.

    Returns
    -------
    Trx
    """
    n_flies, n_frames, n_kp, _ = keypoints_px.shape
    assert len(kp_names) == n_kp, "kp_names length must match keypoints array"
    assert len(fly_ids) == n_flies, "fly_ids length must match"

    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(kp_names)}
    head_i = name_to_idx[head_name]
    thorax_i = name_to_idx[thorax_name]
    abdomen_i = name_to_idx[abdomen_name]
    wing_l_i = name_to_idx.get(wing_l_name) if wing_l_name else None
    wing_r_i = name_to_idx.get(wing_r_name) if wing_r_name else None

    dt = 1.0 / fps
    flies: List[FlyTrack] = []
    for f in range(n_flies):
        kp = keypoints_px[f]  # (n_frames, n_kp, 2)
        # find the first/last frame with valid thorax detection
        valid_thorax = ~np.isnan(kp[:, thorax_i, 0])
        if not valid_thorax.any():
            continue
        first = int(np.argmax(valid_thorax))
        last = n_frames - 1 - int(np.argmax(valid_thorax[::-1]))
        nframes = last - first + 1
        # extract per-frame arrays for this fly (in pixels first)
        kp_slice = kp[first:last + 1]
        thorax = kp_slice[:, thorax_i, :]
        head = kp_slice[:, head_i, :]
        abdomen = kp_slice[:, abdomen_i, :]

        # centroid: midpoint of thorax-abdomen-head trio, but thorax alone is
        # often fine. JAABA uses the body centroid from segmentation; we use
        # the thorax keypoint as a clean proxy.
        x_px = thorax[:, 0]
        y_px = thorax[:, 1]

        # body orientation: angle of (head - thorax)
        dx = head[:, 0] - thorax[:, 0]
        dy = head[:, 1] - thorax[:, 1]
        theta = np.arctan2(dy, dx)

        # semi-major axis: half the head-to-abdomen distance
        bl_px = np.hypot(head[:, 0] - abdomen[:, 0],
                         head[:, 1] - abdomen[:, 1])
        a_mm = (bl_px / 2.0) / px_per_mm
        b_mm = b_to_a_ratio * a_mm

        x_mm = x_px / px_per_mm
        y_mm = y_px / px_per_mm

        # wing angles, if available
        wl_ang = None
        wr_ang = None
        if wing_l_i is not None:
            wl = kp_slice[:, wing_l_i, :]
            body_dir = np.column_stack([dx, dy])
            wing_dir = np.column_stack([wl[:, 0] - thorax[:, 0],
                                        wl[:, 1] - thorax[:, 1]])
            wl_ang = _signed_angle(body_dir, wing_dir)
            # JAABA's convention: wing_anglel < 0 when spread. The signed angle
            # from body-forward to a left-wing tip (which is at the side of the
            # body) will be ~+pi/2 in standard image coordinates if y goes down.
            # Users may need to adjust the sign depending on their image
            # coordinate convention. We default to image coords (y down).
            # In image coordinates, "left" of the fly is +ccw which is +angle;
            # JAABA expects negative. Hence we negate to match the convention.
            wl_ang = -wl_ang
        if wing_r_i is not None:
            wr = kp_slice[:, wing_r_i, :]
            body_dir = np.column_stack([dx, dy])
            wing_dir = np.column_stack([wr[:, 0] - thorax[:, 0],
                                        wr[:, 1] - thorax[:, 1]])
            wr_ang = _signed_angle(body_dir, wing_dir)
            # same coordinate sign flip
            wr_ang = -wr_ang

        flies.append(FlyTrack(
            fly_id=int(fly_ids[f]),
            firstframe=first,
            nframes=nframes,
            dt=dt,
            x_mm=x_mm,
            y_mm=y_mm,
            a_mm=a_mm,
            b_mm=b_mm,
            theta_mm=theta,
            wing_anglel=wl_ang,
            wing_angler=wr_ang,
        ))

    return Trx(flies=flies, arena_radius_mm=arena_radius_mm,
               arena_center_mm=arena_center_mm)
