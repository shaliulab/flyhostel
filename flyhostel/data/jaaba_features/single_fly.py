"""
Single-fly per-frame features.

Each function takes a single FlyTrack and returns a 1D numpy array of length
nframes (or nframes-1 for finite-difference features) plus a units string.

This file ports the single-fly compute_*.m files. Pair / closest-fly features
live in ``pair_features.py``. Wing-only features live in ``wing_features.py``.

We don't reproduce the JAABA practice of iterating over flies inside each
feature function and returning a cell array. The caller can vectorize over
flies; that's cleaner Python.
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.signal import convolve

from .trx import FlyTrack, modrange, central_diff, angular_diff


# Each feature returns (data, units). Units strings follow JAABA's parseunits().

# -------- basic ellipse / appearance ----------------------------------------


def ecc(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Eccentricity = b/a (JAABA uses this ratio rather than the geometric eccentricity)."""
    # compute_ecc.m: data{i} = trx(fly).b_mm ./ trx(fly).a_mm
    return fly.b_mm / fly.a_mm, "unit"


def area(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Body ellipse area in pixels^2 (proxy: 4*a*b since JAABA uses full axes 2a, 2b)."""
    # JAABA compute_area.m returns the area field of the trx (which FlyTracker
    # populates from segmentation). Here we approximate with the ellipse area.
    return 4.0 * fly.a_mm * fly.b_mm, "px^2"


def area_mm(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Body ellipse area in mm^2."""
    return fly.area_mm, "mm^2"


# -------- positions and orientation -----------------------------------------


def xnose_mm(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """X coordinate of the nose (tip of major axis in front)."""
    # compute_xnose_mm.m:  x_mm + 2*a_mm*cos(theta_mm)
    # Recall that in JAABA, the a_mm field is the SEMI-major axis but many
    # downstream functions pass 2*a_mm because they want the full axis. The
    # nose is at +a from the centroid along the orientation direction.
    return fly.x_mm + 2.0 * fly.a_mm * np.cos(fly.theta_mm), "mm"


def ynose_mm(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Y coordinate of the nose."""
    return fly.y_mm + 2.0 * fly.a_mm * np.sin(fly.theta_mm), "mm"


def phi(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Direction of motion (angle of the velocity vector), in radians.

    Matches compute_phi.m: when the fly doesn't move (dx=dy=0), falls back
    to the body orientation theta_mm.
    """
    if fly.nframes < 2:
        return fly.theta_mm.copy(), "rad"
    dx = central_diff(fly.x_mm, dt=1.0)  # length nframes
    dy = central_diff(fly.y_mm, dt=1.0)
    out = np.arctan2(dy, dx)
    bad = (dx == 0.0) & (dy == 0.0)
    out[bad] = fly.theta_mm[bad]
    return out, "rad"


def smooththeta(fly: FlyTrack, thetafil: np.ndarray = None) -> Tuple[np.ndarray, str]:
    """Smoothed body orientation, used by smoothed angular-velocity features.

    JAABA convolves theta_mm with a filter (trx.perframe_params.thetafil). To
    avoid issues with angle wrap-around, we smooth (cos(theta), sin(theta))
    independently and recover the angle with atan2.
    """
    if thetafil is None:
        f = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
        thetafil = f / f.sum()
    c = convolve(np.cos(fly.theta_mm), thetafil, mode="same")
    s = convolve(np.sin(fly.theta_mm), thetafil, mode="same")
    return np.arctan2(s, c), "rad"


# -------- velocity and acceleration ------------------------------------------


def velmag_ctr(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Speed of the body centroid in mm/s.

    Length: nframes - 1 (matching MATLAB diff semantics).
    """
    if fly.nframes < 2:
        return np.zeros(0), "mm/s"
    dx = np.diff(fly.x_mm)
    dy = np.diff(fly.y_mm)
    return np.sqrt(dx * dx + dy * dy) / fly.dt, "mm/s"


def velmag_nose(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Speed of the nose point in mm/s. Length: nframes - 1."""
    if fly.nframes < 2:
        return np.zeros(0), "mm/s"
    xn, _ = xnose_mm(fly)
    yn, _ = ynose_mm(fly)
    dx = np.diff(xn)
    dy = np.diff(yn)
    return np.sqrt(dx * dx + dy * dy) / fly.dt, "mm/s"


def velmag_tail(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Speed of the tail point in mm/s. Length: nframes - 1.

    Tail is at -a from the centroid along the orientation direction.
    """
    if fly.nframes < 2:
        return np.zeros(0), "mm/s"
    xt = fly.x_mm - 2.0 * fly.a_mm * np.cos(fly.theta_mm)
    yt = fly.y_mm - 2.0 * fly.a_mm * np.sin(fly.theta_mm)
    dx = np.diff(xt)
    dy = np.diff(yt)
    return np.sqrt(dx * dx + dy * dy) / fly.dt, "mm/s"


def velmag(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Magnitude of velocity of the center of rotation, fallback to centroid speed.

    JAABA computes the center of rotation explicitly from corfrac_maj/corfrac_min.
    For simplicity we fall back to centroid speed; if you need the COR version,
    extend this following compute_velmag.m + center_of_rotation.m.
    """
    return velmag_ctr(fly)


def accmag(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Acceleration magnitude in mm/s^2. Length: nframes - 2."""
    if fly.nframes < 3:
        return np.zeros(0), "mm/s^2"
    dx = np.diff(fly.x_mm) / fly.dt
    dy = np.diff(fly.y_mm) / fly.dt
    ax = np.diff(dx) / fly.dt
    ay = np.diff(dy) / fly.dt
    return np.sqrt(ax * ax + ay * ay), "mm/s^2"


def veltoward(fly1: FlyTrack, fly2: FlyTrack) -> Tuple[np.ndarray, str]:
    """Velocity of fly1 in the direction of fly2.

    Defined over the overlap of the two flies; outside the overlap returns NaN.
    Length: fly1.nframes (NaN-padded outside overlap, and at the final frame).
    """
    out = np.full(fly1.nframes, np.nan)
    rng = fly1.overlap_range(fly2)
    if rng is None:
        return out, "mm/s"
    o1, o2 = fly1.off, fly2.off
    for t in rng:
        i, j = t + o1, t + o2
        if i + 1 >= fly1.nframes or j + 1 >= fly2.nframes:
            continue
        vx = (fly1.x_mm[i + 1] - fly1.x_mm[i]) / fly1.dt
        vy = (fly1.y_mm[i + 1] - fly1.y_mm[i]) / fly1.dt
        dx = fly2.x_mm[j] - fly1.x_mm[i]
        dy = fly2.y_mm[j] - fly1.y_mm[i]
        d = np.hypot(dx, dy)
        if d > 0:
            out[i] = (vx * dx + vy * dy) / d
    return out, "mm/s"


# -------- angular velocity / acceleration -----------------------------------


def dtheta(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Angular velocity (rad/s). Length: nframes - 1.

    Matches compute_dtheta.m: modrange(diff(theta_mm), -pi, pi) / dt.
    """
    return angular_diff(fly.theta_mm, dt=fly.dt), "rad/s"


def d2theta(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Angular acceleration (rad/s^2). Length: nframes - 2."""
    if fly.nframes < 3:
        return np.zeros(0), "rad/s^2"
    return np.diff(dtheta(fly)[0]) / fly.dt, "rad/s^2"


def smoothdtheta(fly: FlyTrack, thetafil: np.ndarray = None) -> Tuple[np.ndarray, str]:
    """Angular velocity of the smoothed theta."""
    s, _ = smooththeta(fly, thetafil)
    return angular_diff(s, dt=fly.dt), "rad/s"


def smoothd2theta(fly: FlyTrack, thetafil: np.ndarray = None) -> Tuple[np.ndarray, str]:
    """Angular acceleration of the smoothed theta."""
    sd, _ = smoothdtheta(fly, thetafil)
    if len(sd) < 2:
        return np.zeros(0), "rad/s^2"
    return np.diff(sd) / fly.dt, "rad/s^2"


def dtheta_tail(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Angular velocity of the tail point about the centroid.

    Equivalent to dtheta but expressed in the tail's frame.
    """
    return dtheta(fly)


def signdtheta(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Sign of the angular velocity (+1, -1, or 0)."""
    d, _ = dtheta(fly)
    return np.sign(d), "unit"


# -------- shape change rate -------------------------------------------------


def da(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Rate of change of semi-major axis. Length: nframes - 1."""
    if fly.nframes < 2:
        return np.zeros(0), "mm/s"
    return np.diff(fly.a_mm) / fly.dt, "mm/s"


def db(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Rate of change of semi-minor axis. Length: nframes - 1."""
    if fly.nframes < 2:
        return np.zeros(0), "mm/s"
    return np.diff(fly.b_mm) / fly.dt, "mm/s"


def darea(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Rate of change of body area. Length: nframes - 1."""
    if fly.nframes < 2:
        return np.zeros(0), "mm^2/s"
    return np.diff(fly.area_mm) / fly.dt, "mm^2/s"


def decc(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Rate of change of eccentricity. Length: nframes - 1."""
    if fly.nframes < 2:
        return np.zeros(0), "1/s"
    e, _ = ecc(fly)
    return np.diff(e) / fly.dt, "1/s"


# -------- velocity components in body frame (sideways / forward) ------------


def du_ctr(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Forward (along theta) velocity component of the centroid, mm/s.

    Length: nframes - 1.
    """
    if fly.nframes < 2:
        return np.zeros(0), "mm/s"
    dx = np.diff(fly.x_mm) / fly.dt
    dy = np.diff(fly.y_mm) / fly.dt
    # use orientation at the start of each pair of frames
    th = fly.theta_mm[:-1]
    return dx * np.cos(th) + dy * np.sin(th), "mm/s"


def dv_ctr(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Sideways (perpendicular to theta) velocity component of the centroid."""
    if fly.nframes < 2:
        return np.zeros(0), "mm/s"
    dx = np.diff(fly.x_mm) / fly.dt
    dy = np.diff(fly.y_mm) / fly.dt
    th = fly.theta_mm[:-1]
    return -dx * np.sin(th) + dy * np.cos(th), "mm/s"


def du_tail(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Forward velocity component of the tail point."""
    if fly.nframes < 2:
        return np.zeros(0), "mm/s"
    xt = fly.x_mm - 2.0 * fly.a_mm * np.cos(fly.theta_mm)
    yt = fly.y_mm - 2.0 * fly.a_mm * np.sin(fly.theta_mm)
    dx = np.diff(xt) / fly.dt
    dy = np.diff(yt) / fly.dt
    th = fly.theta_mm[:-1]
    return dx * np.cos(th) + dy * np.sin(th), "mm/s"


def dv_tail(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Sideways velocity component of the tail point."""
    if fly.nframes < 2:
        return np.zeros(0), "mm/s"
    xt = fly.x_mm - 2.0 * fly.a_mm * np.cos(fly.theta_mm)
    yt = fly.y_mm - 2.0 * fly.a_mm * np.sin(fly.theta_mm)
    dx = np.diff(xt) / fly.dt
    dy = np.diff(yt) / fly.dt
    th = fly.theta_mm[:-1]
    return -dx * np.sin(th) + dy * np.cos(th), "mm/s"


# -------- "yaw" and "phi" derivatives ---------------------------------------


def yaw(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Difference between body orientation and direction of motion (rad).

    Matches the conceptual definition in compute_yaw.m: yaw = phi - theta,
    wrapped to [-pi, pi).
    """
    p, _ = phi(fly)
    return modrange(p - fly.theta_mm, -np.pi, np.pi), "rad"


def phisideways(fly: FlyTrack) -> Tuple[np.ndarray, str]:
    """Direction of motion relative to body axis, "sideways" component.

    The angle in [-pi/2, pi/2] that measures how sideways the motion is.
    """
    y, _ = yaw(fly)
    return modrange(y, -np.pi / 2, np.pi / 2), "rad"


# -------- arena-based features (assume circular arena) ----------------------


def arena_r(fly: FlyTrack, arena_center: tuple = (0.0, 0.0)) -> Tuple[np.ndarray, str]:
    """Distance from fly centroid to arena center."""
    cx, cy = arena_center
    return np.hypot(fly.x_mm - cx, fly.y_mm - cy), "mm"


def arena_angle(fly: FlyTrack, arena_center: tuple = (0.0, 0.0)) -> Tuple[np.ndarray, str]:
    """Angle from arena center to fly centroid (rad)."""
    cx, cy = arena_center
    return np.arctan2(fly.y_mm - cy, fly.x_mm - cx), "rad"


def dist2wall(fly: FlyTrack, arena_radius_mm: float,
              arena_center: tuple = (0.0, 0.0)) -> Tuple[np.ndarray, str]:
    """Distance from fly centroid to the (circular) arena wall."""
    r, _ = arena_r(fly, arena_center)
    return arena_radius_mm - r, "mm"


def angle2wall(fly: FlyTrack, arena_center: tuple = (0.0, 0.0)) -> Tuple[np.ndarray, str]:
    """Angle from the fly's body axis to the nearest wall point (rad).

    Matches compute_angle2wall.m: modrange(arena_angle - theta_mm, -pi, pi).
    """
    aa, _ = arena_angle(fly, arena_center)
    return modrange(aa - fly.theta_mm, -np.pi, np.pi), "rad"


def dangle2wall(fly: FlyTrack, arena_center: tuple = (0.0, 0.0)) -> Tuple[np.ndarray, str]:
    """Rate of change of angle2wall. Length: nframes - 1."""
    aw, _ = angle2wall(fly, arena_center)
    if len(aw) < 2:
        return np.zeros(0), "rad/s"
    return modrange(np.diff(aw), -np.pi, np.pi) / fly.dt, "rad/s"


# -------- generic abs / d-wrappers ------------------------------------------
# Instead of porting one file per abs_* or d_* wrapper, expose two combinators.


def absfeat(feat_data: np.ndarray) -> np.ndarray:
    """Equivalent to JAABA's compute_abs__template: absolute value of a feature."""
    return np.abs(feat_data)


def dfeat(feat_data: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """First-difference of a feature, divided by dt. Output length: len(x) - 1.

    Most compute_d*.m wrappers are this simple. For angular features that need
    wrap-around handling, use ``angular_diff`` from trx.py instead.
    """
    if len(feat_data) < 2:
        return np.zeros(0)
    return np.diff(feat_data) / dt
