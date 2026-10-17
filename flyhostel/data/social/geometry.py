"""
Array kernels shared by both pipelines. Every function takes ``xp`` (numpy or cupy)
and broadcasts over leading axes. Coordinates are arena mm, shape (..., 2).
"""
import warnings
import numpy as np


def pair_distance(xy, i, j, xp=np):
    """Centroid distance between flies i and j over time. xy: (T, N, 2) -> (T,)"""
    return xp.sqrt(((xy[:, i] - xy[:, j]) ** 2).sum(-1))


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def point_segment_distance(p, a, b, xp=np):
    """Distance from point p to segment a-b (a zero-length segment is a point)."""
    ab = b - a
    denom = (ab * ab).sum(-1)
    safe = xp.where(denom > 0, denom, 1)
    u = xp.clip(((p - a) * ab).sum(-1) / safe, 0, 1)
    u = xp.where(denom > 0, u, 0)
    closest = a + u[..., None] * ab
    return xp.sqrt(((p - closest) ** 2).sum(-1))


def segment_distance(p0, p1, q0, q1, xp=np):
    """Shortest distance between 2D segments p0-p1 and q0-q1; 0 when they cross.

    Broadcasts over leading axes. Any NaN endpoint (keypoint not tracked) gives +inf,
    so a missing segment can never win a minimum.
    """
    d = xp.minimum(
        xp.minimum(point_segment_distance(p0, q0, q1, xp), point_segment_distance(p1, q0, q1, xp)),
        xp.minimum(point_segment_distance(q0, p0, p1, xp), point_segment_distance(q1, p0, p1, xp)),
    )
    r, s = p1 - p0, q1 - q0
    crosses = (_cross(r, q0 - p0) * _cross(r, q1 - p0) < 0) & (_cross(s, p0 - q0) * _cross(s, p1 - q0) < 0)
    d = xp.where(crosses, 0, d)
    return xp.where(xp.isnan(d), xp.inf, d)


def centered_velocity(xy, fps, window_s):
    """Velocity (units/s) as displacement across a centred window of ``window_s``.

    Differencing across a window instead of consecutive frames keeps tracking jitter
    from dominating the speed of a still fly. Edges are NaN. Time is axis 0.
    """
    h = max(1, int(round(window_s * fps / 2)))
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    if xy.shape[0] > 2 * h:
        v[h:-h] = (xy[2 * h:] - xy[:-2 * h]) * (fps / (2 * h))
    return v


def path_speed(xy, fps, window_s):
    """Moving average of per-frame step length (units/s); time is axis 0, xy (..., 2).

    Unlike centered_velocity this is a path length, so an oscillation (a leg kicking back
    and forth) does not cancel out when its period matches the window. Tracking jitter
    adds a positive floor, which callers must threshold against a baseline.
    """
    step = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * fps
    step = np.concatenate([np.full((1,) + step.shape[1:], np.nan, step.dtype), step])
    n = max(1, int(round(window_s * fps)))
    ok = np.isfinite(step)
    kernel = np.ones(n)
    smooth = lambda x: np.apply_along_axis(np.convolve, 0, x, kernel, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        return (smooth(np.where(ok, step, 0)) / smooth(ok.astype(float))).astype(np.float32)


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def body_axes(pose, nodes):
    """Heading and anatomical-left axis of one fly from its pose.

    pose: (L, K, 2) arena coordinates.
    Returns heading (L, 2) unit vector abdomen->head and left (L, 2): the unit vector
    perpendicular to heading that points to the fly's LEFT legs. Left is taken from the
    legs, not from a cross product, so it does not depend on whether the camera image
    is mirrored. left is NaN when no leg pair is tracked.
    """
    idx = {n: k for k, n in enumerate(nodes)}
    heading = pose[:, idx["head"]] - pose[:, idx["abdomen"]]
    heading = heading / np.linalg.norm(heading, axis=-1, keepdims=True)

    left_legs = [idx[n] for n in ("fLL", "mLL", "rLL") if n in idx]
    right_legs = [idx[n] for n in ("fRL", "mRL", "rRL") if n in idx]
    perp = np.stack([-heading[:, 1], heading[:, 0]], axis=-1)
    if left_legs and right_legs:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows
            lateral = np.nanmean(pose[:, left_legs], axis=1) - np.nanmean(pose[:, right_legs], axis=1)
        sign = np.sign((lateral * perp).sum(-1))
        left = perp * np.where(sign == 0, np.nan, sign)[:, None]
    else:
        left = np.full_like(perp, np.nan)
    return heading, left


def bearing_sector(heading, left, vector, front_deg=45.0, back_deg=135.0):
    """Where ``vector`` points relative to a fly's body: anterior/left/right/posterior.

    heading, left, vector: (..., 2). Returns (angle_deg, sector) where angle_deg is the
    unsigned angle between heading and vector.
    """
    v = vector / np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = np.degrees(np.arccos(np.clip((heading * v).sum(-1), -1, 1)))
    side = np.where((left * v).sum(-1) > 0, "left", "right")
    side = np.where(np.isnan(left).any(-1), "lateral", side)
    sector = np.where(angle <= front_deg, "anterior", np.where(angle >= back_deg, "posterior", side))
    sector = np.where(np.isnan(angle), "unknown", sector)
    return angle, sector

