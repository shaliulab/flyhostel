"""
Pairwise (fly-pair) features.

Each pair function takes two FlyTrack objects (fly1, fly2) and returns a 1D array
of length fly1.nframes, with NaN outside the overlap interval where both flies
exist. This matches JAABA's convention exactly.

The geometric helpers (anglesubtended, ellipse2ellipsedist, etc.) are ports of
the corresponding .m files.
"""

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np

from .trx import FlyTrack


# -------- pairwise centroid/nose/tail distances -----------------------------


def dcenter_pair(fly1: FlyTrack, fly2: FlyTrack) -> np.ndarray:
    """Centroid-to-centroid distance, mm. Length fly1.nframes; NaN outside overlap.

    Port of dcenter_pair.m.
    """
    out = np.full(fly1.nframes, np.nan)
    rng = fly1.overlap_range(fly2)
    if rng is None:
        return out
    o1, o2 = fly1.off, fly2.off
    t0, t1 = rng.start, rng.stop - 1
    i0, i1 = t0 + o1, t1 + o1
    j0, j1 = t0 + o2, t1 + o2
    dx = fly2.x_mm[j0:j1 + 1] - fly1.x_mm[i0:i1 + 1]
    dy = fly2.y_mm[j0:j1 + 1] - fly1.y_mm[i0:i1 + 1]
    out[i0:i1 + 1] = np.sqrt(dx * dx + dy * dy)
    return out


def dnose2tail_pair(fly1: FlyTrack, fly2: FlyTrack) -> np.ndarray:
    """Distance from nose of fly1 to tail of fly2."""
    out = np.full(fly1.nframes, np.nan)
    rng = fly1.overlap_range(fly2)
    if rng is None:
        return out
    o1, o2 = fly1.off, fly2.off
    t0, t1 = rng.start, rng.stop - 1
    i0, i1 = t0 + o1, t1 + o1
    j0, j1 = t0 + o2, t1 + o2
    # nose of fly1
    x_nose = fly1.x_mm[i0:i1 + 1] + 2.0 * fly1.a_mm[i0:i1 + 1] * np.cos(fly1.theta_mm[i0:i1 + 1])
    y_nose = fly1.y_mm[i0:i1 + 1] + 2.0 * fly1.a_mm[i0:i1 + 1] * np.sin(fly1.theta_mm[i0:i1 + 1])
    # tail of fly2
    x_tail = fly2.x_mm[j0:j1 + 1] - 2.0 * fly2.a_mm[j0:j1 + 1] * np.cos(fly2.theta_mm[j0:j1 + 1])
    y_tail = fly2.y_mm[j0:j1 + 1] - 2.0 * fly2.a_mm[j0:j1 + 1] * np.sin(fly2.theta_mm[j0:j1 + 1])
    dx = x_tail - x_nose
    dy = y_tail - y_nose
    out[i0:i1 + 1] = np.sqrt(dx * dx + dy * dy)
    return out


def dnose2center_pair(fly1: FlyTrack, fly2: FlyTrack) -> np.ndarray:
    """Distance from nose of fly1 to centroid of fly2."""
    out = np.full(fly1.nframes, np.nan)
    rng = fly1.overlap_range(fly2)
    if rng is None:
        return out
    o1, o2 = fly1.off, fly2.off
    t0, t1 = rng.start, rng.stop - 1
    i0, i1 = t0 + o1, t1 + o1
    j0, j1 = t0 + o2, t1 + o2
    x_nose = fly1.x_mm[i0:i1 + 1] + 2.0 * fly1.a_mm[i0:i1 + 1] * np.cos(fly1.theta_mm[i0:i1 + 1])
    y_nose = fly1.y_mm[i0:i1 + 1] + 2.0 * fly1.a_mm[i0:i1 + 1] * np.sin(fly1.theta_mm[i0:i1 + 1])
    dx = fly2.x_mm[j0:j1 + 1] - x_nose
    dy = fly2.y_mm[j0:j1 + 1] - y_nose
    out[i0:i1 + 1] = np.sqrt(dx * dx + dy * dy)
    return out


def dcenter2nose_pair(fly1: FlyTrack, fly2: FlyTrack) -> np.ndarray:
    """Distance from centroid of fly1 to nose of fly2.

    Returned in fly1's frame (length fly1.nframes, NaN outside overlap).
    """
    out = np.full(fly1.nframes, np.nan)
    rng = fly1.overlap_range(fly2)
    if rng is None:
        return out
    o1, o2 = fly1.off, fly2.off
    t0, t1 = rng.start, rng.stop - 1
    i0, i1 = t0 + o1, t1 + o1
    j0, j1 = t0 + o2, t1 + o2
    xn2 = fly2.x_mm[j0:j1 + 1] + 2.0 * fly2.a_mm[j0:j1 + 1] * np.cos(fly2.theta_mm[j0:j1 + 1])
    yn2 = fly2.y_mm[j0:j1 + 1] + 2.0 * fly2.a_mm[j0:j1 + 1] * np.sin(fly2.theta_mm[j0:j1 + 1])
    dx = xn2 - fly1.x_mm[i0:i1 + 1]
    dy = yn2 - fly1.y_mm[i0:i1 + 1]
    out[i0:i1 + 1] = np.sqrt(dx * dx + dy * dy)
    return out


def dell2nose_pair(fly1: FlyTrack, fly2: FlyTrack) -> np.ndarray:
    """Minimum distance from a point on fly1's body ellipse to fly2's nose.

    Uses a sampling approach (20 samples around the ellipse) matching JAABA's
    dell2nose_pair.m, which calls ellipse2ellipsedist_hack.
    """
    nsamples = 20
    out = np.full(fly1.nframes, np.nan)
    rng = fly1.overlap_range(fly2)
    if rng is None:
        return out
    o1, o2 = fly1.off, fly2.off
    t0, t1 = rng.start, rng.stop - 1
    # Sample angles around ellipse 1
    psi = np.linspace(0, 2 * np.pi, nsamples, endpoint=False)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    for t in range(t0, t1 + 1):
        i, j = t + o1, t + o2
        # ellipse 1 sample points in world frame
        a1, b1 = 2.0 * fly1.a_mm[i], 2.0 * fly1.b_mm[i]
        th1 = fly1.theta_mm[i]
        x_e = (a1 * cos_psi * np.cos(th1)
               - b1 * sin_psi * np.sin(th1) + fly1.x_mm[i])
        y_e = (a1 * cos_psi * np.sin(th1)
               + b1 * sin_psi * np.cos(th1) + fly1.y_mm[i])
        # nose of fly2
        xn = fly2.x_mm[j] + 2.0 * fly2.a_mm[j] * np.cos(fly2.theta_mm[j])
        yn = fly2.y_mm[j] + 2.0 * fly2.a_mm[j] * np.sin(fly2.theta_mm[j])
        d = np.hypot(x_e - xn, y_e - yn)
        out[i] = d.min()
    return out


def dnose2ell_pair(fly1: FlyTrack, fly2: FlyTrack) -> np.ndarray:
    """Minimum distance from fly1's nose to a point on fly2's body ellipse.

    Port of dnose2ell_pair.m. By symmetry with dell2nose_pair.
    """
    nsamples = 20
    out = np.full(fly1.nframes, np.nan)
    rng = fly1.overlap_range(fly2)
    if rng is None:
        return out
    o1, o2 = fly1.off, fly2.off
    t0, t1 = rng.start, rng.stop - 1
    psi = np.linspace(0, 2 * np.pi, nsamples, endpoint=False)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    for t in range(t0, t1 + 1):
        i, j = t + o1, t + o2
        # nose of fly1
        xn = fly1.x_mm[i] + 2.0 * fly1.a_mm[i] * np.cos(fly1.theta_mm[i])
        yn = fly1.y_mm[i] + 2.0 * fly1.a_mm[i] * np.sin(fly1.theta_mm[i])
        # ellipse 2 sample points
        a2, b2 = 2.0 * fly2.a_mm[j], 2.0 * fly2.b_mm[j]
        th2 = fly2.theta_mm[j]
        x_e = (a2 * cos_psi * np.cos(th2)
               - b2 * sin_psi * np.sin(th2) + fly2.x_mm[j])
        y_e = (a2 * cos_psi * np.sin(th2)
               + b2 * sin_psi * np.cos(th2) + fly2.y_mm[j])
        d = np.hypot(x_e - xn, y_e - yn)
        out[i] = d.min()
    return out


def dell2ell_pair(fly1: FlyTrack, fly2: FlyTrack) -> np.ndarray:
    """Minimum distance between points on fly1's and fly2's body ellipses.

    Port of dell2ell_pair.m + ellipse2ellipsedist_hack.m (sample-based approximation).
    """
    nsamples = 20
    out = np.full(fly1.nframes, np.nan)
    rng = fly1.overlap_range(fly2)
    if rng is None:
        return out
    o1, o2 = fly1.off, fly2.off
    t0, t1 = rng.start, rng.stop - 1
    psi = np.linspace(0, 2 * np.pi, nsamples, endpoint=False)
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    for t in range(t0, t1 + 1):
        i, j = t + o1, t + o2
        # ellipse 1 points
        a1, b1 = 2.0 * fly1.a_mm[i], 2.0 * fly1.b_mm[i]
        th1 = fly1.theta_mm[i]
        x_e1 = (a1 * cos_psi * np.cos(th1)
                - b1 * sin_psi * np.sin(th1) + fly1.x_mm[i])
        y_e1 = (a1 * cos_psi * np.sin(th1)
                + b1 * sin_psi * np.cos(th1) + fly1.y_mm[i])
        # ellipse 2 points
        a2, b2 = 2.0 * fly2.a_mm[j], 2.0 * fly2.b_mm[j]
        th2 = fly2.theta_mm[j]
        x_e2 = (a2 * cos_psi * np.cos(th2)
                - b2 * sin_psi * np.sin(th2) + fly2.x_mm[j])
        y_e2 = (a2 * cos_psi * np.sin(th2)
                + b2 * sin_psi * np.cos(th2) + fly2.y_mm[j])
        # min pairwise distance
        dx = x_e1[:, None] - x_e2[None, :]
        dy = y_e1[:, None] - y_e2[None, :]
        d = np.hypot(dx, dy)
        out[i] = d.min()
    return out


# -------- pairwise angles ---------------------------------------------------


def anglefrom1to2_pair(fly1: FlyTrack, fly2: FlyTrack) -> np.ndarray:
    """Angle in fly1's body frame from fly1's heading to fly2's centroid.

    A value of 0 means fly2 is directly in front of fly1; +pi/2 means to the
    left (in the fly's frame); -pi/2 to the right.
    """
    out = np.full(fly1.nframes, np.nan)
    rng = fly1.overlap_range(fly2)
    if rng is None:
        return out
    o1, o2 = fly1.off, fly2.off
    t0, t1 = rng.start, rng.stop - 1
    i0, i1 = t0 + o1, t1 + o1
    j0, j1 = t0 + o2, t1 + o2
    dx = fly2.x_mm[j0:j1 + 1] - fly1.x_mm[i0:i1 + 1]
    dy = fly2.y_mm[j0:j1 + 1] - fly1.y_mm[i0:i1 + 1]
    world_ang = np.arctan2(dy, dx)
    rel = world_ang - fly1.theta_mm[i0:i1 + 1]
    out[i0:i1 + 1] = ((rel + np.pi) % (2 * np.pi)) - np.pi
    return out


def magveldiff_pair(fly1: FlyTrack, fly2: FlyTrack) -> np.ndarray:
    """Magnitude of the velocity difference between two flies, mm/s.

    Length: fly1.nframes (NaN-padded). Uses one-sided forward differences inside
    the overlap (so the last frame in the overlap is NaN).
    """
    out = np.full(fly1.nframes, np.nan)
    rng = fly1.overlap_range(fly2)
    if rng is None:
        return out
    o1, o2 = fly1.off, fly2.off
    # we need t in [t0, t1 - 1] so that t+1 is still in overlap
    t0, t1 = rng.start, rng.stop - 1
    if t1 == t0:
        return out
    i0, i1 = t0 + o1, (t1 - 1) + o1   # inclusive end
    j0, j1 = t0 + o2, (t1 - 1) + o2
    v1x = (fly1.x_mm[i0 + 1:i1 + 2] - fly1.x_mm[i0:i1 + 1]) / fly1.dt
    v1y = (fly1.y_mm[i0 + 1:i1 + 2] - fly1.y_mm[i0:i1 + 1]) / fly1.dt
    v2x = (fly2.x_mm[j0 + 1:j1 + 2] - fly2.x_mm[j0:j1 + 1]) / fly2.dt
    v2y = (fly2.y_mm[j0 + 1:j1 + 2] - fly2.y_mm[j0:j1 + 1]) / fly2.dt
    out[i0:i1 + 1] = np.hypot(v1x - v2x, v1y - v2y)
    return out


# -------- anglesubtended geometric port -------------------------------------
# This block ports anglesubtended.m + its helpers (eyeoffly1givenfly2,
# checkinborder, computetangentpoints, limitbyfov). The math is preserved
# exactly; only the loop is replaced by a Python for-loop over frames.
# Inputs: a, b are the FULL axes (not semi-axes), matching JAABA's
# convention when anglesub_pair calls anglesubtended with 2*a_mm, 2*b_mm.


_EPS = 1e-5


def _eye_of_fly1_given_fly2(x1: float, y1: float, x2: float, y2: float,
                            a1: float, theta1: float, theta2: float) -> Tuple[float, float]:
    """The "eye" of fly1 is at a1 ahead of its centroid along theta1.
    Returns this point in fly2's body frame (centered on fly2, rotated to its axis).
    """
    c1 = x1 + a1 * np.cos(theta1)
    d1 = y1 + a1 * np.sin(theta1)
    c1 -= x2
    d1 -= y2
    c = c1 * np.cos(theta2) + d1 * np.sin(theta2)
    d = d1 * np.cos(theta2) - c1 * np.sin(theta2)
    return c, d


def _check_in_border(c: float, d: float, a: float, b: float) -> Tuple[bool, bool]:
    """Is point (c,d) inside or on the border of the ellipse x^2/a^2 + y^2/b^2 = 1?"""
    A = (c * c) / (a * a) + (d * d) / (b * b)
    if abs(A - 1.0) < _EPS:
        return True, False
    if A < 1.0:
        return False, True
    return False, False


def _compute_tangent_points(c: float, d: float, a: float, b: float,
                            theta1: float, theta2: float) -> Tuple[float, float]:
    """The angles psi1, psi2 from the "eye" of fly1 (at (c,d) in fly2's frame)
    to the two tangent points on fly2's ellipse, expressed in fly1's body frame.
    Direct port of computetangentpoints from anglesubtended.m.
    """
    # Quadratic for cos(phi)
    A = b * b * c * c + a * a * d * d
    B = -2.0 * a * b * b * c
    C = a * a * (b * b - d * d)
    D = B * B - 4.0 * A * C
    D = np.sqrt(max(D, 0.0))

    possiblephi = np.zeros(6)
    cost = np.zeros(6)
    possiblephi[0] = 0.0
    possiblephi[1] = np.pi
    cost[0] = 0.0 if abs(c - a) < _EPS else np.inf
    cost[1] = 0.0 if abs(c + a) < _EPS else np.inf

    cosphi_p = max(-1.0, min(1.0, (-B + D) / (2.0 * A)))
    possiblephi[2] = np.arccos(cosphi_p)
    possiblephi[3] = -possiblephi[2]
    cosphi_m = max(-1.0, min(1.0, (-B - D) / (2.0 * A)))
    possiblephi[4] = np.arccos(cosphi_m)
    possiblephi[5] = -possiblephi[4]

    sinphi = np.sin(possiblephi[2:])
    cosphi = np.cos(possiblephi[2:])
    cost[2:] = np.abs((b * sinphi - d) * (-a * sinphi)
                      - (a * cosphi - c) * (b * cosphi))

    order = np.argsort(cost)
    phi1 = possiblephi[order[0]]
    phi2 = possiblephi[order[1]]

    x1 = a * np.cos(phi1)
    y1 = b * np.sin(phi1)
    x2 = a * np.cos(phi2)
    y2 = b * np.sin(phi2)

    psi1 = np.arctan2(y1 - d, x1 - c)
    psi2 = np.arctan2(y2 - d, x2 - c)
    psi0 = np.arctan2(-d, -c)

    # rotate into fly1's frame
    psi0 += theta2 - theta1
    psi1 += theta2 - theta1
    psi2 += theta2 - theta1

    # put psi1 in [-pi, pi)
    psi1 = ((psi1 + np.pi) % (2 * np.pi)) - np.pi
    dpsi01 = (psi0 - psi1) % (2 * np.pi)
    psi0 = psi1 + dpsi01
    dpsi21 = (psi2 - psi1) % (2 * np.pi)
    psi2 = psi1 + dpsi21

    if psi2 < psi0:
        psi1, psi2 = psi2, psi1
        psi1 = ((psi1 + np.pi) % (2 * np.pi)) - np.pi
        dpsi21 = (psi2 - psi1) % (2 * np.pi)
        psi2 = psi1 + dpsi21

    return psi1, psi2


def _limit_by_fov(psi1: float, psi2: float, fov: float) -> float:
    """Clip the angular interval [psi1, psi2] to the field of view [-fov/2, fov/2].

    Direct port of limitbyfov from anglesubtended.m.
    """
    fov1 = -fov / 2.0
    fov2 = fov1 + fov
    d = (psi1 - fov1) % (2 * np.pi)
    psi1 = fov1 + d
    d = (psi2 - fov1) % (2 * np.pi)
    psi2 = fov1 + d

    if fov2 <= psi1 <= psi2:
        return 0.0
    elif fov2 <= psi2 <= psi1:
        return ((fov1 - psi1) % (2 * np.pi)) + ((fov2 - psi2) % (2 * np.pi))
    elif psi1 <= fov2 <= psi2:
        return fov2 - psi1
    elif psi1 <= psi2 <= fov2:
        return psi2 - psi1
    elif psi2 <= fov2 <= psi1:
        return psi2 - fov1
    else:  # psi2 <= psi1 <= fov2
        return (psi2 - fov1) + (fov2 - psi1)


def anglesubtended(x1: float, y1: float, a1: float, b1: float, theta1: float,
                   x2: float, y2: float, a2: float, b2: float, theta2: float,
                   fov: float) -> float:
    """Angle subtended by ellipse 2 in the visual field of fly 1.

    Inputs are passed exactly as JAABA does: a, b here are the FULL axes
    (not semi-axes), but JAABA's inner geometry then treats them as semi-axes
    in the equation x = a*cos(phi), y = b*sin(phi). The net effect is that
    the body's projected visual size is computed using axes equal to 2*a_mm
    and 2*b_mm of the fly's tracked ellipse — i.e., effectively double the
    body half-width — which is a JAABA convention preserved by this port.

    fov : field-of-view in radians, centered on the fly's heading direction
          (so a 270° fov covers ±135° from forward).

    Direct port of anglesubtended.m.
    """
    if max(a2, b2) < _EPS:
        return _EPS
    c, d = _eye_of_fly1_given_fly2(x1, y1, x2, y2, a1, theta1, theta2)
    on_border, in_border = _check_in_border(c, d, a2, b2)
    if on_border:
        return min(np.pi, fov)
    if in_border:
        return fov
    psi1, psi2 = _compute_tangent_points(c, d, a2, b2, theta1, theta2)
    return _limit_by_fov(psi1, psi2, fov)


def anglesub_pair(fly1: FlyTrack, fly2: FlyTrack, fov: float = None) -> np.ndarray:
    """Per-frame angle subtended by fly2 in fly1's visual field, in radians.

    Length: fly1.nframes (NaN-padded outside overlap). fov defaults to 270 deg.
    """
    if fov is None:
        fov = np.deg2rad(270.0)
    out = np.full(fly1.nframes, np.nan)
    rng = fly1.overlap_range(fly2)
    if rng is None:
        return out
    o1, o2 = fly1.off, fly2.off
    for t in rng:
        i, j = t + o1, t + o2
        out[i] = anglesubtended(
            fly1.x_mm[i], fly1.y_mm[i],
            2.0 * fly1.a_mm[i], 2.0 * fly1.b_mm[i], fly1.theta_mm[i],
            fly2.x_mm[j], fly2.y_mm[j],
            2.0 * fly2.a_mm[j], 2.0 * fly2.b_mm[j], fly2.theta_mm[j],
            fov,
        )
    return out
