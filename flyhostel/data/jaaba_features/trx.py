"""
Trx data model - a Python replacement for JAABA/FlyTracker's MATLAB ``trx`` struct array.

In the original MATLAB code, ``trx`` is a struct array where ``trx(fly).x_mm`` is a
1xN vector of the fly's x-position in millimeters, with N = trx(fly).nframes. Flies
in the same experiment may have different start and end frames. This is handled with
three fields:
    firstframe : absolute frame index (1-based) where this fly first appears
    endframe   : absolute frame index where this fly last appears
    off        : offset such that trx(fly).x_mm( t + off ) is the x-position at
                 absolute frame t. In MATLAB with 1-based indexing,
                 off = 1 - firstframe.

We adopt the same convention here but use 0-based indexing throughout:
    off = -firstframe
    trx[fly].x_mm[t + off] is the value at absolute frame t.

Pairwise functions compute features only on the overlap [t0, t1] where both flies
exist; outside the overlap the value is NaN (matching MATLAB).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


@dataclass
class FlyTrack:
    """Per-fly trajectory in mm units.

    Coordinate convention: x_mm, y_mm are centroid coordinates. theta_mm is the
    body orientation in radians, measured counter-clockwise from the +x axis,
    where the +theta direction points from tail toward nose. a_mm and b_mm are
    the SEMI-major and SEMI-minor axes of the body ellipse (note: many
    downstream JAABA functions pass 2*a_mm, 2*b_mm when they want the full axes).

    Wing angles: wing_anglel and wing_angler are signed angles relative to the
    body axis. By JAABA convention, when both wings are extended symmetrically
    forward, wing_anglel < 0 and wing_angler > 0.
    """
    fly_id: int
    firstframe: int            # absolute (0-based) frame index of first appearance
    nframes: int               # number of consecutive frames this fly is tracked
    dt: float                  # frame interval in seconds (scalar; assumes uniform fps)

    # all the per-frame arrays below have length nframes
    x_mm: np.ndarray
    y_mm: np.ndarray
    a_mm: np.ndarray           # SEMI-major axis
    b_mm: np.ndarray           # SEMI-minor axis
    theta_mm: np.ndarray       # orientation, radians

    # optional wing tracks (None if not available)
    wing_anglel: Optional[np.ndarray] = None
    wing_angler: Optional[np.ndarray] = None
    wing_areal_mm: Optional[np.ndarray] = None
    wing_arear_mm: Optional[np.ndarray] = None
    wing_lengthl_mm: Optional[np.ndarray] = None
    wing_lengthr_mm: Optional[np.ndarray] = None

    # optional area in mm^2 (set from a_mm and b_mm if not provided)
    area_mm: Optional[np.ndarray] = None

    # ROI (region-of-interest) index, used by JAABA to restrict pairwise features
    # to flies in the same arena. If you have one arena, set to 0 for all flies.
    roi: int = 0

    @property
    def endframe(self) -> int:
        """Absolute frame index of last appearance (inclusive)."""
        return self.firstframe + self.nframes - 1

    @property
    def off(self) -> int:
        """Offset such that x_mm[t + off] gives the value at absolute frame t."""
        return -self.firstframe

    def overlap_range(self, other: "FlyTrack") -> Optional[range]:
        """Return absolute-frame range [t0, t1+1) over which both flies exist."""
        t0 = max(self.firstframe, other.firstframe)
        t1 = min(self.endframe, other.endframe)
        if t1 < t0:
            return None
        return range(t0, t1 + 1)

    def __post_init__(self) -> None:
        arrs = [self.x_mm, self.y_mm, self.a_mm, self.b_mm, self.theta_mm]
        for arr in arrs:
            if len(arr) != self.nframes:
                raise ValueError(
                    f"Array length {len(arr)} does not match nframes={self.nframes}"
                )
        # compute area_mm if not provided (using ellipse area: pi * a * b * 4
        # since a_mm and b_mm are SEMI-axes, the full ellipse area is pi*a*b
        # but the MATLAB code uses 2a and 2b in many places, so the "area" they
        # report is pi * (2a) * (2b) / 4 = pi * a * b. We follow that.)
        if self.area_mm is None:
            self.area_mm = np.pi * self.a_mm * self.b_mm


@dataclass
class Trx:
    """A collection of FlyTrack objects forming one experiment.

    JAABA's ``trx`` MATLAB struct array carries some experiment-level metadata
    in fields like ``trx.perframe_params`` and ``trx.landmark_params``. We bundle
    that here.
    """
    flies: List[FlyTrack]

    # parameters used by some features
    fov: float = np.deg2rad(270.0)     # field of view for anglesub features
    arena_radius_mm: Optional[float] = None    # for round arenas, used by dist2wall
    arena_center_mm: tuple = (0.0, 0.0)        # arena center in mm
    thetafil: Optional[np.ndarray] = None      # smoothing filter for theta features

    def __post_init__(self) -> None:
        # build fast lookup by id
        self._by_id: Dict[int, FlyTrack] = {f.fly_id: f for f in self.flies}
        # default theta smoothing filter (a small triangular kernel; JAABA uses
        # one specified by trx.perframe_params.thetafil. The exact filter
        # depends on JAABA config; this is a sensible default.)
        if self.thetafil is None:
            f = np.array([1.0, 4.0, 6.0, 4.0, 1.0])
            self.thetafil = f / f.sum()

    def __getitem__(self, fly_id: int) -> FlyTrack:
        return self._by_id[fly_id]

    @property
    def fly_ids(self) -> List[int]:
        return [f.fly_id for f in self.flies]

    def flies_in_roi(self, roi: int) -> List[int]:
        return [f.fly_id for f in self.flies if f.roi == roi]


# -------- utility functions --------------------------------------------------


def modrange(x: np.ndarray, low: float, high: float) -> np.ndarray:
    """Equivalent to MATLAB modrange(x, low, high): wraps x into [low, high)."""
    return ((x - low) % (high - low)) + low


def central_diff(x: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Central difference matching JAABA's convention for d/dt features.

    JAABA's pattern (see compute_phi.m):
        dx = [x[1]-x[0], (x[2:]-x[:-2])/2, x[-1]-x[-2]]   (MATLAB indexing)
    which is forward-diff at the boundaries and central-diff in the middle,
    producing an output of length len(x).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return np.zeros_like(x)
    out = np.empty_like(x)
    out[0] = x[1] - x[0]
    out[-1] = x[-1] - x[-2]
    if n > 2:
        out[1:-1] = (x[2:] - x[:-2]) * 0.5
    return out / dt


def angular_diff(theta: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """diff(theta) wrapped into [-pi, pi), divided by dt. Matches compute_dtheta.m.

    Output length is len(theta) - 1.
    """
    theta = np.asarray(theta, dtype=float)
    if len(theta) < 2:
        return np.zeros(0)
    d = modrange(np.diff(theta), -np.pi, np.pi)
    return d / dt
