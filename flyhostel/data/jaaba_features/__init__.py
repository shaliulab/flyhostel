"""
jaaba_features - Python port of JAABA per-frame feature extraction.

A faithful port of the single-fly, wing, pair, and closest-fly features from
JAABA/perframe/compute_perframe_features (MATLAB). Spacetime/HOG-HOF features
and APT (Animal Part Tracker) features are deliberately not included; those
require the raw video and an APT-style pose pipeline, respectively.

Quick start::

    from jaaba_features import FlyTrack, Trx
    from jaaba_features.single_fly import velmag_ctr, dtheta, yaw
    from jaaba_features.pair_features import dcenter_pair, anglesub_pair
    from jaaba_features.closest import closest_fly, nflies_close

    # build FlyTrack objects from your tracking data
    fly1 = FlyTrack(fly_id=0, firstframe=0, nframes=1000, dt=1/30, ...)
    fly2 = FlyTrack(fly_id=1, firstframe=0, nframes=1000, dt=1/30, ...)
    trx = Trx(flies=[fly1, fly2], arena_radius_mm=12.5)

    # single-fly features
    v, _ = velmag_ctr(fly1)
    omega, _ = dtheta(fly1)

    # pair features
    d = dcenter_pair(fly1, fly2)
    asub = anglesub_pair(fly1, fly2)

    # closest-fly features
    cf, mind = closest_fly(trx, fly1_id=0, metric=dcenter_pair)
"""

from .trx import FlyTrack, Trx, modrange, central_diff, angular_diff

__all__ = [
    "FlyTrack",
    "Trx",
    "modrange",
    "central_diff",
    "angular_diff",
]

__version__ = "0.1.0"
