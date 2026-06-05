"""
Tests for the JAABA features port.

These verify against analytically-known cases where the expected output can
be derived by hand. They are not a substitute for a head-to-head comparison
against MATLAB on real data (which we still recommend for full validation),
but they catch most bugs in the port and pin down the conventions.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaaba_features import FlyTrack, Trx, modrange, central_diff, angular_diff
from jaaba_features.single_fly import (
    ecc, area_mm, xnose_mm, ynose_mm, phi, velmag_ctr, velmag_nose,
    dtheta, d2theta, da, db, decc, du_ctr, dv_ctr, yaw,
    arena_r, dist2wall, angle2wall,
)
from jaaba_features.wing_features import (
    max_wing_angle, min_wing_angle, mean_wing_angle, wing_angle_diff,
    wing_angle_imbalance,
)
from jaaba_features.pair_features import (
    dcenter_pair, dnose2tail_pair, dnose2center_pair, dcenter2nose_pair,
    anglefrom1to2_pair, anglesub_pair, anglesubtended,
)
from jaaba_features.closest import closest_fly, nflies_close


def assert_close(actual, expected, atol=1e-9, msg=""):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    # treat NaNs as equal
    mask = ~(np.isnan(actual) & np.isnan(expected))
    diff = np.abs(actual[mask] - expected[mask])
    if diff.size and diff.max() > atol:
        raise AssertionError(
            f"{msg}: max diff {diff.max()} > {atol}\n"
            f"actual:   {actual}\nexpected: {expected}"
        )


def make_simple_fly(fly_id=0, firstframe=0, nframes=5, dt=1.0,
                    x0=0.0, y0=0.0, vx=1.0, vy=0.0,
                    a=1.0, b=0.5, theta=0.0):
    """A fly moving in a straight line at constant velocity, constant orientation."""
    t = np.arange(nframes)
    return FlyTrack(
        fly_id=fly_id, firstframe=firstframe, nframes=nframes, dt=dt,
        x_mm=x0 + vx * t * dt,
        y_mm=y0 + vy * t * dt,
        a_mm=np.full(nframes, a),
        b_mm=np.full(nframes, b),
        theta_mm=np.full(nframes, theta),
    )


def test_modrange():
    assert abs(modrange(np.array([np.pi + 0.1]), -np.pi, np.pi)[0] - (-np.pi + 0.1)) < 1e-12
    assert abs(modrange(np.array([0.0]), -np.pi, np.pi)[0]) < 1e-12


def test_central_diff():
    x = np.array([0.0, 1.0, 4.0, 9.0, 16.0])  # x^2 at integers
    d = central_diff(x, dt=1.0)
    # forward at first, central in middle, backward at last
    assert d[0] == 1.0
    assert d[1] == (4.0 - 0.0) / 2.0  # = 2.0
    assert d[2] == (9.0 - 1.0) / 2.0  # = 4.0
    assert d[3] == (16.0 - 4.0) / 2.0  # = 6.0
    assert d[4] == 7.0


def test_ecc_and_area():
    fly = make_simple_fly(a=2.0, b=1.0)
    e, _ = ecc(fly)
    assert_close(e, np.full(5, 0.5))
    am, _ = area_mm(fly)
    assert_close(am, np.full(5, np.pi * 2.0 * 1.0))


def test_xnose_ynose():
    # fly at origin, theta=0, semi-major axis 1.0, JAABA uses 2*a so nose at (2,0)
    fly = make_simple_fly(x0=0.0, y0=0.0, vx=0.0, theta=0.0, a=1.0)
    xn, _ = xnose_mm(fly)
    yn, _ = ynose_mm(fly)
    assert_close(xn, np.full(5, 2.0))
    assert_close(yn, np.full(5, 0.0))
    # rotate 90 degrees
    fly2 = make_simple_fly(theta=np.pi / 2, vx=0.0, a=1.0)
    xn2, _ = xnose_mm(fly2)
    yn2, _ = ynose_mm(fly2)
    assert_close(xn2, np.full(5, 0.0), atol=1e-9)
    assert_close(yn2, np.full(5, 2.0))


def test_velmag_constant_velocity():
    fly = make_simple_fly(vx=3.0, vy=4.0, dt=1.0)
    v, _ = velmag_ctr(fly)
    # length = nframes - 1 = 4
    assert v.shape == (4,)
    assert_close(v, np.full(4, 5.0))


def test_phi_constant_motion():
    # moving in +x direction
    fly = make_simple_fly(vx=1.0, vy=0.0, theta=np.pi / 4)
    p, _ = phi(fly)
    assert_close(p, np.full(5, 0.0))
    # if stationary, phi falls back to theta
    fly_still = make_simple_fly(vx=0.0, vy=0.0, theta=np.pi / 4)
    p2, _ = phi(fly_still)
    assert_close(p2, np.full(5, np.pi / 4))


def test_yaw():
    fly = make_simple_fly(vx=1.0, vy=0.0, theta=np.pi / 4)
    y, _ = yaw(fly)
    # motion is along x (phi=0), body axis is +pi/4, so yaw = 0 - pi/4 = -pi/4
    assert_close(y, np.full(5, -np.pi / 4))


def test_dtheta_constant_rotation():
    nframes = 5
    fly = FlyTrack(
        fly_id=0, firstframe=0, nframes=nframes, dt=1.0,
        x_mm=np.zeros(nframes), y_mm=np.zeros(nframes),
        a_mm=np.ones(nframes), b_mm=0.5 * np.ones(nframes),
        theta_mm=np.linspace(0, np.pi / 2, nframes),
    )
    d, _ = dtheta(fly)
    # constant angular velocity
    expected = np.full(nframes - 1, (np.pi / 2) / (nframes - 1))
    assert_close(d, expected)


def test_dtheta_wraparound():
    # jump from -pi+0.1 to pi-0.1 should wrap to -0.2, not 2pi - 0.2
    nframes = 2
    fly = FlyTrack(
        fly_id=0, firstframe=0, nframes=nframes, dt=1.0,
        x_mm=np.zeros(nframes), y_mm=np.zeros(nframes),
        a_mm=np.ones(nframes), b_mm=0.5 * np.ones(nframes),
        theta_mm=np.array([-np.pi + 0.1, np.pi - 0.1]),
    )
    d, _ = dtheta(fly)
    assert_close(d, np.array([-0.2]), atol=1e-9)


def test_arena_features():
    fly = FlyTrack(
        fly_id=0, firstframe=0, nframes=3, dt=1.0,
        x_mm=np.array([0.0, 1.0, 0.0]),
        y_mm=np.array([0.0, 0.0, 1.0]),
        a_mm=np.ones(3), b_mm=0.5 * np.ones(3),
        theta_mm=np.zeros(3),
    )
    r, _ = arena_r(fly, arena_center=(0.0, 0.0))
    assert_close(r, np.array([0.0, 1.0, 1.0]))
    d, _ = dist2wall(fly, arena_radius_mm=5.0)
    assert_close(d, np.array([5.0, 4.0, 4.0]))


def test_wing_angles():
    # symmetric extension: anglel = -0.5, angler = +0.5
    nframes = 3
    fly = FlyTrack(
        fly_id=0, firstframe=0, nframes=nframes, dt=1.0,
        x_mm=np.zeros(nframes), y_mm=np.zeros(nframes),
        a_mm=np.ones(nframes), b_mm=0.5 * np.ones(nframes),
        theta_mm=np.zeros(nframes),
        wing_anglel=np.full(nframes, -0.5),
        wing_angler=np.full(nframes, 0.5),
    )
    mx, _ = max_wing_angle(fly)
    mn, _ = min_wing_angle(fly)
    me, _ = mean_wing_angle(fly)
    df, _ = wing_angle_diff(fly)
    im, _ = wing_angle_imbalance(fly)
    assert_close(mx, np.full(nframes, 0.5))  # max(-(-0.5), 0.5) = 0.5
    assert_close(mn, np.full(nframes, 0.5))  # min(-(-0.5), 0.5) = 0.5 too (symmetric)
    assert_close(me, np.full(nframes, 0.5))  # (-(-0.5)+0.5)/2 = 0.5
    assert_close(df, np.full(nframes, 1.0))  # 0.5 - (-0.5) = 1.0
    assert_close(im, np.full(nframes, 0.0))  # |0.5 + (-0.5)| = 0

    # asymmetric (only right wing): anglel=0, angler=1.0
    fly2 = FlyTrack(
        fly_id=0, firstframe=0, nframes=nframes, dt=1.0,
        x_mm=np.zeros(nframes), y_mm=np.zeros(nframes),
        a_mm=np.ones(nframes), b_mm=0.5 * np.ones(nframes),
        theta_mm=np.zeros(nframes),
        wing_anglel=np.zeros(nframes),
        wing_angler=np.full(nframes, 1.0),
    )
    im2, _ = wing_angle_imbalance(fly2)
    assert_close(im2, np.full(nframes, 1.0))  # |1.0 + 0| = 1.0


def test_pair_dcenter():
    # two flies, fly1 at origin, fly2 at (3, 4) -> distance 5
    fly1 = make_simple_fly(fly_id=0, vx=0.0, x0=0.0, y0=0.0)
    fly2 = make_simple_fly(fly_id=1, vx=0.0, x0=3.0, y0=4.0)
    d = dcenter_pair(fly1, fly2)
    assert_close(d, np.full(5, 5.0))


def test_pair_dcenter_no_overlap():
    fly1 = make_simple_fly(fly_id=0, firstframe=0, nframes=5)
    fly2 = make_simple_fly(fly_id=1, firstframe=10, nframes=5)
    d = dcenter_pair(fly1, fly2)
    assert np.all(np.isnan(d))


def test_pair_dcenter_partial_overlap():
    # fly1: frames 0-4; fly2: frames 2-6. Overlap: frames 2-4 (3 frames).
    fly1 = make_simple_fly(fly_id=0, firstframe=0, nframes=5, x0=0.0, vx=0.0)
    fly2 = make_simple_fly(fly_id=1, firstframe=2, nframes=5, x0=3.0, y0=4.0, vx=0.0)
    d = dcenter_pair(fly1, fly2)
    # fly1's frames 0,1 are before overlap -> NaN
    # frames 2,3,4 -> distance 5
    assert np.isnan(d[0]) and np.isnan(d[1])
    assert_close(d[2:5], np.full(3, 5.0))


def test_anglefrom1to2():
    # fly1 at origin facing +x, fly2 at (0, 1) (directly to the left in image coords).
    # The vector from fly1 to fly2 is (0,1), angle in world = pi/2.
    # Relative to fly1's heading (theta=0): pi/2.
    fly1 = make_simple_fly(fly_id=0, vx=0.0, theta=0.0)
    fly2 = make_simple_fly(fly_id=1, vx=0.0, x0=0.0, y0=1.0, theta=0.0)
    a = anglefrom1to2_pair(fly1, fly2)
    assert_close(a, np.full(5, np.pi / 2), atol=1e-12)


def test_anglesubtended_trivial():
    # fly1 at origin facing +x, fly2 a long way to the east (small subtended angle)
    # we use small ellipse axes
    out = anglesubtended(0, 0, 1.0, 0.5, 0.0,   # fly1 at origin
                         100.0, 0.0, 0.4, 0.4, 0.0,   # fly2 far away
                         fov=np.deg2rad(270))
    assert out > 0 and out < 0.1  # small subtended angle
    # fly2 inside fly1's "eye" position -> should be huge (close to fov)
    out2 = anglesubtended(0, 0, 1.0, 0.5, 0.0,
                          1.0, 0.0, 10.0, 10.0, 0.0,
                          fov=np.deg2rad(270))
    assert out2 >= np.pi - 0.01  # large subtended angle


def test_closest_fly_simple():
    # 3 flies: fly0 at origin, fly1 at (1,0), fly2 at (10,0)
    fly0 = make_simple_fly(fly_id=0, vx=0.0)
    fly1 = make_simple_fly(fly_id=1, vx=0.0, x0=1.0)
    fly2 = make_simple_fly(fly_id=2, vx=0.0, x0=10.0)
    trx = Trx(flies=[fly0, fly1, fly2])
    closest_ids, mind = closest_fly(trx, 0)
    assert (closest_ids == 1).all()
    assert_close(mind, np.full(5, 1.0))


def test_nflies_close():
    fly0 = make_simple_fly(fly_id=0, vx=0.0)
    fly1 = make_simple_fly(fly_id=1, vx=0.0, x0=1.0)
    fly2 = make_simple_fly(fly_id=2, vx=0.0, x0=10.0)
    trx = Trx(flies=[fly0, fly1, fly2])
    n = nflies_close(trx, 0, threshold_mm=2.0)
    assert (n == 1).all()
    n2 = nflies_close(trx, 0, threshold_mm=20.0)
    assert (n2 == 2).all()


def run_all():
    tests = [
        test_modrange,
        test_central_diff,
        test_ecc_and_area,
        test_xnose_ynose,
        test_velmag_constant_velocity,
        test_phi_constant_motion,
        test_yaw,
        test_dtheta_constant_rotation,
        test_dtheta_wraparound,
        test_arena_features,
        test_wing_angles,
        test_pair_dcenter,
        test_pair_dcenter_no_overlap,
        test_pair_dcenter_partial_overlap,
        test_anglefrom1to2,
        test_anglesubtended_trivial,
        test_closest_fly_simple,
        test_nflies_close,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(run_all())
