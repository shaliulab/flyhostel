"""Local mock for testing the movement adapter without an xarray install.

This duck-types a minimal xarray-like object that satisfies the adapter's
interface: it needs .sizes, .coords, .attrs, .isel, indexing with ["pose_tracks"],
and ["confidence"], and ds.coords["keypoints"].values / ds.coords["individuals"].values.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


class _Coord:
    def __init__(self, values):
        self.values = np.asarray(values)


class _Var:
    def __init__(self, arr):
        self._arr = arr

    @property
    def values(self):
        return self._arr

    def isel(self, **kwargs):
        # only need isel(individuals=int)
        assert list(kwargs.keys()) == ["individuals"]
        idx = kwargs["individuals"]
        return _Var(self._arr[:, idx])

    def __setitem__(self, key, value):
        # support ds["pose_tracks"][:100, 0, 1, :] = np.nan
        self._arr[key] = value


class _MockDS:
    """Mimics enough of xarray.Dataset to drive the adapter."""

    def __init__(self, pose, confidence, fps, individuals, keypoints):
        self._data = {
            "pose_tracks": _Var(pose),
            "confidence": _Var(confidence),
        }
        self.coords = {
            "individuals": _Coord(individuals),
            "keypoints": _Coord(keypoints),
            "space": _Coord(["x", "y"]),
        }
        self.sizes = {
            "time": pose.shape[0],
            "individuals": pose.shape[1],
            "keypoints": pose.shape[2],
            "space": 2,
        }
        self.attrs = {"fps": fps}

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]


# Patch the adapter module to think xr is installed (we don't use any xr APIs
# inside it - only the duck-typed methods above).
import jaaba_features.movement_adapter as ma
ma.xr = object()  # sentinel so the ImportError check passes

from jaaba_features.movement_adapter import movement_to_trx  # noqa: E402
from jaaba_features.single_fly import velmag_ctr  # noqa: E402
from jaaba_features.pair_features import dcenter_pair, anglesub_pair  # noqa: E402
from jaaba_features.closest import closest_fly  # noqa: E402


def make_synthetic_ds(n_frames=1000, fps=47.0,
                      keypoints=("head", "thorax", "abdomen", "lW", "rW")):
    T, K, N = n_frames, len(keypoints), 2
    px_per_mm = 30.0
    t = np.arange(T)
    pose = np.full((T, N, K, 2), np.nan, dtype=float)
    confidence = np.full((T, N, K), 0.95, dtype=np.float32)

    body_half = 75.0 / 2.0
    wing_off = 30.0
    cx, cy = 15.0 * px_per_mm, 15.0 * px_per_mm
    r = 10.0 * px_per_mm
    omega = 2 * np.pi / T
    phi = omega * t
    thorax0 = np.column_stack([cx + r * np.cos(phi), cy + r * np.sin(phi)])
    heading0 = phi + np.pi / 2
    thorax1 = np.tile([cx, cy], (T, 1))
    heading1 = np.zeros(T)

    for fly_idx, (thorax, heading) in enumerate(
        [(thorax0, heading0), (thorax1, heading1)]
    ):
        head_off = body_half * np.column_stack([np.cos(heading), np.sin(heading)])
        head = thorax + head_off
        abdomen = thorax - head_off
        perp = np.column_stack([-np.sin(heading), np.cos(heading)])
        wing_l = thorax + wing_off * perp
        wing_r = thorax - wing_off * perp
        pose[:, fly_idx, 0, :] = head
        pose[:, fly_idx, 1, :] = thorax
        pose[:, fly_idx, 2, :] = abdomen
        pose[:, fly_idx, 3, :] = wing_l
        pose[:, fly_idx, 4, :] = wing_r

    return _MockDS(
        pose=pose, confidence=confidence, fps=fps,
        individuals=[f"fly_{i}" for i in range(N)],
        keypoints=list(keypoints),
    ), px_per_mm


def test_adapter_basic():
    ds, ppm = make_synthetic_ds(n_frames=1000)
    trx = movement_to_trx(ds, px_per_mm=ppm, arena_radius_mm=15.0,
                          arena_center_mm=(15.0, 15.0))
    assert len(trx.flies) == 2
    f0, f1 = trx[0], trx[1]
    assert f0.nframes == 1000
    assert f1.nframes == 1000
    np.testing.assert_allclose(f1.x_mm, 15.0)
    np.testing.assert_allclose(f1.y_mm, 15.0)
    r = np.hypot(f0.x_mm - 15.0, f0.y_mm - 15.0)
    np.testing.assert_allclose(r, 10.0, atol=1e-9)
    np.testing.assert_allclose(f0.a_mm, 1.25, atol=1e-9)
    print("PASS  test_adapter_basic")


def test_adapter_velocity():
    ds, ppm = make_synthetic_ds(n_frames=1000, fps=47.0)
    trx = movement_to_trx(ds, px_per_mm=ppm)
    f0 = trx[0]
    v, _ = velmag_ctr(f0)
    expected_speed = 2 * np.pi * 10.0 / (1000.0 / 47.0)
    avg_v = float(np.nanmean(v))
    assert abs(avg_v - expected_speed) < 0.01, f"got {avg_v}, expected {expected_speed}"
    print(f"PASS  test_adapter_velocity (mean speed = {avg_v:.3f} mm/s, expected {expected_speed:.3f})")


def test_adapter_pair_features():
    ds, ppm = make_synthetic_ds(n_frames=1000)
    trx = movement_to_trx(ds, px_per_mm=ppm)
    d = dcenter_pair(trx[0], trx[1])
    np.testing.assert_allclose(d, 10.0, atol=1e-9)
    print("PASS  test_adapter_pair_features")


def test_adapter_anglesub():
    ds, ppm = make_synthetic_ds(n_frames=1000)
    trx = movement_to_trx(ds, px_per_mm=ppm)
    asub = anglesub_pair(trx[0], trx[1], fov=np.deg2rad(270))
    # Most frames should give a positive subtended angle; cardinal-position
    # degeneracies (where tangent points coincide) can produce exact zeros.
    assert np.nanmean(asub) > 0
    assert np.all(asub >= 0) and np.all(asub <= np.deg2rad(270))
    print(f"PASS  test_adapter_anglesub (mean={np.nanmean(asub):.4f} rad, max={np.nanmax(asub):.4f} rad)")


def test_adapter_closest_fly():
    ds, ppm = make_synthetic_ds(n_frames=1000)
    trx = movement_to_trx(ds, px_per_mm=ppm)
    closest, mind = closest_fly(trx, 0)
    assert np.all(closest == 1)
    print("PASS  test_adapter_closest_fly")


def test_adapter_keypoint_aliases():
    ds, ppm = make_synthetic_ds(n_frames=100)
    trx = movement_to_trx(ds, px_per_mm=ppm)
    assert trx[0].wing_anglel is not None
    assert trx[0].wing_angler is not None
    print("PASS  test_adapter_keypoint_aliases")


def test_adapter_missing_frames():
    ds, ppm = make_synthetic_ds(n_frames=1000)
    ds["pose_tracks"][:100, 0, 1, :] = np.nan
    ds["pose_tracks"][-50:, 0, 1, :] = np.nan
    trx = movement_to_trx(ds, px_per_mm=ppm)
    f0 = trx[0]
    assert f0.firstframe == 100, f"got firstframe={f0.firstframe}"
    assert f0.endframe == 949, f"got endframe={f0.endframe}"
    assert f0.nframes == 850
    print(f"PASS  test_adapter_missing_frames (first={f0.firstframe}, last={f0.endframe})")


def test_adapter_partial_overlap_pair():
    ds, ppm = make_synthetic_ds(n_frames=1000)
    ds["pose_tracks"][500:, 0, :, :] = np.nan
    ds["pose_tracks"][:200, 1, :, :] = np.nan
    trx = movement_to_trx(ds, px_per_mm=ppm)
    f0, f1 = trx[0], trx[1]
    assert f0.firstframe == 0 and f0.endframe == 499
    assert f1.firstframe == 200 and f1.endframe == 999
    d = dcenter_pair(f0, f1)
    assert np.all(np.isnan(d[:200]))
    np.testing.assert_allclose(d[200:500], 10.0, atol=1e-9)
    print("PASS  test_adapter_partial_overlap_pair")


def run_all():
    tests = [
        test_adapter_basic,
        test_adapter_velocity,
        test_adapter_pair_features,
        test_adapter_anglesub,
        test_adapter_closest_fly,
        test_adapter_keypoint_aliases,
        test_adapter_missing_frames,
        test_adapter_partial_overlap_pair,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(run_all())
