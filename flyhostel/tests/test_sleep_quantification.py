"""
Tests for the quantification of sleep (flyhostel.data.sleep, flyhostel.quantification.sleep, flyhostel.data.pose.sleep).

Sleep is defined as >= 5 minutes (300 s) of continuous immobility. In the pose pipeline, behavior is predicted
at 30 fps and each second is immobile when more than `threshold` of its 30 frames are an inactive state.
"""
from collections import namedtuple

import numpy as np
import pandas as pd
import pytest

from flyhostel.data.sleep import (
    bin_apply_all,
    compute_latency,
    compute_time_asleep,
    sleep_annotation_rf,
    sleep_annotation_rf_all,
)

FPS = 30
FRAMERATE = 150
STEP = FRAMERATE // FPS


def behavior(seconds, id="fly|01", t0=0, inactive_frames_per_second=None):
    """
    30 fps behavior of a fly from a list of booleans, one per second (True = inactive).
    inactive_frames_per_second overrides how many of the 30 frames of each second are inactive
    """
    seconds = list(seconds)
    if inactive_frames_per_second is None:
        inactive_frames_per_second = [FPS if inactive else 0 for inactive in seconds]
    rows = []
    for second, n_inactive in enumerate(inactive_frames_per_second):
        for frame in range(FPS):
            rows.append((id, t0 + second + frame / FPS, (t0 + second) * FRAMERATE + frame * STEP, frame < n_inactive))
    return pd.DataFrame(rows, columns=["id", "t", "frame_number", "inactive_states"])


def asleep_seconds(data, **kwargs):
    return sleep_annotation_rf(data, **kwargs)["inactive_rule"].values


def test_sleep_requires_300_seconds_of_immobility():
    awake = [False] * 10
    assert not asleep_seconds(behavior(awake + [True] * 299 + awake)).any()

    out = asleep_seconds(behavior(awake + [True] * 300 + awake))
    assert out.tolist() == [False] * 10 + [True] * 300 + [False] * 10


def test_min_time_immobile_is_a_parameter():
    data = behavior([True] * 120 + [False] * 5)
    assert asleep_seconds(data, min_time_immobile=120)[:120].all()
    assert not asleep_seconds(data, min_time_immobile=121).any()


def test_one_second_of_activity_interrupts_sleep():
    out = asleep_seconds(behavior([True] * 250 + [False] + [True] * 250))
    assert not out.any()


@pytest.mark.parametrize("n_inactive,expected", [(10, False), (11, True), (30, True)])
def test_a_second_is_immobile_when_more_than_threshold_frames_are_inactive(n_inactive, expected):
    data = behavior([None] * 300, inactive_frames_per_second=[n_inactive] * 300)
    out = sleep_annotation_rf(data, threshold=10)
    assert (out["windowed_var"] == expected).all()
    assert (out["inactive_rule"] == expected).all()


def test_windows_carry_their_first_frame_number_and_time():
    out = sleep_annotation_rf(behavior([True] * 5, t0=100))
    assert out["t"].tolist() == [100, 101, 102, 103, 104]
    assert out["frame_number"].tolist() == [fn * FRAMERATE for fn in range(100, 105)]


def test_longer_windows():
    # 10 s windows: a window is immobile when > threshold of its 300 frames are inactive
    data = behavior([True] * 600 + [False] * 20)
    out = sleep_annotation_rf(data, time_window_length=10, threshold=100)
    assert out["t"].tolist() == list(range(0, 620, 10))
    assert out["inactive_rule"].tolist() == [True] * 60 + [False] * 2


def test_flies_are_annotated_independently():
    # fly 1 ends immobile and fly 2 starts immobile: their bouts must not be joined
    fly1 = behavior([False] * 10 + [True] * 200, id="fly|01")
    fly2 = behavior([True] * 200 + [False] * 10, id="fly|02")
    fly3 = behavior([True] * 400, id="fly|03")
    out = sleep_annotation_rf_all(pd.concat([fly1, fly2, fly3]))
    asleep = out.groupby("id")["inactive_rule"].sum()
    assert asleep.to_dict() == {"fly|01": 0, "fly|02": 0, "fly|03": 400}


def test_bin_apply_all_gives_the_fraction_of_time_asleep():
    data = sleep_annotation_rf(behavior([False] * 300 + [True] * 450 + [False] * 150))
    binned = bin_apply_all(data, feature="inactive_rule", summary_FUN="mean", x_bin_length=300)
    assert binned["t_round"].tolist() == [0, 300, 600]
    assert binned["inactive_rule"].tolist() == [0, 1, 0.5]


def sleep_trace(asleep, id="fly|01"):
    return pd.DataFrame({"id": id, "t": np.arange(len(asleep), dtype=float), "asleep": asleep})


def test_time_asleep_counts_seconds_since_falling_asleep():
    out = compute_time_asleep(sleep_trace([False, False, True, True, True, False, True]))
    assert out["time_asleep"].tolist() == [-1, -1, 1, 2, 3, -1, 1]
    assert out["duration_asleep"].tolist() == [2, 2, 3, 3, 3, 1, 1]


def test_latency_is_minus_one_while_asleep_and_nan_after_the_last_bout():
    out = compute_latency(sleep_trace([False] * 3 + [True] * 2 + [False] * 2))
    assert out.loc[out["asleep"], "latency"].tolist() == [-1, -1]
    assert out["latency"].iloc[-2:].isna().all()


@pytest.mark.xfail(strict=True, reason="sleep_onset is taken from the last awake row (diff==1 is located one row early), "
                                       "so latency is 1 window short")
def test_latency_counts_seconds_until_the_next_sleep_onset():
    out = compute_latency(sleep_trace([False] * 3 + [True] * 2 + [False] * 2))
    assert out["latency"].iloc[:3].tolist() == [3, 2, 1]


AnalysisParams = namedtuple("AnalysisParams", ["min_time_immobile", "time_window_length", "velocity_correction_coef", "max_brief_awakening"])


def test_legacy_sleep_annotation_from_velocity():
    from flyhostel.quantification.sleep import sleep_annotation
    params = AnalysisParams(min_time_immobile=300, time_window_length=10, velocity_correction_coef=0.2, max_brief_awakening=30)
    # 30 immobile windows of 10 s = 300 s, then 29 = 290 s
    velocity = [1] + [0.1] * 30 + [1] + [0.1] * 29 + [1]
    data = pd.DataFrame({"velocity": velocity, "t_round": np.arange(len(velocity)) * 10})
    out = sleep_annotation(data, params)
    assert out["asleep"].tolist() == [False] + [True] * 30 + [False] * 31
    assert out["t"].tolist() == data["t_round"].tolist()
    assert "t_round" not in out.columns


def test_ethoscopy_sleep_contiguous_follows_the_r_definition():
    from ethoscopy.analyse import sleep_contiguous
    moving = np.array([True] + [False] * 30 + [True] + [False] * 29)
    # fs = 0.1 Hz (10 s windows), 300 s -> 30 windows
    assert sleep_contiguous(moving, fs=0.1, min_valid_time=300) == [False] + [True] * 30 + [False] * 30


@pytest.mark.parametrize("min_valid_time,fs,n_immobile,expected", [
    (300, 1, 301, True),
    pytest.param(300, 1, 300, True, marks=pytest.mark.xfail(
        strict=True, reason="uses > instead of >=, unlike the R function it emulates")),
    pytest.param(300, 0.1, 30, True, marks=pytest.mark.xfail(
        strict=True, reason="compares the run length with min_valid_time instead of min_len = fs * min_valid_time")),
])
def test_pose_sleep_contiguous(min_valid_time, fs, n_immobile, expected):
    from flyhostel.data.pose.sleep import sleep_contiguous
    moving = np.array([True] + [False] * n_immobile + [True])
    out = sleep_contiguous(moving, min_valid_time=min_valid_time, fs=fs)
    assert out[1:-1].all() == expected
    assert not out[0] and not out[-1]


@pytest.mark.xfail(strict=True, raises=TypeError,
                   reason="sleep_annotation_inactive calls bin_behavior_table_v2 without its chunksize and framerate arguments")
def test_sleep_annotation_from_inactive_behaviors():
    from flyhostel.data.pose.sleep import sleep_annotation_inactive
    data = behavior([True] * 400)
    data["behavior"] = "inactive"
    sleep_annotation_inactive(data)


@pytest.mark.xfail(strict=True, reason="ethoscopy's flyhostel_sleep_annotation asserts framerate is None, then calls "
                                       "downsample_to_fps which raises NotImplementedError. "
                                       "FlyHostelLoader.sleep_annotation(source='centroids') depends on it")
def test_sleep_annotation_from_centroids():
    from flyhostel.data.pose.loaders.centroids import flyhostel_sleep_annotation_primitive
    t = np.arange(0, 1000, 1 / 2)
    data = pd.DataFrame({"t": t, "x": 0.5, "y": 0.5, "frame_number": (t * FRAMERATE).astype(int)})
    flyhostel_sleep_annotation_primitive(data, framerate=FRAMERATE, chunksize=45000)
