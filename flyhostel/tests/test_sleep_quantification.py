"""
Tests for the quantification of sleep (flyhostel.data.sleep), the only definition of sleep in flyhostel.

Sleep is defined as >= 5 minutes (300 s) of continuous immobility. Behavior is predicted at 30 fps (47 in some
experiments) and each second is immobile when more than 1/3 of its frames are an inactive state.
Walk, feed, groom and background are active, the rest is inactive.
sleep_from_centroid_velocity is the velocity (centroid) based alternative, for data without behavior.
"""
import os

import numpy as np
import pandas as pd
import pytest

import flyhostel.data.sleep as sleep_module
from flyhostel.data.sleep import (
    ACTIVE_STATES,
    INACTIVE_STATES,
    annotate_sleep,
    annotate_sleep_from_centroids,
    bin_sleep,
    compute_sleep_file,
    compute_time_asleep,
    mask_unreliable_frames,
    sleep_from_centroid_velocity,
    sleep_path,
    sleep_inactive_states,
)

FPS = 30
FRAMERATE = 150
STEP = FRAMERATE // FPS


def behavior(seconds, id="fly|01", t0=0, inactive_frames_per_second=None, active="walk", inactive="inactive", fps=FPS):
    """
    Behavior of a fly (at fps, default 30) from a list of booleans, one per second (True = inactive).
    inactive_frames_per_second overrides how many of the frames of each second are inactive
    """
    seconds = list(seconds)
    if inactive_frames_per_second is None:
        inactive_frames_per_second = [fps if is_inactive else 0 for is_inactive in seconds]
    rows = []
    for second, n_inactive in enumerate(inactive_frames_per_second):
        for frame in range(fps):
            rows.append((id, t0 + second + frame / fps, (t0 + second) * FRAMERATE + frame * STEP, inactive if frame < n_inactive else active))
    return pd.DataFrame(rows, columns=["id", "t", "frame_number", "prediction2"])


def asleep_seconds(data, **kwargs):
    return annotate_sleep(data, **kwargs)["inactive_rule"].values


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


@pytest.mark.parametrize("fps,n_inactive,expected", [
    (30, 10, False), (30, 11, True), (30, 30, True),
    (47, 15, False), (47, 16, True),
])
def test_a_second_is_immobile_when_more_than_a_third_of_its_frames_are_inactive(fps, n_inactive, expected):
    data = behavior([None] * 300, inactive_frames_per_second=[n_inactive] * 300, fps=fps)
    out = annotate_sleep(data)
    assert (out["windowed_var"] == expected).all()
    assert (out["inactive_rule"] == expected).all()


def test_seconds_with_missing_frames_are_judged_against_the_usual_frames_per_second():
    # a second with only 20 of its 30 frames still needs > 10 inactive frames
    data = behavior([True] * 300)
    second = data["t"] // 1 == 100
    data = data.loc[~second | (data.index % 30 < 20)]
    data.loc[second, "prediction2"] = ["inactive"] * 10 + ["walk"] * 10
    out = annotate_sleep(data)
    assert not out.loc[out["t"] == 100, "windowed_var"].item()


@pytest.mark.parametrize("state", ACTIVE_STATES)
def test_active_states_break_sleep(state):
    out = asleep_seconds(behavior([True] * 250 + [False] + [True] * 250, active=state))
    assert not out.any()


@pytest.mark.parametrize("state", INACTIVE_STATES)
def test_inactive_states_make_sleep(state):
    assert asleep_seconds(behavior([True] * 300, inactive=state)).all()


def test_unknown_behaviors_count_as_active():
    assert not asleep_seconds(behavior([True] * 250 + [False] + [True] * 250, active="twitch")).any()


def test_windows_carry_their_first_frame_number_and_time():
    out = annotate_sleep(behavior([True] * 5, t0=100))
    assert out["t"].tolist() == [100, 101, 102, 103, 104]
    assert out["frame_number"].tolist() == [fn * FRAMERATE for fn in range(100, 105)]


def test_longer_windows():
    # 10 s windows: a window is immobile when > threshold of its 300 frames are inactive
    data = behavior([True] * 600 + [False] * 20)
    out = annotate_sleep(data, time_window_length=10)
    assert out["t"].tolist() == list(range(0, 620, 10))
    assert out["inactive_rule"].tolist() == [True] * 60 + [False] * 2


def test_flies_are_annotated_independently():
    # fly 1 ends immobile and fly 2 starts immobile: their bouts must not be joined
    fly1 = behavior([False] * 10 + [True] * 200, id="fly|01")
    fly2 = behavior([True] * 200 + [False] * 10, id="fly|02")
    fly3 = behavior([True] * 400, id="fly|03")
    out = annotate_sleep(pd.concat([fly1, fly2, fly3]))
    asleep = out.groupby("id")["inactive_rule"].sum()
    assert asleep.to_dict() == {"fly|01": 0, "fly|02": 0, "fly|03": 400}


def test_bin_sleep_gives_the_fraction_of_time_asleep():
    data = annotate_sleep(behavior([False] * 300 + [True] * 450 + [False] * 150))
    binned = bin_sleep(data, bin_size=300)
    assert binned["t_round"].tolist() == [0, 300, 600]
    assert binned["inactive_rule"].tolist() == [0, 1, 0.5]
    assert binned["frame_number"].tolist() == [0, 300 * FRAMERATE, 600 * FRAMERATE]


def test_unreliable_seconds_take_the_behavior_from_the_past(tmp_path):
    data = behavior([True] * 10 + [False] + [True] * 10)
    os.makedirs(tmp_path / "flyhostel" / "validation")
    # a mistake in second 10 makes seconds 9, 10 and 11 unreliable
    pd.DataFrame({"frame_number": [10 * FRAMERATE]}).to_csv(tmp_path / "flyhostel" / "validation" / "in_frame_index_mistakes.csv")
    out = mask_unreliable_frames(data, str(tmp_path))
    assert (out["prediction2"] == "inactive").all()
    assert (mask_unreliable_frames(data, str(tmp_path / "other"))["prediction2"] == data["prediction2"]).all()


def test_sleep_files(tmp_path):
    animal = "FlyHostel1_1X_2024-01-01_10-00-00__00"
    assert sleep_path("/basedir", animal) == f"/basedir/sleep/00/{animal}_sleep5min.feather"
    assert sleep_path("/basedir", animal, bin_size=300) == f"/basedir/sleep/00/{animal}_sleep5min_bin300.feather"

    basedir = str(tmp_path)
    os.makedirs(tmp_path / "motionmapper" / "00")
    behavior([False] * 10 + [True] * 600).to_feather(tmp_path / "motionmapper" / "00" / f"{animal}.feather")
    paths = compute_sleep_file(basedir, animal)
    assert paths == [sleep_path(basedir, animal), sleep_path(basedir, animal, bin_size=300)]
    assert pd.read_feather(paths[0])["inactive_rule"].sum() == 600
    assert pd.read_feather(paths[1])["inactive_rule"].tolist() == [290 / 300, 1, 1]


def test_feed_is_inactive_in_arenas_but_walking_still_interrupts_sleep(tmp_path):
    arena = "FlyHostel1_4X_2026-10-13_17-00-00_arena4__00"
    assert "feed" in sleep_inactive_states(arena)
    assert "feed" not in sleep_inactive_states("FlyHostel1_1X_2024-01-01_10-00-00__00")

    os.makedirs(tmp_path / "motionmapper" / "00")
    # 1 s of feed in the middle of 600 immobile seconds, then 1 s of walk
    data = behavior([True] * 300 + [False] + [True] * 300, active="feed")
    data = pd.concat([data, behavior([False] + [True] * 10, t0=601)])
    for animal, asleep in [(arena, 601), ("FlyHostel1_1X_2024-01-01_10-00-00__00", 600)]:
        data.reset_index(drop=True).to_feather(tmp_path / "motionmapper" / "00" / f"{animal}.feather")
        out = pd.read_feather(compute_sleep_file(str(tmp_path), animal)[0])
        assert out["inactive_rule"].sum() == asleep
        assert not out["inactive_rule"].iloc[601:].any()


def centroids(positions, id="fly|01", t0=0, fps=47):
    """
    Centroid of a fly (at fps, default 47 as in the arenas) from a list of (x, y) in mm, one per second
    """
    rows = []
    for second, (x, y) in enumerate(positions):
        for frame in range(fps):
            rows.append((id, t0 + second + frame / fps, (t0 + second) * fps + frame, x, y))
    return pd.DataFrame(rows, columns=["id", "t", "frame_number", "x_mm", "y_mm"])


def centroid_asleep(data, **kwargs):
    return annotate_sleep_from_centroids(data, **kwargs)["inactive_rule"].values


def test_centroid_sleep_requires_300_seconds_without_moving_more_than_the_threshold():
    # 0.8 mm per second is below 1 mm: immobile. The first second has no previous one and is not
    still = [(0.8 * (i % 2), 0) for i in range(301)]
    out = annotate_sleep_from_centroids(centroids(still))
    assert out["displacement_mm"].iloc[1:].round(6).eq(0.8).all()
    assert out["inactive_rule"].tolist() == [False] + [True] * 300
    assert not centroid_asleep(centroids(still[:300])).any()
    assert not centroid_asleep(centroids(still), min_displacement_mm=0.5).any()


def test_one_second_of_centroid_displacement_interrupts_sleep():
    positions = [(0, 0)] * 251 + [(2, 0)] * 251
    assert not centroid_asleep(centroids(positions)).any()
    assert centroid_asleep(centroids(positions), min_time_immobile=250)[1:].sum() == 500


def test_centroid_sleep_uses_the_median_position_of_each_second():
    # a few frames with a wrong centroid (e.g. a reflection) do not move the fly
    data = centroids([(0, 0)] * 400)
    data.loc[data.index % 47 < 5, "x_mm"] = 10
    assert centroid_asleep(data)[1:].all()


def test_seconds_without_centroid_are_skipped_and_displacement_is_measured_across_them():
    data = centroids([(0, 0)] * 200 + [(5, 5)] * 10 + [(0, 0)] * 200)
    data.loc[data["t"].between(200, 210, inclusive="left"), ["x_mm", "y_mm"]] = np.nan
    out = annotate_sleep_from_centroids(data)
    assert len(out) == 400 and out["inactive_rule"].iloc[1:].all()

    # if the fly reappears somewhere else, it moved during the gap
    data = centroids([(0, 0)] * 200 + [(5, 5)] * 10 + [(5, 5)] * 200)
    data.loc[data["t"].between(200, 210, inclusive="left"), ["x_mm", "y_mm"]] = np.nan
    assert not centroid_asleep(data).any()


def test_centroid_sleep_of_flies_is_computed_per_fly():
    # fly 2 starts where fly 1 ends: its first second is not a displacement of fly 1
    fly1 = centroids([(0, 0)] * 350, id="fly|01")
    fly2 = centroids([(20, 20)] * 350, id="fly|02")
    out = annotate_sleep_from_centroids(pd.concat([fly1, fly2]))
    assert out.groupby("id")["inactive_rule"].sum().to_dict() == {"fly|01": 349, "fly|02": 349}
    assert out.groupby("id")["n_frames"].first().tolist() == [47, 47]


def test_centroid_sleep_files(tmp_path, monkeypatch):
    animal = "FlyHostel1_4X_2026-10-13_17-00-00_arena4__00"
    assert sleep_path("/basedir", animal, method="centroid") == f"/basedir/sleep/00/{animal}_sleep5min_centroid.feather"
    assert sleep_path("/basedir", animal, bin_size=300, method="centroid") == f"/basedir/sleep/00/{animal}_sleep5min_centroid_bin300.feather"
    with pytest.raises(ValueError):
        sleep_path("/basedir", animal, method="other")

    # moving 2 mm per second during 10 s, then still
    data = centroids([(2 * i, 0) for i in range(10)] + [(18, 0)] * 600)
    monkeypatch.setattr(sleep_module, "load_centroids", lambda basedir, animal: data)
    paths = compute_sleep_file(str(tmp_path), animal, method="centroid")
    assert paths == [sleep_path(str(tmp_path), animal, method="centroid"), sleep_path(str(tmp_path), animal, bin_size=300, method="centroid")]
    assert pd.read_feather(paths[0])["inactive_rule"].sum() == 600
    assert pd.read_feather(paths[1])["inactive_rule"].tolist() == [290 / 300, 1, 10 / 10]
    with pytest.raises(ValueError):
        compute_sleep_file(str(tmp_path), animal, input="behavior.feather", method="centroid")


def test_velocity_sleep():
    # 10 Hz: moving above 1, sleep = >= 30 s (300 samples) without moving
    speed = np.array([2.0] * 5 + [0.0] * 300 + [2.0] * 5 + [0.0] * 299 + [2.0])
    moving, asleep = sleep_from_centroid_velocity(speed, fps=10, speed_threshold=1, min_time_immobile=30)
    assert asleep.tolist() == [False] * 5 + [True] * 300 + [False] * 305
    assert (moving == (speed > 1)).all()
    _, asleep = sleep_from_centroid_velocity(speed, fps=10, speed_threshold=1, min_time_immobile=31)
    assert not asleep.any()


def test_velocity_sleep_ignores_brief_movements_and_missing_speed_is_moving():
    # 0.2 s of movement is shorter than min_time_moving (0.5 s): the fly sleeps through it
    speed = np.array([0.0] * 200 + [2.0] * 2 + [0.0] * 200)
    moving, asleep = sleep_from_centroid_velocity(speed, fps=10, speed_threshold=1, min_time_immobile=30, min_time_moving=0.5)
    assert asleep.all() and not moving.any()
    _, asleep = sleep_from_centroid_velocity(speed, fps=10, speed_threshold=1, min_time_immobile=30)
    assert not asleep[200:202].any()

    # 1 s without tracking breaks the immobility bout
    speed = np.array([0.0] * 200 + [np.nan] * 10 + [0.0] * 200)
    moving, asleep = sleep_from_centroid_velocity(speed, fps=10, speed_threshold=1, min_time_immobile=30, min_time_moving=0.5)
    assert moving[200:210].all() and not asleep[200:210].any()


def test_velocity_sleep_of_several_flies_is_computed_per_fly():
    speed = np.zeros((400, 2))
    speed[350, 0] = 5
    _, asleep = sleep_from_centroid_velocity(speed, fps=1, speed_threshold=1, min_time_immobile=300)
    assert asleep[:, 0].sum() == 350 and asleep[:, 1].all()


def test_simulated_flies_keep_their_own_timeseries():
    from flyhostel.modelling.analysis import SleepAnalyser

    class Model:
        # fly 0 always quiescent (asleep), fly 1 never
        timeseries = np.stack([np.ones(100), np.zeros(100)], axis=1)

    out = SleepAnalyser(min_time_immobile=300, time_window_length=10).summarise(Model())
    assert out["asleep"].tolist() == [1, 0]
    assert out["moving"].tolist() == [0, 1]


def sleep_trace(asleep, id="fly|01"):
    return pd.DataFrame({"id": id, "t": np.arange(len(asleep), dtype=float), "asleep": asleep})


def test_time_asleep_counts_seconds_since_falling_asleep():
    out = compute_time_asleep(sleep_trace([False, False, True, True, True, False, True]))
    assert out["time_asleep"].tolist() == [-1, -1, 1, 2, 3, -1, 1]
    assert out["duration_asleep"].tolist() == [2, 2, 3, 3, 3, 1, 1]


def test_ethoscopy_sleep_contiguous_follows_the_r_definition():
    from ethoscopy.analyse import sleep_contiguous
    moving = np.array([True] + [False] * 30 + [True] + [False] * 29)
    # fs = 0.1 Hz (10 s windows), 300 s -> 30 windows
    assert sleep_contiguous(moving, fs=0.1, min_valid_time=300) == [False] + [True] * 30 + [False] * 30
