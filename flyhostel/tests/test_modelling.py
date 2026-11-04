"""Tests of the agent based model of group sleep (flyhostel.modelling.modelling)"""
import random

import numpy as np
import pandas as pd
import pytest

from flyhostel.data.interactions.sociability.behavior_integration import annotate_behavior_database_id
from flyhostel.modelling.modelling.compile.interactions_table import classify, interaction_rates, seconds_by_moving
from flyhostel.modelling.modelling.fly_agent_models.fly_agent import ArousalTable, FlyAgent, RealBouts
from flyhostel.modelling.modelling.fly_agent_models.multi_fly_simulator import MultiFlySimulator
from flyhostel.modelling.modelling.run import simulate
from flyhostel.modelling.modelling.single_fly import bouts_of
from flyhostel.modelling.modelling.utils.fly_traces_handlers import sleep_of_activity


class Fixed:
    def __init__(self, length):
        self.length = length

    def sample(self, size=1):
        return self.length


BOUTS = pd.DataFrame({
    "moving": [True] * 3 + [False] * 3,
    "length": [30, 90, 300, 10, 60, 3000],
})


def asleep_fly(arousal):
    """A fly in the second 1 of an inactive bout that does not end by itself"""
    fly = FlyAgent(0, Fixed(1), Fixed(10**9), arousal)
    fly.step()
    assert not fly.is_active
    return fly


def test_arousal_table_is_a_fraction_of_the_closest_window(tmp_path):
    path = tmp_path / "arousal.csv"
    pd.DataFrame({"window": [1, 10, 100], "y": [0.5, 0.4, 0.2]}).to_csv(path)
    table = ArousalTable(path)
    assert table(60) == 0.5
    assert table(11 * 60) == 0.4
    assert table(10**6) == 0.2
    pd.DataFrame({"window": [1], "y": [50]}).to_csv(path)
    with pytest.raises(ValueError):
        ArousalTable(path)


def test_an_interaction_wakes_up_with_the_probability_of_the_table():
    random.seed(0)
    woken = 0
    for _ in range(5000):
        fly = asleep_fly(lambda seconds: 0.3)
        fly.inform_interaction()
        fly.step()
        woken += fly.is_active
    assert 0.27 < woken / 5000 < 0.33


def test_an_interaction_is_rolled_only_once():
    # a fly that does not respond at first stays asleep, however long: the interaction is over
    random.seed(0)
    for _ in range(200):
        fly = asleep_fly(lambda seconds: 0.0 if seconds < 3 else 1.0)
        fly.inform_interaction()
        for _ in range(100):
            fly.step()
        assert not fly.is_active


def test_no_interactions_no_waking():
    fly = asleep_fly(lambda seconds: 1.0)
    for _ in range(1000):
        fly.step()
    assert not fly.is_active


def test_short_bouts_are_not_sleep():
    # 1 s inactive, then awake, then 200 s inactive: no run of 300 s, even if they span more than 300 s. Then a real bout
    activity = np.array([False] + [True] * 150 + [False] * 200 + [True] * 10 + [False] * 300 + [True])[:, None]
    asleep = sleep_of_activity(activity)
    assert asleep[:351].sum() == 0
    assert asleep[361:661].all() and asleep.sum() == 300


def test_bouts_are_resampled_from_the_real_ones():
    np.random.seed(0)
    bouts = RealBouts([5, 500])
    assert {bouts.sample() for _ in range(100)} == {5, 500}


def test_bouts_leave_out_the_ones_cut_by_the_window():
    moving = [True] * 3 + [False] * 5 + [True] * 2 + [False] * 4 + [True] * 7
    active, inactive = bouts_of(moving)
    assert active.tolist() == [2]
    assert inactive.tolist() == [5, 4]


def rates(ai=0.0, aa=0.0):
    return pd.DataFrame({
        "number_of_animals": [2, 2], "n_moving": [1, 2], "interaction_type": ["ai", "aa"], "frequency": [ai, aa],
    })


def test_simulations_are_reproducible():
    arguments = (2, 1, 0, BOUTS, rates(0.01, 0.01), lambda seconds: 0.5)
    assert simulate(*arguments) == simulate(*arguments)


def test_interactions_only_wake_up_inactive_flies():
    random.seed(0)
    np.random.seed(0)
    awake = MultiFlySimulator(BOUTS, 2, rates(ai=0.0), lambda seconds: 1.0).run(36000).mean()
    disturbed = MultiFlySimulator(BOUTS, 2, rates(ai=0.2), lambda seconds: 1.0).run(36000).mean()
    assert disturbed > awake


def interactions_table():
    rows = []
    for t, id1_immobile, id2_immobile in [(100, False, False), (101, True, False), (102, True, True)]:
        for a, b, immobile in (("x|1", "x|2", id1_immobile), ("x|2", "x|1", id2_immobile)):
            rows.append(dict(id=a, nn=b, frame_number=t * 150, first_t=float(t), experiment="x", pre_longImmobile=immobile))
    return pd.DataFrame(rows)


def test_interactions_are_counted_once_and_by_the_state_of_both_flies():
    assert classify(interactions_table())["interaction_type"].tolist() == ["aa", "ai", "ii"]


def test_rates_are_events_over_all_the_seconds_of_the_state():
    sleep = pd.DataFrame({
        "experiment": "x", "number_of_animals": 2, "t": np.repeat(np.arange(100, 110), 2), "moving": np.tile([True, False], 10),
    })
    moving = seconds_by_moving(sleep)
    out = interaction_rates(interactions_table(), moving)
    assert out["seconds"].eq(10).all() and out["n_events"].sum() == 3
    assert out["frequency"].tolist() == [0.1, 0.1, 0.1]


def test_interactions_keep_their_behavior_when_the_frames_are_on_other_grids():
    behavior = pd.DataFrame({"frame_number": np.arange(0, 1000, 5)})
    # interactions every 3 frames: only 1 in 5 falls exactly on the frames of the behavior
    interactions = pd.DataFrame({"frame_number_start": np.arange(0, 300, 3), "frame_number_end": np.arange(0, 300, 3) + 30})
    out = annotate_behavior_database_id(interactions, behavior)
    assert out.shape[0] == interactions.shape[0]
    assert (np.abs(behavior["frame_number"].to_numpy()[out["row_id_start"]] - out["frame_number_start"]) <= 2).all()
    # beyond the behavior there is nothing to annotate
    far = pd.DataFrame({"frame_number_start": [5000], "frame_number_end": [5030]})
    assert annotate_behavior_database_id(far, behavior).shape[0] == 0
