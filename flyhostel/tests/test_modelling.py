"""Tests of the agent based model of group sleep (flyhostel.modelling.modelling)"""
import itertools
import random

import numpy as np
import pandas as pd
import pytest

from flyhostel.data.interactions.sociability.behavior_integration import annotate_behavior_database_id
from flyhostel.modelling.modelling.compile.interactions_table import classify, contact_events, contact_rates, contacts_of_group, interaction_rates, seconds_by_moving
from flyhostel.modelling.modelling.fly_agent_models.fly_agent import ArousalTable, ContactRates, FlyAgent, RecoveringFly
from flyhostel.modelling.modelling.fly_agent_models.multi_fly_simulator import MultiFlySimulator
from flyhostel.modelling.modelling.run import simulate
from flyhostel.modelling.modelling.single_fly import bouts_of, spontaneous_moving
from flyhostel.modelling.modelling.utils.fly_traces_handlers import sleep_of_activity
from flyhostel.modelling.modelling.variants import CONSTANT_AROUSAL, VARIANTS, constant_arousal, contact_rates_from_pairs, contact_rates_independent_of_moving, flat_contact_rates


@pytest.mark.parametrize("probability,woken", [(1.0, True), (0.0, False)])
def test_constant_arousal_wakes_always_or_never_at_any_time_inactive(probability, woken):
    for seconds_inactive in (10, 100000):
        fly = asleep_fly(constant_arousal(probability))
        fly.time_in_bout = seconds_inactive
        fly.inform_interaction()
        fly.step()
        assert fly.is_active == woken


def test_every_level_of_constant_arousal_is_a_variant():
    assert sorted(CONSTANT_AROUSAL.values()) == [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1]
    assert all(variant in VARIANTS for variant in CONSTANT_AROUSAL)


NIGHT = pd.DataFrame({"moving": [True, False] * 3, "length": [30, 300, 90, 10, 60, 3000]})


def asleep_fly(arousal):
    """A fly in the second 1 of an inactive bout that does not end by itself"""
    fly = FlyAgent(0, pd.DataFrame({"moving": [True, False], "length": [1, 10**9]}), arousal)
    fly.step()
    assert not fly.is_active
    return fly


def test_arousal_table_is_a_fraction_of_the_closest_window_in_seconds(tmp_path):
    path = tmp_path / "arousal.csv"
    pd.DataFrame({"window": [1, 300, 2700], "y": [0.5, 0.4, 0.2]}).to_csv(path)
    table = ArousalTable(path)
    assert table(2) == 0.5
    assert table(5 * 60) == 0.4
    assert table(45 * 60) == 0.2
    pd.DataFrame({"window": [1], "y": [50]}).to_csv(path)
    with pytest.raises(ValueError):
        ArousalTable(path)


def test_a_fly_without_interactions_does_exactly_what_the_real_fly_did():
    rng = np.random.default_rng(0)
    real = np.repeat(rng.integers(0, 2, 40).astype(bool), rng.integers(1, 1200, 40))[:7200]
    fly = FlyAgent(0, bouts_of(real), lambda seconds: 1.0)
    assert np.array_equal([fly.step() for _ in range(len(real))], real)


def test_a_woken_fly_goes_on_with_its_next_active_bout():
    fly = FlyAgent(0, NIGHT.assign(length=[1, 1000, 7, 1000, 7, 1000]), lambda seconds: 1.0)
    fly.step()
    fly.step()
    fly.inform_interaction()
    assert not fly.step()
    assert fly.is_active and fly.bout == 2


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


def test_bouts_are_all_the_bouts_in_order():
    out = bouts_of([True] * 3 + [False] * 5 + [True] * 2)
    assert out["moving"].tolist() == [True, False, True]
    assert out["length"].tolist() == [3, 5, 2]


def rates(ai=0.0, aa=0.0):
    return pd.DataFrame({
        "number_of_animals": [2, 2], "n_moving": [1, 2], "interaction_type": ["ai", "aa"], "frequency": [ai, aa],
    })


def half(seconds):
    return 0.5


def contact_table(rate):
    return pd.DataFrame({"number_of_animals": 2, "n_moving": 1, "inactive_bin": range(7), "frequency": rate})


def test_simulations_are_reproducible():
    arguments = (2, 1, 0, [NIGHT, NIGHT.iloc[1:], NIGHT], rates(aa=0.01), contact_table(0.01), half)
    assert simulate(*arguments) == simulate(*arguments)


def test_contacts_only_wake_up_inactive_flies():
    random.seed(0)
    np.random.seed(0)
    awake = MultiFlySimulator([NIGHT, NIGHT.iloc[1:]], rates(), contact_table(0.0), lambda seconds: 1.0).run(36000).mean()
    disturbed = MultiFlySimulator([NIGHT, NIGHT.iloc[1:]], rates(), contact_table(0.2), lambda seconds: 1.0).run(36000).mean()
    assert disturbed > awake


def test_a_contact_depends_on_the_time_inactive():
    table = contact_table(0.0)
    table.loc[table["inactive_bin"] == 3, "frequency"] = 0.5  # 600-1200 s
    contact = ContactRates(table, 2)
    assert contact(1, 5) == 0 and contact(1, 700) == 0.5 and contact(1, 2000) == 0 and contact(2, 700) == 0


def test_arousal_leaves_out_flies_that_would_have_moved_anyway(tmp_path):
    path = tmp_path / "arousal.csv"
    pd.DataFrame({"window": [1, 300], "y": [0.5, 0.25]}).to_csv(path)
    table = ArousalTable(path, spontaneous=lambda seconds: np.array([0.6, 0.0625]))
    assert table(1) == 0
    assert table(300) == pytest.approx(0.2)


def test_spontaneous_moving_is_the_hazard_without_cut_bouts():
    # the first and the last bout, inactive and very long or short, are cut by the window. The 30 others last 100 s
    night = pd.DataFrame({"id": "a", "moving": [False] + [True, False] * 30 + [True, False], "length": [999] + [10, 100] * 30 + [10, 5]})
    assert spontaneous_moving(night)(np.array([10, 90])).tolist() == [0.0, 1.0]


def test_contacts_are_per_inactive_fly_and_time_inactive():
    sleep = pd.DataFrame({
        "id": np.repeat(["a", "b"], 100), "t": np.tile(np.arange(100), 2),
        "moving": np.r_[np.zeros(100, dtype=bool), np.ones(100, dtype=bool)],
        "number_of_animals": 2,
    })
    interactions = pd.DataFrame({
        "id": ["a", "b", "a", "b"], "nn": ["b", "a", "b", "a"], "frame_number": [1, 1, 2, 2], "first_t": [70.0, 70.0, 80.0, 80.0],
        "pre_longImmobile": [True, False, True, False],
    })
    out = contacts_of_group(sleep, interactions).query("n_moving == 1 and id == 'a'").set_index("inactive_bin")
    assert out.loc[1, "n_events"] == 2  # 60-300 s
    assert out.loc[0, "seconds"] == 50 and out.loc[1, "seconds"] == 41
    assert contact_rates(out.reset_index()).set_index("inactive_bin").loc[1, "frequency"] == pytest.approx(2 / 41)


def test_contacts_carry_the_outcome_of_the_fly_and_make_the_counts_of_the_rates():
    # a is still up to 90 s, moves 90-94, still again; b always moves
    moving_a = np.zeros(200, dtype=bool)
    moving_a[90:95] = True
    sleep = pd.DataFrame({
        "id": np.repeat(["a", "b"], 200), "t": np.tile(np.arange(200), 2), "moving": np.r_[moving_a, np.ones(200, dtype=bool)],
        "number_of_animals": 2,
    })
    # contacts at 5 s (inactive < 10 s: not a contact), 72-75 s (moves 15 s after the end), 110-112 s (still for good), 190 s (trace ends)
    starts, ends = [5.0, 72.0, 110.0, 190.0], [6.0, 75.0, 112.0, 190.5]
    interactions = pd.DataFrame({
        "id": ["a", "b"] * 4, "nn": ["b", "a"] * 4, "frame_number": np.repeat(np.arange(4), 2),
        "first_t": np.repeat(starts, 2), "last_t": np.repeat(ends, 2), "pre_longImmobile": [True, False] * 4,
    })
    events = contact_events(sleep, interactions)
    assert events["first_t"].tolist() == [72.0, 110.0, 190.0]
    assert events["seconds_inactive"].tolist() == [73, 16, 96]
    assert events["still_after_s"].tolist() == [15, 20, 10]
    assert events["responder"].tolist()[:2] == [1.0, 0.0] and np.isnan(events["responder"].iloc[2])
    counts = contacts_of_group(sleep, interactions, events=events)
    assert counts["n_events"].sum() == 3


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


def rates_table():
    return pd.DataFrame({
        "number_of_animals": [2, 2, 3, 3, 3, 3], "n_moving": [1, 1, 1, 1, 2, 2], "inactive_bin": [0, 1, 0, 1, 0, 1],
        "n_events": [10, 10, 30, 10, 40, 0], "seconds": [100, 300, 100, 300, 100, 100], "frequency": [.1, .03, .3, .03, .4, 0.],
    })


def test_flat_contacts_keep_the_contacts_of_a_state_at_every_time_inactive():
    flat=flat_contact_rates(rates_table())
    assert flat.groupby(["number_of_animals", "n_moving"]).frequency.nunique().eq(1).all()
    assert flat.query("number_of_animals==2").frequency.iloc[0] == pytest.approx(19 / 400)
    assert flat.query("number_of_animals==3 and n_moving==2").frequency.iloc[0] == pytest.approx(.2)


def test_contacts_from_pairs_add_up_with_the_flies_moving():
    table = contact_rates_from_pairs(rates_table()).query("number_of_animals==3 and n_moving==2").set_index("inactive_bin").frequency
    assert table[0] == pytest.approx(.2)
    assert table[1] == pytest.approx(.06)


def test_random_contacts_happen_with_no_fly_active():
    any_moving = pd.DataFrame({"number_of_animals": [2, 2], "inactive_bin": [0, 1], "frequency": [1.0, 1.0]})
    table = contact_rates_independent_of_moving(any_moving, rates_table().query("number_of_animals==2"))
    assert set(table.n_moving) == {0, 1}
    night = pd.DataFrame({"moving": [False, True], "length": [50, 50]})
    empty = pd.DataFrame(columns=["number_of_animals", "interaction_type", "n_moving", "frequency"])
    for need, contacted in ((True, False), (False, True)):
        model = MultiFlySimulator([night, night], empty, table, lambda seconds: 0.0, contacts_need_active=need)
        model.run(10)
        assert (model.contact_count > 0) == contacted


def test_flat_arousal_is_the_mean_over_the_interactions(tmp_path):
    path = tmp_path / "arousal.csv"
    pd.DataFrame({"window": [1, 300], "y": [0.5, 0.1], "n": [3, 1]}).to_csv(path)
    flat = ArousalTable(path).flat()
    assert flat(1) == flat(10000) == pytest.approx(.4)


def test_a_recovering_fly_makes_up_what_a_contact_interrupted():
    fly = RecoveringFly(0, pd.DataFrame({"moving": [False, True, False], "length": [100, 5, 50]}), lambda seconds: 1.0)
    for _ in range(10):
        fly.step()
    fly.inform_interaction()
    fly.step()
    assert fly.is_active and fly.lengths[2] == 50 + 89
    assert fly.contact_wakes == 1


def test_contacts_of_a_group_are_per_fly_and_bins_can_change():
    sleep = pd.DataFrame({
        "t": list(range(100)) * 2, "id": ["a"] * 100 + ["b"] * 100, "moving": [False] * 100 + [True] * 100, "number_of_animals": 2,
    })
    interactions = pd.DataFrame({
        "id": ["a", "b"], "nn": ["b", "a"], "frame_number": [1, 1], "pre_longImmobile": [True, False], "first_t": [50, 50],
    })
    table = contacts_of_group(sleep, interactions, bins=[10, 60])
    a = table.query("id=='a' and n_moving==1").set_index("inactive_bin")
    assert a.n_events.to_dict() == {0: 1, 1: 0}
    assert a.seconds.to_dict() == {0: 50, 1: 41}


# the spatial model: flies walk in the dish and interact when they come within the contact distance
from flyhostel.modelling.modelling.compile.walk import SPEED_CLASSES, START_CLASS, entries, speed_class, still_of, state_of, steps_of, to_rest_of
from flyhostel.modelling.modelling.fly_agent_models.spatial_simulator import PACE_FEATURES, ZT_FEATURES, PausedWalk, PositionedWalk, pauses_of, survival_of, RandomWalk, SpatialSimulator, segment_distance


def straight_walk(step_mm=3.0):
    """A walk of steps of the same length that never turns"""
    return RandomWalk(pd.DataFrame({"step_mm": [step_mm] * 10, "turn_rad": [0.0] * 10, "speed_class": [int(speed_class(step_mm))] * 10}), SPEED_CLASSES)


def test_the_first_step_of_a_bout_is_a_first_step_of_a_real_bout():
    random.seed(0)
    # real flies started their bouts with 2 mm steps, and went on with 0.1 mm steps when their previous step was below 0.5 mm
    steps = pd.DataFrame({"step_mm": [2.0] * 5 + [0.1] * 5, "turn_rad": np.nan, "speed_class": [START_CLASS] * 5 + [0] * 5})
    walk = RandomWalk(steps, SPEED_CLASSES)
    assert walk.sample(state(0.0, start=True))[0][0] == 2.0 and walk.sample(state(0.0, start=False))[0][0] == 0.1
    positioned = PositionedWalk(real_steps([(10.0, 10.0, np.nan, np.nan, 0.0, 2.0, np.nan, 0.0)] * 5).assign(start=True), k=5)
    nights = [pd.DataFrame({"moving": [False, True], "length": [2, 10**6]})]
    model = SpatialSimulator(nights, positioned, lambda seconds: 0.0, radius_mm=20.0, contact_distance_mm=0.0)
    model.run(3)
    assert np.isclose(model.distance_walked[0], 2.0)


def state(previous_step_mm=0.0, d_wall_mm=10.0, d_food_mm=10.0, wall_angle_rad=np.nan, food_angle_rad=np.nan, to_rest_s=1000.0, start=False, zt_h=13.0, still_s=600.0, pace_mm=3.0):
    """The state of one fly, as the walks take it"""
    return {k: np.array([v], dtype=(bool if k == "start" else float)) for k, v in locals().items()}


def real_steps(rows):
    """Real seconds for a PositionedWalk: rows of (d_wall, d_food, wall_angle, food_angle, previous_step, step, turn, new_wall_angle), 1000 s from resting"""
    columns = ["d_wall_mm", "d_food_mm", "wall_angle_rad", "food_angle_rad", "previous_step_mm", "step_mm", "turn_rad", "new_wall_angle_rad"]
    return pd.DataFrame(rows, columns=columns).assign(to_rest_s=1000.0, start=False, zt_h=13.0, still_s=600.0, pace_mm=3.0)


def test_steps_are_the_distance_and_turn_between_seconds():
    positions = pd.DataFrame({"t": [0, 1, 2, 3, 5], "x_mm": [0, 3, 3, 0, 0], "y_mm": [0, 0, 4, 4, 4]})
    steps = steps_of(positions)
    assert np.allclose(steps["step_mm"].to_numpy()[1:4], [3, 4, 3])
    assert np.isnan(steps["step_mm"].iloc[0]) and np.isnan(steps["step_mm"].iloc[4])  # no previous second, or a gap
    assert np.isnan(steps["turn_rad"].iloc[1])  # the first step has no previous heading
    assert np.allclose(steps["turn_rad"].to_numpy()[2:4], [np.pi / 2, np.pi / 2])
    assert np.allclose(steps["previous_step_mm"].to_numpy()[2:4], [3, 4])
    assert np.allclose(steps["radius_mm"].to_numpy(), [0, 3, 5, 4, 4])


def test_speed_classes_cover_every_step_and_nan_is_unknown():
    assert list(speed_class([0, 0.4, 0.5, 3, 7, 100, np.nan])) == [0, 0, 1, 2, 3, 4, -1]


def test_entries_count_crossings_into_the_distance():
    assert entries([10, 3, 2, 10, 10, 4, 10], 4) == 2
    assert entries([3, 3, 3], 4) == 0


def test_segment_distance_is_the_closest_approach_within_the_second():
    still = np.array([0.0, 0.0])
    # a fly passing by a still one at 1 mm, from one side to the other: the closest approach is in the middle of the second
    assert np.isclose(segment_distance(np.array([-10.0, 1.0]), np.array([10.0, 1.0]), still, still), 1.0)
    # the end points are 10 mm away, the fly never jumps over
    assert segment_distance(np.array([-10.0, 0.0]), np.array([10.0, 0.0]), still, still) == 0.0
    # two flies moving in parallel keep their distance
    assert np.isclose(segment_distance(np.array([0.0, 0.0]), np.array([5.0, 0.0]), np.array([0.0, 2.0]), np.array([5.0, 2.0])), 2.0)


def test_the_walk_samples_steps_of_the_class_of_the_previous_step():
    steps = pd.DataFrame({"step_mm": [0.1, 8.0], "turn_rad": [0.0, 1.0], "speed_class": speed_class([0.0, 7.0])})
    walk = RandomWalk(steps, SPEED_CLASSES)
    assert walk.sample(state(0.0))[:2] == (0.1, 0.0)
    assert walk.sample(state(6.0))[:2] == (8.0, 1.0)
    # a class without data takes the closest one, and the plain walk never gives a direction from the wall
    step, turn, new_wall_angle = walk.sample(state(100.0))
    assert (step, turn) == (8.0, 1.0) and np.isnan(new_wall_angle)


def test_flies_stay_in_the_dish_and_an_inactive_fly_does_not_move():
    random.seed(0)
    nights = [pd.DataFrame({"moving": [True], "length": [10**6]}), pd.DataFrame({"moving": [False], "length": [10**6]})]
    model = SpatialSimulator(nights, straight_walk(step_mm=12.0), lambda seconds: 0.0, radius_mm=30.0, contact_distance_mm=0.0)
    resting = model.positions[1].copy()
    model.run(2000)
    assert np.hypot(*model.positions[0]) <= 30.0 + 1e-9
    assert np.array_equal(model.positions[1], resting)
    assert model.distance_walked[0] == 2000 * 12.0 and model.distance_walked[1] == 0


def test_a_walker_passing_by_a_resting_fly_contacts_it_once_and_may_wake_it():
    random.seed(0)
    nights = [pd.DataFrame({"moving": [True], "length": [10**6]}), pd.DataFrame({"moving": [False, True], "length": [10**6, 10**6]})]
    model = SpatialSimulator(nights, straight_walk(step_mm=3.0), lambda seconds: 1.0, radius_mm=100.0, contact_distance_mm=4.0)
    model.positions[:] = [[-20.0, 1.0], [0.0, 0.0]]
    model.headings[0] = 0.0
    activity = model.run(7)
    # the segment of the sixth second, from -5 to -2, is the first within 4 mm of the resting fly (2.2 mm at its end)
    assert model.interaction_count == 1 and model.contact_count == 1
    assert activity[6, 1] and not activity[:6, 1].any()


def test_two_active_flies_interact_without_contact():
    random.seed(0)
    nights = [pd.DataFrame({"moving": [True], "length": [10**6]})] * 2
    model = SpatialSimulator(nights, straight_walk(step_mm=3.0), lambda seconds: 1.0, radius_mm=100.0, contact_distance_mm=4.0)
    model.positions[:] = [[-20.0, 0.0], [20.0, 0.0]]
    model.headings[:] = [0.0, np.pi]
    model.run(10)
    assert model.interaction_count == 1 and model.contact_count == 0


def test_no_fly_active_no_interactions():
    random.seed(0)
    nights = [pd.DataFrame({"moving": [False], "length": [10**6]})] * 3
    model = SpatialSimulator(nights, straight_walk(), lambda seconds: 1.0, radius_mm=10.0, contact_distance_mm=50.0)
    model.run(100)
    assert model.interaction_count == 0 and model.contact_count == 0


def test_spatial_simulations_are_reproducible(tmp_path):
    from flyhostel.modelling.modelling.spatial import simulate_spatial
    arguments = (2, 0, [NIGHT, NIGHT.iloc[1:], NIGHT], straight_walk(), lambda seconds: 0.5, 30.0, 4.0)
    (row, (occupancy, distances, hourly)), (row_again, (occupancy_again, distances_again, hourly_again)) = simulate_spatial(*arguments), simulate_spatial(*arguments)
    assert row == row_again and occupancy.equals(occupancy_again) and distances.equals(distances_again) and hourly.equals(hourly_again)


def test_a_fly_waking_up_next_to_its_neighbour_is_not_a_new_interaction():
    random.seed(0)
    # the second fly rests 2 mm from the first, wakes up after 5 s and walks away: they were already within the contact distance
    nights = [pd.DataFrame({"moving": [False], "length": [10**6]}), pd.DataFrame({"moving": [False, True], "length": [5, 10**6]})]
    model = SpatialSimulator(nights, straight_walk(step_mm=3.0), lambda seconds: 0.0, radius_mm=100.0, contact_distance_mm=4.0)
    model.positions[:] = [[0.0, 0.0], [2.0, 0.0]]
    model.headings[1] = 0.0
    model.close[0, 1] = True
    model.run(10)
    assert model.interaction_count == 0 and model.contact_count == 0


def test_the_state_is_relative_to_the_wall_and_the_nearest_food():
    # a fly 2 mm left of the centre of a dish of radius 10, heading right (+x), with food patches at (5, 0) and (-20, 0)
    s = state_of([-2.0], [0.0], [0.0], 10.0, [[5.0, 0.0], [-20.0, 0.0]])
    assert np.isclose(s["d_wall_mm"][0], 8.0) and np.isclose(s["d_food_mm"][0], 7.0)
    # the outward normal points left (-x): the fly heads straight away from the wall, and straight to the food
    assert np.isclose(abs(s["wall_angle_rad"][0]), np.pi) and np.isclose(s["food_angle_rad"][0], 0.0)
    # without a heading the angles are not known
    assert np.isnan(state_of([-2.0], [0.0], [np.nan], 10.0, [[5.0, 0.0]])["wall_angle_rad"][0])


def test_the_positioned_walk_takes_the_steps_of_the_closest_state():
    # near the wall heading into it, real flies turned around; far from it they went straight
    at_wall = [(1.0, 20.0, 0.0, np.pi, 3.0, 3.0, np.pi, np.nan)] * 5
    far = [(15.0, 5.0, np.pi, 0.0, 3.0, 3.0, 0.0, np.nan)] * 5
    walk = PositionedWalk(real_steps(at_wall + far), k=5)
    assert np.isclose(walk.sample(state(3.0, d_wall_mm=1.5, d_food_mm=19.0, wall_angle_rad=0.1, food_angle_rad=np.pi))[1][0], np.pi)
    assert np.isclose(walk.sample(state(3.0, d_wall_mm=14.0, d_food_mm=6.0, wall_angle_rad=np.pi, food_angle_rad=0.0))[1][0], 0.0)


def test_a_fly_without_a_heading_takes_the_direction_from_the_wall_of_the_data():
    random.seed(0)
    # the only real second starts without a heading and heads straight away from the wall (new_wall_angle pi), 3 mm
    walk = PositionedWalk(real_steps([(5.0, 5.0, np.nan, np.nan, 0.0, 3.0, np.nan, np.pi)] * 5), k=5)
    nights = [pd.DataFrame({"moving": [True], "length": [10**6]})]
    model = SpatialSimulator(nights, walk, lambda seconds: 0.0, radius_mm=20.0, contact_distance_mm=0.0)
    model.positions[:] = [[15.0, 0.0]]
    model.headings[:] = [np.pi / 2]  # stale: the fly has not moved (last_step 0)
    model.run(1)
    assert np.allclose(model.positions[0], [12.0, 0.0])
    # now it has a heading (-x) and the data has no turn for it: it keeps it
    model.run(1)
    assert np.allclose(model.positions[0], [9.0, 0.0]) and model.wall_hits == 0


def test_the_occupancy_map_counts_every_fly_every_second_by_state():
    random.seed(0)
    nights = [pd.DataFrame({"moving": [False], "length": [10**6]})] * 2
    model = SpatialSimulator(nights, straight_walk(), lambda seconds: 0.0, radius_mm=10.0, contact_distance_mm=0.0)
    model.positions[:] = [[0.0, 0.0], [9.0, 9.0]]
    model.run(5)
    assert model.occupancy[0].sum() == 10 and model.occupancy[1].sum() == 0
    assert model.occupancy[0, 5, 5] == 5 and model.occupancy[0, 9, 9] == 5
    # the fly at the centre is 10 mm from the wall and on the food (at the centre); the other is 10 - 12.7 < 0 -> bin 0 from the wall
    assert model.wall_distances[0, 10] == 5 and model.wall_distances[0, 0] == 5 and model.food_distances[0, 0] == 5 and model.food_distances[0, 12] == 5


def test_seconds_to_rest_count_down_to_the_next_inactive_second():
    to_rest = to_rest_of([True, True, False, True, False, True])
    assert np.array_equal(to_rest[:5], [2, 1, np.nan, 1, np.nan], equal_nan=True) and np.isnan(to_rest[5])  # no rest after the last active second


def test_a_fly_about_to_rest_takes_the_steps_of_flies_about_to_rest():
    # real flies far from resting went straight at 3 mm; those about to rest went 0.2 mm
    far = real_steps([(10.0, 10.0, np.nan, np.nan, 0.0, 3.0, np.nan, 0.0)] * 5)
    soon = real_steps([(10.0, 10.0, np.nan, np.nan, 0.0, 0.2, np.nan, np.nan)] * 5).assign(to_rest_s=2.0)
    walk = PositionedWalk(pd.concat([far, soon]), k=5)
    nights = [pd.DataFrame({"moving": [True, False], "length": [3, 10**6]})]
    model = SpatialSimulator(nights, walk, lambda seconds: 0.0, radius_mm=20.0, contact_distance_mm=0.0)
    model.positions[:] = [[10.0, 0.0]]
    model.run(1)
    assert np.isclose(model.distance_walked[0], 0.2)  # 3 s left in the bout: closer to the flies 2 s from resting than 1000 s


def test_the_walk_along_the_night_takes_the_steps_of_the_hour_and_counts_them_by_hour():
    random.seed(0)
    # real flies went 3 mm a second at ZT13 and 1 mm at ZT20 (nothing else differs); the time is continuous, so a fly at ZT 13.5 is halfway
    early = real_steps([(10.0, 10.0, np.nan, np.nan, 0.0, 3.0, np.nan, 0.0)] * 5).assign(zt_h=13.0)
    late = early.assign(step_mm=1.0, zt_h=20.0)
    walk = PositionedWalk(pd.concat([early, late]), k=5, features=ZT_FEATURES)
    nights = [pd.DataFrame({"moving": [True], "length": [10**6]})]
    model = SpatialSimulator(nights, walk, lambda seconds: 0.0, radius_mm=100.0, contact_distance_mm=0.0)
    model.run(3600 + 10)
    assert np.isclose(model.hourly_distance[0], 3600 * 3.0) and np.isclose(model.hourly_distance[1], 10 * 3.0)
    assert model.hourly_active[0] == 3600 and model.hourly_active[1] == 10
    assert np.isclose(model.hourly_locomotion_distance[0], 3600 * 3.0) and model.hourly_locomotion[1] == 10  # 3 mm steps are locomotion
    model = SpatialSimulator(nights, walk, lambda seconds: 0.0, radius_mm=100.0, contact_distance_mm=0.0, start_zt_s=20 * 3600)
    model.run(10)
    assert np.isclose(model.distance_walked[0], 10 * 1.0)
    # the walk without the time of the night mixes both
    model = SpatialSimulator(nights, PositionedWalk(pd.concat([early, late]), k=10), lambda seconds: 0.0, radius_mm=100.0, contact_distance_mm=0.0)
    model.run(100)
    assert 100 * 1.0 < model.distance_walked[0] < 100 * 3.0


def test_the_state_knows_how_long_the_fly_has_been_still():
    # walked at t=1 and t=4 (3 mm), still otherwise; a gap in the trace counts its seconds
    assert np.allclose(still_of([0, 1, 2, 3, 4, 7], [np.nan, 3.0, 0.1, 0.1, 3.0, 0.0]), [600, 600, 1, 2, 3, 3])
    # the simulator keeps the same count: 10 s walking, then 5 s inactive
    walk = PositionedWalk(real_steps([(10.0, 10.0, np.nan, np.nan, 0.0, 3.0, np.nan, 0.0)] * 5), k=5)
    nights = [pd.DataFrame({"moving": [True, False], "length": [10, 10**6]})]
    model = SpatialSimulator(nights, walk, lambda seconds: 0.0, radius_mm=100.0, contact_distance_mm=0.0)
    assert model.still_s[0] == 600  # has not walked yet
    model.run(15)
    assert model.still_s[0] == 6


def test_a_paced_fly_takes_the_steps_of_flies_of_its_pace():
    slow = real_steps([(10.0, 10.0, np.nan, np.nan, 2.0, 2.0, 0.0, np.nan)] * 5).assign(pace_mm=2.0)
    fast = real_steps([(10.0, 10.0, np.nan, np.nan, 2.0, 5.0, 0.0, np.nan)] * 5).assign(pace_mm=5.0)
    walk = PositionedWalk(pd.concat([slow, fast]), k=5, features=PACE_FEATURES)
    nights = [pd.DataFrame({"moving": [True], "length": [10**6]})] * 2
    model = SpatialSimulator(nights, walk, lambda seconds: 0.0, radius_mm=100.0, contact_distance_mm=0.0, pace_mm=[2.0, 5.0])
    model.run(10)
    assert np.isclose(model.distance_walked[0], 10 * 2.0) and np.isclose(model.distance_walked[1], 10 * 5.0)


def test_pauses_are_runs_of_still_seconds_within_a_bout():
    # one bout: still 2 s (opens the bout), walk 1 s, still 3 s, walk; a second bout: still 2 s and the bout ends (censored)
    step = [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0]
    steps = pd.DataFrame({"id": "a", "t": [0, 1, 2, 3, 4, 5, 6, 10, 11], "step_mm": step,
                          "previous_step_mm": [0.0] + step[:-1], "start": [True] + [False] * 6 + [True, False]})
    pauses = pauses_of(steps)
    assert pauses[["at_start", "seconds", "ended"]].values.tolist() == [[True, 2, True], [False, 3, True], [True, 2, False]]
    # Kaplan-Meier: of pauses of 1 (ended), 2 (censored) and 3 s (ended), a third ends after 1 s, the one at risk at 3 s ends then
    assert np.allclose(survival_of([1, 2, 3], [True, False, True]), [1, 2 / 3, 2 / 3, 0])


def test_a_paused_fly_pauses_as_long_as_its_real_night():
    random.seed(0)
    np.random.seed(0)
    # the real fly pauses 3 s, walks 2 s (3 mm), over and over, in one long bout (every pause ends: none is censored)
    pattern = [0.0, 0.0, 0.0, 3.0, 3.0] * 40
    real = real_steps([(10.0, 10.0, np.nan, np.nan, 0.0, s, 0.0, 0.0) for s in pattern])
    real = real.assign(id="night", t=np.arange(len(pattern)), previous_step_mm=[0.0] + pattern[:-1], start=[True] + [False] * (len(pattern) - 1))
    walk = PausedWalk(real, k=5, min_pauses=5)
    assert walk.draw_pause(["night"], [False])[0] == 3 and walk.draw_pause(["night"], [True])[0] == 3
    nights = [pd.DataFrame({"id": "night", "moving": [True], "length": [10**6]})]
    model = SpatialSimulator(nights, walk, lambda seconds: 0.0, radius_mm=1000.0, contact_distance_mm=0.0, record=True)
    model.positions[:] = 0.0  # far from the wall, which would shorten the steps
    model.run(200)
    trace = np.stack(model.trace)[:, 0]
    walked = np.hypot(*np.diff(trace, axis=0).T) >= 0.5
    runs = [len(list(run)) for still, run in itertools.groupby(~walked) if still]  # lengths of the still runs
    assert len(runs) > 10 and set(runs[1:-1]) == {3}  # the first has no displacement for its first second, the last may be cut by the end of the run


def test_active_seconds_without_a_step_are_not_locomotion():
    # real flies went 0.1 mm a second: active, but not walking
    still = real_steps([(10.0, 10.0, np.nan, np.nan, 0.0, 0.1, np.nan, np.nan)] * 5)
    nights = [pd.DataFrame({"moving": [True], "length": [10**6]})]
    model = SpatialSimulator(nights, PositionedWalk(still, k=5), lambda seconds: 0.0, radius_mm=100.0, contact_distance_mm=0.0)
    model.run(10)
    assert model.hourly_active[0] == 10 and model.hourly_locomotion[0] == 0 and model.hourly_locomotion_distance[0] == 0
    assert np.isclose(model.hourly_distance[0], 10 * 0.1)
