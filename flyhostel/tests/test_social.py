"""
Synthetic tests for flyhostel.data.social. No experiment data needed.

Scenario used throughout: fly 0 sits still at (10, 10) for the whole recording; fly 1
walks in from (30, 10), stops next to fly 0, walks back out and settles far away; fly 2
never moves and stays far from everyone.
"""
import numpy as np
import pandas as pd
import pytest

from flyhostel.data.social.config import SocialConfig
from flyhostel.data.social.runs import runs, bridge, drop_short, ffill_short_gaps, state_runs
from flyhostel.data.social.passby import passby_events
from flyhostel.data.social.config import PassByConfig
from flyhostel.data.social.geometry import segment_distance, body_axes, bearing_sector
from flyhostel.data.social.tracks import GroupTracks
from flyhostel.data.social.movement import compute_movement
from flyhostel.data.social.centroid import proximity_bouts, annotate_approach, orient, to_directed, interaction_summary
from flyhostel.data.social.classify import classify_bouts
from flyhostel.data.social.contact import NODES, detect_contacts, make_skeleton, pair_gap
from flyhostel.data.social.approach_response import approach_response, recipient_response_from_pose, grade_response

FRAMERATE, STRIDE = 150, 5
FPS = FRAMERATE / STRIDE

# canonical fly in its body frame: x anterior, +y = anatomical left
BODY = {
    "head": (1.0, 0.0), "thorax": (0.0, 0.0), "abdomen": (-1.2, 0.0), "proboscis": (1.2, 0.0),
    "lW": (-1.0, 0.4), "rW": (-1.0, -0.4),
    "fLL": (0.9, 1.3), "fLLJ": (0.5, 0.7), "mLL": (0.0, 1.5), "mLLJ": (0.0, 0.8),
    "rLL": (-0.9, 1.3), "rLLJ": (-0.5, 0.7),
    "fRL": (0.9, -1.3), "fRLJ": (0.5, -0.7), "mRL": (0.0, -1.5), "mRLJ": (0.0, -0.8),
    "rRL": (-0.9, -1.3), "rRLJ": (-0.5, -0.7),
}


def fly_pose(center, heading):
    """(K, 2) keypoints in NODES order for a fly at ``center`` facing ``heading`` (rad)."""
    c, s = np.cos(heading), np.sin(heading)
    local = np.array([BODY[n] for n in NODES])
    return np.stack([local[:, 0] * c - local[:, 1] * s, local[:, 0] * s + local[:, 1] * c], -1) + np.asarray(center)


class FakeReader:
    def __init__(self, frame_number, pose):
        self.frame_number, self.pose, self.nodes, self.stride = frame_number, pose.astype(np.float32), list(NODES), STRIDE

    def read(self, start_fn, end_fn):
        fns = np.arange(start_fn, end_fn, self.stride)
        out = np.full((len(fns), len(self.nodes), 2), np.nan, np.float32)
        idx = (fns - self.frame_number[0]) // self.stride
        ok = (idx >= 0) & (idx < len(self.frame_number))
        out[ok] = self.pose[idx[ok]]
        return fns, out


def scenario(duration_s=1200, arrive_s=400, stay_s=5):
    """Centroid xy (T, 3, 2) and per-frame pose (T, 3, K, 2) of the 3-fly scenario."""
    t = np.arange(0, duration_s, 1 / FPS)
    xy = np.zeros((len(t), 3, 2), np.float32)
    xy[:, 0] = (10, 10)
    xy[:, 2] = (50, 50)
    walk_s = 18 / 2.0                       # 30 -> 12 mm at 2 mm/s
    x1 = np.full(len(t), 30.0)
    inbound = (t >= arrive_s - walk_s) & (t < arrive_s)
    x1[inbound] = 30 - 2.0 * (t[inbound] - (arrive_s - walk_s))
    still = (t >= arrive_s) & (t < arrive_s + stay_s)
    x1[still] = 12.0
    outbound = (t >= arrive_s + stay_s) & (t < arrive_s + stay_s + walk_s)
    x1[outbound] = 12 + 2.0 * (t[outbound] - (arrive_s + stay_s))
    xy[:, 1, 0], xy[:, 1, 1] = x1, 10
    heading1 = np.where(t < arrive_s + stay_s, np.pi, 0.0)   # faces fly 0 while coming in

    pose = np.stack([
        np.repeat(fly_pose(xy[0, 0], 0.0)[None], len(t), 0),
        np.stack([fly_pose(p, h) for p, h in zip(xy[:, 1], heading1)]),
        np.repeat(fly_pose(xy[0, 2], 0.0)[None], len(t), 0),
    ], axis=1)
    return t, xy, pose


def make_tracks(t, xy):
    return GroupTracks(
        experiment="FlyHostel9_3X_2026-01-01_00-00-00",
        ids=[f"FlyHostel9_3X_2026-01-01_0|0{i}" for i in (1, 2, 3)],
        animals=[f"FlyHostel9_3X_2026-01-01_00-00-00__0{i}" for i in (1, 2, 3)],
        identities=[1, 2, 3],
        frame_number=np.arange(len(t), dtype=np.int64) * STRIDE,
        t=t + 3600.0, xy=xy, fps=FPS, stride=STRIDE, chunksize=45000, pixels_per_mm=17.0,
    )


@pytest.fixture(scope="module")
def world():
    t, xy, pose = scenario()
    tracks = make_tracks(t, xy)
    cfg = SocialConfig()
    movement = compute_movement(tracks, cfg.movement)
    readers = {fly_id: FakeReader(tracks.frame_number, pose[:, i]) for i, fly_id in enumerate(tracks.ids)}
    return tracks, cfg, movement, readers, pose


# ---------------------------------------------------------------- runs / geometry
def test_runs_bridge_drop_short():
    m = np.array([0, 1, 1, 0, 0, 1, 0, 0, 0, 1], bool)
    assert [list(x) for x in runs(m)] == [[1, 5, 9], [3, 6, 10]]
    assert bridge(m, 2).astype(int).tolist() == [0, 1, 1, 1, 1, 1, 0, 0, 0, 1]
    assert drop_short(m, 2).astype(int).tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]


def test_ffill_short_gaps_leaves_long_gaps():
    x = np.array([1, np.nan, 3, np.nan, np.nan, np.nan, 7])
    np.testing.assert_array_equal(ffill_short_gaps(x, 1), [1, 1, 3, np.nan, np.nan, np.nan, 7])


def test_segment_distance():
    p = lambda *v: np.array(v, float)
    assert segment_distance(p(0, 0), p(2, 2), p(0, 2), p(2, 0)) == 0                # crossing
    assert segment_distance(p(0, 0), p(1, 0), p(0, 1), p(1, 1)) == pytest.approx(1)  # parallel
    assert segment_distance(p(0, 0), p(0, 0), p(3, 4), p(3, 4)) == pytest.approx(5)  # two points
    assert np.isinf(segment_distance(p(0, np.nan), p(1, 0), p(0, 1), p(1, 1)))      # untracked


def test_bearing_uses_anatomical_left():
    pose = fly_pose((0, 0), np.pi / 2)[None]
    heading, left = body_axes(pose, NODES)
    for vector, expected in [((0, 1), "anterior"), ((0, -1), "posterior"), ((-1, 0), "left"), ((1, 0), "right")]:
        assert bearing_sector(heading, left, np.array([vector], float))[1][0] == expected


# ---------------------------------------------------------------- pipeline 1
def test_proximity_and_approach_attribution(world):
    tracks, cfg, movement, _, _ = world
    bouts = proximity_bouts(tracks, cfg.proximity.max_dist_mm, cfg.proximity.min_duration_s, cfg.proximity.merge_gap_s)
    assert len(bouts) == 1
    bout = bouts.iloc[0]
    assert (bout.fly_a, bout.fly_b) == (0, 1)
    assert bout.min_dist_mm == pytest.approx(2.0, abs=1e-4)
    assert 5 < bout.duration_s < 6                     # 5 s still + walking in/out of 2.5 mm

    oriented = orient(annotate_approach(bouts, tracks, movement, cfg.proximity.approach_s))
    row = classify_bouts(oriented, cfg.movement, cfg.response).iloc[0]
    assert row.initiator == tracks.ids[1] and row.recipient == tracks.ids[0]
    assert row.initiator_share == pytest.approx(1.0)
    assert row.label == "approach_to_immobile"

    directed = to_directed(bouts)
    assert sorted(directed["id"]) == sorted(tracks.ids[:2])
    summary = interaction_summary(oriented, tracks)
    assert summary["n_bouts"].sum() == 2               # one bout, seen by both flies
    assert summary.set_index("id").loc[tracks.ids[1], "n_initiated"] == 1


def test_centroid_approach_response_and_sleep_latency(world):
    tracks, cfg, movement, _, _ = world
    p = cfg.proximity
    bouts = proximity_bouts(tracks, p.max_dist_mm, p.min_duration_s, p.merge_gap_s)
    classified = classify_bouts(orient(annotate_approach(bouts, tracks, movement, p.approach_s)), cfg.movement, cfg.response)
    events = approach_response(classified, tracks, movement, bouts, cfg)
    ev = events.iloc[0]
    assert ev.response_label == "none"
    assert ev.rec_asleep and not ev.rec_woke
    # fly 1 leaves 2.5 mm at ~405.25 s and stops at 414 s: latency ~8.75 s, far from fly 0
    assert ev.ini_sleep_latency_s == pytest.approx(8.75, abs=0.3)
    assert not ev.ini_sleep_censored
    assert ev.ini_sleep_dist_to_rec_mm == pytest.approx(20, abs=0.1)
    assert ev.ini_bouts_before_sleep == 0


# ---------------------------------------------------------------- pipeline 2
def test_pair_gap_touch_and_parts():
    skel = make_skeleton(NODES, SocialConfig().contact.radius_mm)
    a = fly_pose((0, 0), 0.0)[None]
    side_by_side = fly_pose((0, 2.6), 0.0)[None]     # left legs of a overlap right legs of b
    far = fly_pose((0, 10), 0.0)[None]
    gap, sa, sb = pair_gap(a, side_by_side, skel)
    assert gap[0] < 0
    assert skel.parts[sa[0]] == "leg" and skel.parts[sb[0]] == "leg"
    gap, _, _ = pair_gap(a, far, skel)
    assert gap[0] > 5


def test_pose_pipeline_end_to_end(world):
    tracks, cfg, movement, readers, _ = world
    c = cfg.contact
    candidates = proximity_bouts(tracks, c.gate_mm, 0, c.merge_gap_s)
    contacts = detect_contacts(tracks, readers, candidates, c)
    assert len(contacts) == 1
    oriented = orient(annotate_approach(contacts, tracks, movement, cfg.proximity.approach_s))
    classified = classify_bouts(oriented, cfg.movement, cfg.response)
    row = classified.iloc[0]
    assert row.label == "approach_to_immobile"
    assert row.contact_kind in ("head->head", "leg->head", "head->leg", "leg->leg")
    assert row.rec_side == "anterior"                 # fly 1 came in head-on

    events = approach_response(classified, tracks, movement, contacts, cfg, readers=readers)
    ev = events.iloc[0]
    assert ev.response_label == "none"
    assert ev.pose_coverage == 1.0


@pytest.mark.parametrize("perturb, expected", [
    ("none", "none"),
    ("twitch", "limb"),
    ("kick", "postural"),
    ("walk", "locomotor"),
])
def test_response_grades(world, perturb, expected):
    tracks, cfg, _, _, pose = world
    rng = np.random.default_rng(0)
    rec = pose[:, 0].copy() + rng.normal(0, 0.005, pose[:, 0].shape)    # tracking jitter
    onset_i = int(400 * FPS)
    mLL = NODES.index("mLL")
    if perturb == "twitch":                       # one leg out 0.5 mm and back within 0.2 s
        rec[onset_i + 5:onset_i + 11, mLL, 1] += 0.5
    elif perturb == "kick":                       # leg kicking at 5 Hz for 1 s: the period equals
        for k in range(0, 30, 6):                 #   the smoothing window, so displacement-based
            rec[onset_i + k:onset_i + k + 3, mLL, 1] += 0.4   # speed would cancel to 0
    elif perturb == "walk":                       # whole fly moves 2 mm away within 1 s
        shift = np.clip(np.arange(len(rec) - onset_i) / FPS, 0, 1) * 2.0
        rec[onset_i:] -= shift[:, None, None] * np.array([1.0, 0.0])
    reader = FakeReader(tracks.frame_number, rec)
    event = {"start_fn": tracks.frame_number[onset_i]}
    measures = recipient_response_from_pose(event, reader, tracks, cfg.response, cfg.proximity.approach_s)
    graded = grade_response(pd.DataFrame([measures]).assign(rec_side="anterior"), cfg.response).iloc[0]
    assert graded.response_label == expected, measures
    if expected in ("limb", "postural"):
        assert graded.responding_limb == "mLL" and graded.responding_side == "left"


@pytest.mark.parametrize("with_pose", [False, True])
def test_pipeline_analyze(world, with_pose):
    """The exact code path of run_experiment after loading."""
    from flyhostel.data.social.pipeline import analyze
    tracks, cfg, movement, readers, _ = world
    out = analyze(tracks, movement, cfg, readers=readers if with_pose else None)
    assert set(out) >= {"proximity", "summary", "sleep", "approach", "passby"}
    assert ("contacts" in out) == with_pose
    ev = out["approach"].iloc[0]
    assert ev.initiator == tracks.ids[1] and ev.response_label == "none"
    assert ev.ini_sleep_latency_s > 0 and ev.ini_bouts_before_sleep == 0
    assert (out["sleep"]["id"] == tracks.ids[0]).any()
    assert len(out["passby"]) == 1


# ---------------------------------------------------------------- pass-by events
def test_state_runs():
    starts, ends = state_runs(np.array([1, 1, 0, 0, 0, 1], bool))
    assert starts.tolist() == [0, 2, 5] and ends.tolist() == [2, 5, 6]


@pytest.mark.parametrize("focal_state, n_events", [("asleep", 1), ("immobile", 1), ("any", 1), ("awake", 0)])
def test_passby_focal_state(world, focal_state, n_events):
    tracks, cfg, movement, _, _ = world
    pcfg = PassByConfig(radius_mm=5.0, focal_state=focal_state)
    events = passby_events(tracks, movement, pcfg, cfg.movement.move_speed_mm_s, cfg.proximity.approach_s)
    assert len(events) == n_events


def test_passby_timing_and_states(world):
    """Fly 1 crosses 15 mm (5 mm from fly 0) inbound at 398.5 s and outbound at 406.5 s.
    It slept still at (30, 10) until it started walking at 391 s and falls asleep again
    from 414 s (the start of its final immobility bout)."""
    tracks, cfg, movement, _, _ = world
    ev = passby_events(tracks, movement, PassByConfig(radius_mm=5.0), cfg.movement.move_speed_mm_s,
                       cfg.proximity.approach_s).iloc[0]
    assert (ev.focal, ev.passer) == (tracks.ids[0], tracks.ids[1])
    assert ev.t_enter - 3600 == pytest.approx(398.5, abs=0.05)
    assert ev.t_exit - 3600 == pytest.approx(406.5, abs=0.05)
    assert ev.inside_s == pytest.approx(8.0, abs=0.1)
    assert ev.min_dist_mm == pytest.approx(2.0, abs=1e-4)

    # focal: asleep for the whole recording -> both durations censored
    assert ev.focal_asleep and ev.focal_before_censored and ev.focal_after_censored
    assert ev.focal_state_before_s == pytest.approx(398.5, abs=0.1)
    assert not ev.focal_state_changed_during
    # passer: awake since ~391 s, stays awake until ~414 s
    assert not ev.passer_asleep
    assert ev.passer_state_before_s == pytest.approx(7.5, abs=0.5)
    assert ev.passer_state_after_s == pytest.approx(7.5, abs=0.5)
    assert not ev.passer_before_censored and not ev.passer_after_censored
