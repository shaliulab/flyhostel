"""
PASS-BY events: an active fly enters a circle of ``radius_mm`` around a focal fly and later
leaves it. Centroids only; no contact is required.

    pair distance < radius_mm  --bridge merge_gap_s-->  inside runs  (one per pass)
    keep runs with an observed entry AND exit (not cut by the start/end of the data)
    for each run, each fly of the pair is tried as focal, the other as passer:
        passer  awake at entry and moving (mean speed in the approach_s before entry
                above movement.move_speed_mm_s)
        focal   state at entry matches focal_state:
                    asleep    in a sleep bout (immobility >= min_sleep_s)
                    immobile  immobile for >= min_focal_immobile_s
                    awake     not in a sleep bout          (negative control)
                    any       no requirement               (split on focal_asleep)

For both flies the sleep/wake STATE at entry is reported with
    *_state_before_s   time in that state up to entry
    *_state_after_s    time the same state continues after exit (0 if it ended inside)
    *_state_changed_during   the state ended while the passer was inside
    *_before_censored / *_after_censored   the state run reaches the start / end of
                       the data, so the duration is a lower bound

The sleep state is the one in Movement.asleep: by default immobility >= min_sleep_s on the
centroids, where the WHOLE immobility bout counts as asleep (ethoscopy sleep_contiguous
convention). A focal fly can therefore be "asleep" at entry with focal_state_before_s < 300 s
if it went on to stay still; focal_immobile_s gives the strictly causal alternative.
"""
import warnings

import numpy as np
import pandas as pd

from .centroid import proximity_bouts
from .runs import state_runs

FOCAL_STATES = ("asleep", "immobile", "awake", "any")


def _state(runs_, asleep, enter, exit_, n_frames, fps):
    """Sleep/wake state of one fly at ``enter`` and how long it lasts around [enter, exit_)."""
    starts, ends = runs_
    k = np.searchsorted(starts, enter, side="right") - 1
    s, e = starts[k], ends[k]
    return dict(
        asleep=bool(asleep[enter]),
        state_before_s=(enter - s) / fps,
        state_after_s=max(0, e - exit_) / fps,
        state_changed_during=bool(e < exit_),
        before_censored=bool(s == 0),
        after_censored=bool(e == n_frames),
    )


def _focal_ok(focal_state, asleep, immobile_s, cfg):
    if focal_state == "asleep":
        return asleep
    if focal_state == "immobile":
        return immobile_s >= cfg.min_focal_immobile_s
    if focal_state == "awake":
        return not asleep
    return True


def passby_events(tracks, movement, cfg, move_speed_mm_s, approach_s, use_gpu=False):
    """
    Arguments:
        tracks (GroupTracks), movement (Movement)
        cfg (PassByConfig)
        move_speed_mm_s (float): passer must move faster than this before entry (MovementConfig)
        approach_s (float): window before entry where the passer speed is measured (ProximityConfig)

    Returns one row per (focal, passer, pass).
    """
    if cfg.focal_state not in FOCAL_STATES:
        raise ValueError(f"focal_state must be one of {FOCAL_STATES}, got {cfg.focal_state!r}")

    bouts = proximity_bouts(tracks, cfg.radius_mm, 0, cfg.merge_gap_s, use_gpu=use_gpu)
    bouts = bouts.loc[(bouts["start_idx"] > 0) & (bouts["end_idx"] < tracks.n_frames)]
    if cfg.max_inside_s is not None:
        bouts = bouts.loc[bouts["duration_s"] <= cfg.max_inside_s]

    fps, n = tracks.fps, tracks.n_frames
    w = tracks.frames(approach_s, 1)
    runs_by_fly = [state_runs(movement.asleep[:, fly]) for fly in range(tracks.n_flies)]

    rows = []
    for b in bouts.itertuples(index=False):
        enter, exit_ = int(b.start_idx), int(b.end_idx)
        for focal, passer in ((b.fly_a, b.fly_b), (b.fly_b, b.fly_a)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)   # untracked window
                passer_speed = np.nanmean(movement.speed[max(0, enter - w):enter, passer])
            if movement.asleep[enter, passer] or not passer_speed > move_speed_mm_s:
                continue
            focal_immobile_s = float(movement.since_moving_s[enter - 1, focal])
            if not _focal_ok(cfg.focal_state, movement.asleep[enter, focal], focal_immobile_s, cfg):
                continue

            row = dict(
                experiment=tracks.experiment,
                focal=tracks.ids[focal], passer=tracks.ids[passer],
                focal_identity=tracks.identities[focal], passer_identity=tracks.identities[passer],
                focal_fly=focal, passer_fly=passer,
                radius_mm=cfg.radius_mm,
                enter_idx=enter, exit_idx=exit_,
                enter_fn=tracks.frame_number[enter], exit_fn=tracks.end_frame(exit_),
                t_enter=tracks.t[enter], t_exit=tracks.t[exit_ - 1] + 1 / fps,
                inside_s=(exit_ - enter) / fps,
                min_dist_mm=b.min_dist_mm, fn_min_dist=b.fn_min_dist,
                t_min_dist=tracks.t[tracks.to_index(b.fn_min_dist)],
                passer_speed_pre_mm_s=float(passer_speed),
                focal_immobile_s=focal_immobile_s,
            )
            for who, fly in (("focal", focal), ("passer", passer)):
                state = _state(runs_by_fly[fly], movement.asleep[:, fly], enter, exit_, n, fps)
                row.update({f"{who}_{key}": value for key, value in state.items()})
            rows.append(row)

    events = pd.DataFrame(rows)
    if events.empty:
        return events
    return events.sort_values(["enter_fn", "focal"]).reset_index(drop=True)
