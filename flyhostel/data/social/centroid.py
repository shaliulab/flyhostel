"""
PIPELINE 1 — interactions from centroids only.

    GroupTracks --pair distance--> proximity mask --bridge/min duration--> bouts
                --approach window--> who approached whom, who was moving/immobile

A bout table has one row per UNORDERED pair-bout with columns suffixed _a / _b
(fly_a < fly_b). ``to_directed`` gives the classic (id, nn) view with each bout twice.
The same function, run with a larger distance, gives the candidate windows of the pose
pipeline (contact.py), so both pipelines share one definition of "near".
"""
import numpy as np
import pandas as pd

from .backend import get_xp, to_numpy
from .geometry import pair_distance
from .runs import bridge, drop_short, runs


def proximity_bouts(tracks, max_dist_mm, min_duration_s, merge_gap_s, use_gpu=False):
    """Bouts during which the centroids of a pair are closer than ``max_dist_mm``.

    Returns a DataFrame with columns:
        experiment, id_a, id_b, fly_a, fly_b      pair (fly_* index into tracks.ids)
        start_idx, end_idx                        grid indices, end exclusive
        start_fn, end_fn, t_start, t_end          end exclusive
        duration_s
        min_dist_mm, fn_min_dist                  closest approach and when
        frac_close                                fraction of frames truly below threshold
                                                  (< 1 when short gaps were bridged)
    """
    xp = get_xp(use_gpu)
    xy = xp.asarray(tracks.xy)
    merge = tracks.frames(merge_gap_s)
    min_len = tracks.frames(min_duration_s, 1)

    tables = []
    for i, j in tracks.pairs:
        d = to_numpy(pair_distance(xy, i, j, xp))
        with np.errstate(invalid="ignore"):
            close = d < max_dist_mm
        starts, ends = runs(drop_short(bridge(close, merge), min_len))
        if len(starts) == 0:
            continue
        d_inf = np.where(np.isfinite(d), d, np.inf)
        argmin = np.array([s + np.argmin(d_inf[s:e]) for s, e in zip(starts, ends)])
        n_close = np.add.reduceat(np.append(close, False).astype(np.int64), np.ravel(np.column_stack([starts, ends])))[::2]
        tables.append(pd.DataFrame({
            "id_a": tracks.ids[i], "id_b": tracks.ids[j], "fly_a": i, "fly_b": j,
            "start_idx": starts, "end_idx": ends,
            "min_dist_mm": d[argmin], "fn_min_dist": tracks.frame_number[argmin],
            "frac_close": n_close / (ends - starts),
        }))

    columns = ["experiment", "id_a", "id_b", "fly_a", "fly_b", "start_idx", "end_idx", "start_fn", "end_fn",
               "t_start", "t_end", "duration_s", "min_dist_mm", "fn_min_dist", "frac_close"]
    if not tables:
        return pd.DataFrame(columns=columns)
    bouts = pd.concat(tables, ignore_index=True)
    bouts["experiment"] = tracks.experiment
    bouts["start_fn"] = tracks.frame_number[bouts["start_idx"]]
    bouts["end_fn"] = tracks.end_frame(bouts["end_idx"])
    bouts["t_start"] = tracks.t[bouts["start_idx"]]
    bouts["t_end"] = tracks.t[bouts["end_idx"] - 1] + 1 / tracks.fps
    bouts["duration_s"] = (bouts["end_idx"] - bouts["start_idx"]) / tracks.fps
    return bouts[columns].sort_values(["start_fn", "fly_a", "fly_b"]).reset_index(drop=True)


def annotate_approach(bouts, tracks, movement, approach_s):
    """Who approached whom, and the movement state of both flies at onset.

    In the ``approach_s`` window before onset, the rate at which the pair closes is split
    into what each fly contributes by moving towards the other:
        c_a = v_a . u_ab      c_b = v_b . u_ba      (u = unit vector to the partner)
    share_a = sum(c_a+) / (sum(c_a+) + sum(c_b+)); 1 = only a moved in, 0 = only b did.

    Adds, for x in {a, b}:
        share_x          fraction of the approach contributed by x (NaN if nobody closed in)
        speed_pre_x      mean centroid speed in the approach window          (mm/s)
        immobile_s_x     time x had been immobile when the bout started      (s)
    """
    w = tracks.frames(approach_s, 1)
    share, speed_a, speed_b = [], [], []
    for fa, fb, s in zip(bouts["fly_a"], bouts["fly_b"], bouts["start_idx"]):
        s0 = max(0, s - w)
        if s0 == s:
            share.append(np.nan); speed_a.append(np.nan); speed_b.append(np.nan)
            continue
        rel = tracks.xy[s0:s, fb] - tracks.xy[s0:s, fa]
        u = rel / np.linalg.norm(rel, axis=-1, keepdims=True)
        c_a = np.nansum(np.clip((movement.velocity[s0:s, fa] * u).sum(-1), 0, None))
        c_b = np.nansum(np.clip(-(movement.velocity[s0:s, fb] * u).sum(-1), 0, None))
        share.append(c_a / (c_a + c_b) if (c_a + c_b) > 0 else np.nan)
        speed_a.append(np.nanmean(movement.speed[s0:s, fa]))
        speed_b.append(np.nanmean(movement.speed[s0:s, fb]))

    bouts = bouts.copy()
    onset = np.maximum(bouts["start_idx"].to_numpy() - 1, 0)
    bouts["share_a"] = share
    bouts["share_b"] = 1 - bouts["share_a"]
    bouts["speed_pre_a"] = speed_a
    bouts["speed_pre_b"] = speed_b
    bouts["immobile_s_a"] = movement.since_moving_s[onset, bouts["fly_a"].to_numpy()]
    bouts["immobile_s_b"] = movement.since_moving_s[onset, bouts["fly_b"].to_numpy()]
    return bouts


def orient(bouts):
    """Replace every _a/_b column pair with ini_/rec_ (initiator / recipient).

    The initiator is the fly with the larger approach share; ties and NaN keep a as
    initiator, and ``initiator_share`` then reads 0.5 / NaN, so they stay identifiable.
    """
    flip = (bouts["share_a"] < 0.5).to_numpy()
    out = bouts.copy()
    for col_a in [c for c in bouts.columns if c.endswith("_a")]:
        col_b = col_a[:-2] + "_b"
        if col_b not in bouts.columns:
            continue
        base = col_a[:-2]
        out["ini_" + base] = np.where(flip, bouts[col_b], bouts[col_a])
        out["rec_" + base] = np.where(flip, bouts[col_a], bouts[col_b])
        out = out.drop(columns=[col_a, col_b])
    return out.rename(columns={"ini_share": "initiator_share", "ini_id": "initiator", "rec_id": "recipient"})


def to_directed(bouts):
    """Each pair-bout twice, as (id, nn): the _a/_b columns become _self/_nn."""
    def view(focal, other):
        mapping = {}
        for col in bouts.columns:
            if col.endswith("_" + focal):
                mapping[col] = col[:-2] + "_self"
            elif col.endswith("_" + other):
                mapping[col] = col[:-2] + "_nn"
        return bouts.rename(columns=mapping).rename(columns={"id_self": "id", "id_nn": "nn"})

    return pd.concat([view("a", "b"), view("b", "a")], ignore_index=True)\
        .sort_values(["start_fn", "id"]).reset_index(drop=True)


def interaction_summary(bouts, tracks, bin_s=3600):
    """Per fly and time bin: number of bouts, time spent interacting, bouts initiated.

    ``bouts`` must be oriented (see ``orient``). Bins are ZT seconds floored to ``bin_s``;
    a bout counts in the bin where it starts. A bout counts as initiated only when the
    approach share is > 0.5, so bouts nobody clearly started are initiated by no one.
    """
    columns = ["t_start", "duration_s", "start_fn"]
    initiated = (bouts["initiator_share"] > 0.5).to_numpy()
    directed = pd.concat([
        bouts[columns].assign(id=bouts["initiator"], nn=bouts["recipient"], initiated=initiated),
        bouts[columns].assign(id=bouts["recipient"], nn=bouts["initiator"], initiated=False),
    ], ignore_index=True)
    directed["t_bin"] = (directed["t_start"] // bin_s) * bin_s
    summary = directed.groupby(["id", "t_bin"]).agg(
        n_bouts=("start_fn", "size"),
        time_interacting_s=("duration_s", "sum"),
        n_initiated=("initiated", "sum"),
        median_duration_s=("duration_s", "median"),
        n_partners=("nn", "nunique"),
    ).reset_index()

    # observed time per fly and bin, so rates are comparable across bins with missing data
    tracked = ~np.isnan(tracks.xy[..., 0])
    t_bin = (tracks.t // bin_s) * bin_s
    observed = pd.DataFrame(tracked, columns=tracks.ids).assign(t_bin=t_bin)\
        .groupby("t_bin").sum().div(tracks.fps)\
        .reset_index().melt(id_vars="t_bin", var_name="id", value_name="observed_s")
    summary = observed.merge(summary, on=["id", "t_bin"], how="left").fillna(
        {"n_bouts": 0, "time_interacting_s": 0, "n_initiated": 0, "n_partners": 0})
    summary.insert(0, "experiment", tracks.experiment)
    summary["bouts_per_hour"] = summary["n_bouts"] / (summary["observed_s"] / 3600)
    return summary.sort_values(["id", "t_bin"]).reset_index(drop=True)
