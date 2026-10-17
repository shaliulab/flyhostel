"""
APPROACH -> RESPONSE: an immobile fly is approached by a moving fly. How strongly does the
immobile fly respond, and how long until the approaching fly falls asleep?

    events  = classified bouts with label == "approach_to_immobile"
    grade   = recipient movement in [onset, onset + response_s) against its OWN baseline
              [onset - approach_s - baseline_s, onset - approach_s)
    latency = initiator: contact offset -> start of its next sleep bout

RESPONSE GRADE (ordinal, mutually exclusive, highest matching wins)
    3 locomotor   thorax moved >= escape_mm                  (walked / ran off)
    2 postural    heading changed >= turn_deg, or thorax moved >= shift_mm, or limbs
                  active for >= sustained_s                  (turned, shifted, kicked)
    1 limb        a limb (leg tip, wing tip, head) moved faster than its threshold for
                  >= min_limb_s, nothing else                (twitch)
    0 none
   -1 untracked   recipient thorax tracked in < min_coverage of the response window

Limb speed is a PATH speed (step lengths averaged over smooth_s, so a kicking leg does not
cancel out) measured in the recipient's BODY frame (keypoint - thorax, rotated by the
heading), so a fly that turns or walks does not count as moving its limbs. Each limb's
threshold is max(limb_move_mm_s, baseline_k * baseline quantile of that limb) so tracking
jitter of a still fly sets its own floor.

All raw measurements are kept as columns; ``grade_response`` is a pure function of them,
so thresholds can be re-tuned on the saved table without re-reading any pose.

With centroids only (no pose), the same events are graded from centroid displacement:
limb and heading columns are NaN, so limb-only responses read as grade 0.
"""
import logging
import warnings

import numpy as np
import pandas as pd

from .geometry import body_axes, path_speed, wrap_angle
from .runs import drop_short

logger = logging.getLogger(__name__)

GRADE_LABELS = {-1: "untracked", 0: "none", 1: "limb", 2: "postural", 3: "locomotor"}
LIMBS = ("fLL", "mLL", "rLL", "fRL", "mRL", "rRL", "lW", "rW", "head")


def _side(limb):
    if limb in ("lW", "rW"):
        return "left" if limb == "lW" else "right"
    if limb.endswith("LL"):
        return "left"
    if limb.endswith("RL"):
        return "right"
    return "midline"


def select_approaches(classified):
    return classified.loc[classified["label"] == "approach_to_immobile"].reset_index(drop=True)


def recipient_response_from_pose(event, reader, tracks, cfg, approach_s):
    """Raw response measurements of the recipient for one event (dict)."""
    fps = tracks.fps
    n_base = tracks.frames(cfg.baseline_s, 1)
    n_gap = tracks.frames(approach_s)
    n_resp = tracks.frames(cfg.response_s, 1)
    onset = int(event["start_fn"])
    start = onset - (n_base + n_gap) * tracks.stride
    fns, pose = reader.read(start, onset + n_resp * tracks.stride)
    base = slice(0, n_base)
    resp = slice(n_base + n_gap, n_base + n_gap + n_resp)

    nodes = reader.nodes
    k = {n: i for i, n in enumerate(nodes)}
    thorax = pose[:, k["thorax"]]
    coverage = float(np.isfinite(thorax[resp, 0]).mean())

    heading, _ = body_axes(pose, nodes)
    theta = np.arctan2(heading[:, 1], heading[:, 0])

    # limbs in the body frame: x along the heading, y to the side
    limbs = [n for n in LIMBS if n in k]
    rel = pose[:, [k[n] for n in limbs]] - thorax[:, None]
    cos, sin = np.cos(theta)[:, None], np.sin(theta)[:, None]
    body = np.stack([rel[..., 0] * cos + rel[..., 1] * sin, -rel[..., 0] * sin + rel[..., 1] * cos], axis=-1)
    limb_speed = path_speed(body, fps, cfg.smooth_s)                                       # (L, n_limbs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN slices
        floor = np.nanquantile(limb_speed[base], cfg.baseline_q, axis=0) * cfg.baseline_k
        thresh = np.fmax(cfg.limb_move_mm_s, floor)                                         # NaN floor -> config value
        excess = limb_speed[resp] - thresh
        active = drop_short((excess > 0).any(axis=1), tracks.frames(cfg.min_limb_s, 1))

        thorax_ref = np.nanmedian(thorax[base], axis=0)
        displacement = np.linalg.norm(thorax[resp] - thorax_ref, axis=-1)
        theta_ref = np.angle(np.nanmean(np.exp(1j * theta[base])))
        turn = np.degrees(np.abs(wrap_angle(theta[resp] - theta_ref)))

        steps = np.linalg.norm(np.diff(body, axis=0), axis=-1)                              # limb path per frame
        steps = np.where(np.isnan(steps).all(axis=1), np.nan, np.nansum(steps, axis=1))
        path_resp = np.nansum(steps[resp.start - 1:resp.stop - 1])
        path_base = np.nanmean(steps[base]) * n_resp

    moved = active | (displacement >= cfg.shift_mm) | (turn >= cfg.turn_deg)
    any_excess = np.isfinite(excess).any()
    best = np.unravel_index(np.nanargmax(excess), excess.shape) if any_excess else None
    responding = limbs[best[1]] if (best is not None and active.any()) else None
    return dict(
        pose_coverage=coverage,
        latency_s=float(np.argmax(moved) / fps) if moved.any() else np.nan,
        limb_active_s=float(active.sum() / fps),
        max_limb_excess_mm_s=float(np.nanmax(excess)) if any_excess else np.nan,
        responding_limb=responding,
        responding_side=_side(responding) if responding else None,
        heading_change_deg=float(np.nanmax(turn)) if np.isfinite(turn).any() else np.nan,
        body_displacement_mm=float(np.nanmax(displacement)) if np.isfinite(displacement).any() else np.nan,
        excess_limb_path_mm=float(path_resp - path_base) if np.isfinite(path_base) else np.nan,
    )


def recipient_response_from_centroid(event, tracks, cfg, approach_s):
    """Centroid-only fallback: displacement of the recipient centroid (dict)."""
    rec = int(event["rec_fly"])
    s = int(event["start_idx"])
    n_base, n_gap, n_resp = tracks.frames(cfg.baseline_s, 1), tracks.frames(approach_s), tracks.frames(cfg.response_s, 1)
    base = tracks.xy[max(0, s - n_gap - n_base):max(0, s - n_gap), rec]
    resp = tracks.xy[s:s + n_resp, rec]
    coverage = float(np.isfinite(resp[:, 0]).mean()) if len(resp) else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        displacement = np.linalg.norm(resp - np.nanmedian(base, axis=0), axis=-1)
    moved = displacement >= cfg.shift_mm
    return dict(
        pose_coverage=coverage,
        latency_s=float(np.argmax(moved) / tracks.fps) if moved.any() else np.nan,
        limb_active_s=np.nan, max_limb_excess_mm_s=np.nan, responding_limb=None, responding_side=None,
        heading_change_deg=np.nan,
        body_displacement_mm=float(np.nanmax(displacement)) if np.isfinite(displacement).any() else np.nan,
        excess_limb_path_mm=np.nan,
    )


def retreat_mm(events, tracks, movement, response_s):
    """Distance the recipient moved AWAY from the initiator during the response window."""
    n = tracks.frames(response_s, 1)
    out = []
    for ini, rec, s in zip(events["ini_fly"], events["rec_fly"], events["start_idx"]):
        rel = tracks.xy[s:s + n, rec] - tracks.xy[s:s + n, ini]
        u = rel / np.linalg.norm(rel, axis=-1, keepdims=True)
        out.append(np.nansum((movement.velocity[s:s + n, rec] * u).sum(-1)) / tracks.fps)
    return np.array(out)


def grade_response(events, cfg):
    """Ordinal response grade from the raw columns (see module docstring). Pure function."""
    df = events
    with np.errstate(invalid="ignore"):
        untracked = df["pose_coverage"].to_numpy() < cfg.min_coverage
        locomotor = df["body_displacement_mm"].to_numpy() >= cfg.escape_mm
        postural = (
            (df["heading_change_deg"].to_numpy() >= cfg.turn_deg)
            | (df["body_displacement_mm"].to_numpy() >= cfg.shift_mm)
            | (df["limb_active_s"].to_numpy() >= cfg.sustained_s)
        )
        limb = df["limb_active_s"].to_numpy() > 0
    grade = np.select([untracked, locomotor, postural, limb], [-1, 3, 2, 1], default=0)
    out = df.copy()
    out["response_grade"] = grade
    out["response_label"] = [GRADE_LABELS[g] for g in grade]
    out["ipsilateral"] = np.where(
        out["responding_side"].isin(["left", "right"]) & out["rec_side"].isin(["left", "right"]),
        out["responding_side"] == out["rec_side"], np.nan,
    ) if "rec_side" in out.columns else np.nan
    return out


def annotate_sleep_after(events, tracks, movement, all_bouts, cfg):
    """Sleep of both flies after the interaction.

    Initiator
        ini_sleep_onset_t         ZT of the start of its next sleep bout
        ini_sleep_latency_s       contact offset -> that start. Negative when the initiator
                                  stopped moving before the contact ended and never moved
                                  again until it qualified as asleep (it settled in contact)
        ini_sleep_censored        no sleep within max_sleep_latency_s or before data ends
        ini_bouts_before_sleep    other interaction bouts the initiator had in between
                                  (any partner); > 0 means the latency is confounded
        ini_sleep_dist_to_rec_mm  pair distance when the initiator's sleep starts
                                  (small = it fell asleep next to the recipient)
    Recipient
        rec_asleep                immobile >= min_sleep_s at onset (causal: only the past)
        rec_woke                  centroid moved between onset and offset + response_s
        rec_resleep_latency_s     offset -> start of its next sleep bout, if it woke
    """
    fps = tracks.fps
    n_resp = tracks.frames(cfg.response.response_s, 1)
    min_sleep = cfg.movement.min_sleep_s
    # all_bouts may be pair-ordered (fly_a/fly_b) or oriented (ini_fly/rec_fly)
    fly_x, fly_y = ("fly_a", "fly_b") if "fly_a" in all_bouts.columns else ("ini_fly", "rec_fly")
    rows = []
    for ev in events.itertuples(index=False):
        off = min(int(ev.end_idx), tracks.n_frames - 1)
        onset_idx = movement.next_sleep_onset(ev.ini_fly, off)
        latency = (onset_idx - off) / fps if onset_idx >= 0 else np.nan
        censored = onset_idx < 0 or latency > cfg.response.max_sleep_latency_s
        if censored:
            n_between, dist = np.nan, np.nan
        else:
            mine = (all_bouts[fly_x] == ev.ini_fly) | (all_bouts[fly_y] == ev.ini_fly)
            n_between = int((mine & (all_bouts["start_idx"] > off) & (all_bouts["start_idx"] < onset_idx)).sum())
            dist = float(np.linalg.norm(tracks.xy[onset_idx, ev.ini_fly] - tracks.xy[onset_idx, ev.rec_fly]))

        s = int(ev.start_idx)
        rec_woke = bool(movement.moving[s:off + n_resp, ev.rec_fly].any())
        rec_onset = movement.next_sleep_onset(ev.rec_fly, off) if rec_woke else -1
        rows.append(dict(
            ini_sleep_onset_t=tracks.t[onset_idx] if not censored else np.nan,
            ini_sleep_latency_s=latency if not censored else np.nan,
            ini_sleep_censored=censored,
            ini_bouts_before_sleep=n_between,
            ini_sleep_dist_to_rec_mm=dist,
            rec_asleep=bool(ev.rec_immobile_s >= min_sleep),
            rec_woke=rec_woke,
            rec_resleep_latency_s=(rec_onset - off) / fps if rec_onset >= 0 else np.nan,
        ))
    return pd.concat([events.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def approach_response(classified, tracks, movement, all_bouts, cfg, readers=None):
    """Approach->response event table.

    Arguments:
        classified (pd.DataFrame): output of classify.classify_bouts (contacts or proximity)
        tracks, movement: GroupTracks / Movement of the experiment
        all_bouts (pd.DataFrame): every bout of the same kind, used to count interactions
            between the event and the initiator's sleep onset
        cfg (SocialConfig)
        readers (dict): fly id -> PoseReader. None = centroid-only grading.
    """
    events = select_approaches(classified)
    if events.empty:
        return events
    approach_s = cfg.proximity.approach_s
    measures = []
    for _, ev in events.iterrows():
        if readers is not None and ev["recipient"] in readers:
            measures.append(recipient_response_from_pose(ev, readers[ev["recipient"]], tracks, cfg.response, approach_s))
        else:
            measures.append(recipient_response_from_centroid(ev, tracks, cfg.response, approach_s))
    events = pd.concat([events, pd.DataFrame(measures)], axis=1)
    events["retreat_mm"] = retreat_mm(events, tracks, movement, cfg.response.response_s)
    events = grade_response(events, cfg.response)
    return annotate_sleep_after(events, tracks, movement, all_bouts, cfg)
