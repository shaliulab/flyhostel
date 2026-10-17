"""
PIPELINE 2, stage 1 — physical contact from pose.

Each fly is a set of CAPSULES (a segment between two keypoints with a radius): the body
axis (thorax->head, thorax->abdomen), proboscis, wings, and each leg as thorax->joint->tip
(thorax->tip when the joint is not tracked). For every frame the gap between two flies is

    min over segment pairs ( segment_distance - radius_a - radius_b )

so a crossing of two legs, a leg on the abdomen and two bodies side by side are all one
number, and the argmin says WHICH parts touched. This replaces touch.py,
neighbors_gpu.compute_pairwise_distances_using_bodyparts_gpu and
touch_from_pose.utils.detect_touch_pairs.

Pose is only read inside candidate windows (centroids closer than gate_mm), produced by
centroid.proximity_bouts, so the cost scales with time spent near each other, not with
recording length.
"""
import logging
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

from .backend import get_xp, to_numpy
from .geometry import body_axes, bearing_sector, segment_distance
from .runs import bridge, drop_short, runs

logger = logging.getLogger(__name__)

LEGS = ("fLL", "mLL", "rLL", "fRL", "mRL", "rRL")

# Keypoints the social pipeline reads from the raw pose
NODES = ["head", "thorax", "abdomen", "proboscis", "lW", "rW", *LEGS, *[leg + "J" for leg in LEGS]]

# part: coarse region reported in the tables (head / body / leg)
PART_OF_KIND = {"head": "head", "proboscis": "head", "abdomen": "body", "wing": "body", "leg": "leg"}


@dataclass
class Skeleton:
    names: list          # S segment names, e.g. "fLL_dist"
    kinds: list          # S kinds (keys of radius_mm)
    parts: np.ndarray    # S coarse parts
    i0: np.ndarray       # S start keypoint index
    i1: np.ndarray       # S end keypoint index
    chord_joint: np.ndarray  # S joint index for thorax->tip chords (-1 otherwise);
                             #   a chord is only used in frames where its joint is missing
    radius: np.ndarray   # S capsule radius, mm


def make_skeleton(nodes, radius_mm):
    idx = {n: k for k, n in enumerate(nodes)}
    spec = [("head", "thorax", "head", "head", -1),
            ("abdomen", "thorax", "abdomen", "abdomen", -1),
            ("proboscis", "head", "proboscis", "proboscis", -1),
            ("lW", "thorax", "lW", "wing", -1),
            ("rW", "thorax", "rW", "wing", -1)]
    for leg in LEGS:
        joint = leg + "J"
        spec += [(leg + "_prox", "thorax", joint, "leg", -1),
                 (leg + "_dist", joint, leg, "leg", -1),
                 (leg + "_chord", "thorax", leg, "leg", idx.get(joint, -1))]
    spec = [s for s in spec if s[1] in idx and s[2] in idx]
    return Skeleton(
        names=[s[0] for s in spec],
        kinds=[s[3] for s in spec],
        parts=np.array([PART_OF_KIND[s[3]] for s in spec]),
        i0=np.array([idx[s[1]] for s in spec]),
        i1=np.array([idx[s[2]] for s in spec]),
        chord_joint=np.array([s[4] for s in spec]),
        radius=np.array([radius_mm[s[3]] for s in spec], dtype=np.float32),
    )


def _segments(pose, skel, xp):
    p0, p1 = pose[:, skel.i0], pose[:, skel.i1]
    chords = np.flatnonzero(skel.chord_joint >= 0)
    if len(chords):
        joint_seen = ~xp.isnan(pose[:, skel.chord_joint[chords], 0])
        p1[:, chords] = xp.where(joint_seen[..., None], xp.nan, p1[:, chords])
    return p0, p1


def pair_gap(pose_a, pose_b, skel, use_gpu=False, block_frames=5000):
    """Per-frame gap (mm) between two flies and the segments that achieve it.

    pose_a, pose_b: (L, K, 2) on the same frames.
    Returns gap (L,) (NaN where nothing comparable is tracked), seg_a (L,), seg_b (L,).
    """
    xp = get_xp(use_gpu)
    n_seg = len(skel.names)
    radius = xp.asarray(skel.radius)
    gaps, seg_a, seg_b = [], [], []
    for start in range(0, pose_a.shape[0], block_frames):
        pa = xp.asarray(pose_a[start:start + block_frames])
        pb = xp.asarray(pose_b[start:start + block_frames])
        a0, a1 = _segments(pa, skel, xp)
        b0, b1 = _segments(pb, skel, xp)
        d = segment_distance(a0[:, :, None], a1[:, :, None], b0[:, None], b1[:, None], xp)
        d = (d - radius[:, None] - radius[None, :]).reshape(d.shape[0], -1)
        best = xp.argmin(d, axis=1)
        gap = xp.take_along_axis(d, best[:, None], axis=1)[:, 0]
        gaps.append(to_numpy(xp.where(xp.isinf(gap), xp.nan, gap)))
        best = to_numpy(best)
        seg_a.append(best // n_seg)
        seg_b.append(best % n_seg)
    if not gaps:
        empty = np.zeros(0, dtype=np.int64)
        return np.zeros(0, dtype=np.float32), empty, empty
    return np.concatenate(gaps), np.concatenate(seg_a), np.concatenate(seg_b)


def _mode(values):
    names, counts = np.unique(values, return_counts=True)
    return names[np.argmax(counts)]


def _onset_sector(pose_self, pose_other, nodes, s, n):
    """Where the other fly's thorax lies relative to this fly, over the first n contact frames."""
    thorax = nodes.index("thorax")
    window = slice(s, s + n)
    heading, left = body_axes(pose_self[window], nodes)
    angle, sector = bearing_sector(heading, left, pose_other[window, thorax] - pose_self[window, thorax])
    known = sector != "unknown"
    if not known.any():
        return np.nan, "unknown"
    return float(np.nanmedian(angle[known])), _mode(sector[known])


def contacts_in_window(row, readers, tracks, skel, cfg, nodes, use_gpu=False):
    """Contact bouts of one pair inside one candidate window (a proximity_bouts row)."""
    fns, pose_a = readers[row.id_a].read(row.start_fn, row.end_fn)
    _, pose_b = readers[row.id_b].read(row.start_fn, row.end_fn)
    block = cfg.gpu_block_frames if use_gpu else cfg.block_frames
    gap, seg_a, seg_b = pair_gap(pose_a, pose_b, skel, use_gpu=use_gpu, block_frames=block)

    with np.errstate(invalid="ignore"):
        touch_raw = gap < cfg.touch_mm
    touch = drop_short(bridge(touch_raw, tracks.frames(cfg.merge_gap_s)), tracks.frames(cfg.min_duration_s, 1))
    starts, ends = runs(touch)
    onset_n = tracks.frames(0.2, 1)

    rows = []
    for s, e in zip(starts, ends):
        raw = touch_raw[s:e]
        angle_b, sector_b = _onset_sector(pose_b, pose_a, nodes, s, onset_n)
        angle_a, sector_a = _onset_sector(pose_a, pose_b, nodes, s, onset_n)
        rows.append(dict(
            experiment=tracks.experiment, id_a=row.id_a, id_b=row.id_b, fly_a=row.fly_a, fly_b=row.fly_b,
            start_fn=fns[s], end_fn=fns[e - 1] + tracks.stride,
            duration_s=(e - s) / tracks.fps,
            min_gap_mm=float(np.nanmin(gap[s:e])),
            frac_touch=float(raw.mean()),
            seg_onset_a=skel.names[seg_a[s]], seg_onset_b=skel.names[seg_b[s]],
            part_onset_a=skel.parts[seg_a[s]], part_onset_b=skel.parts[seg_b[s]],
            part_a=_mode(skel.parts[seg_a[s:e][raw]]), part_b=_mode(skel.parts[seg_b[s:e][raw]]),
            # where the partner is, in each fly's own body frame, at contact onset
            bearing_deg_a=angle_a, side_a=sector_a,
            bearing_deg_b=angle_b, side_b=sector_b,
        ))
    return rows


def _detect_batch(candidates, readers, tracks, cfg, nodes, use_gpu):
    skel = make_skeleton(nodes, cfg.radius_mm)
    rows = []
    for row in candidates.itertuples(index=False):
        if row.id_a in readers and row.id_b in readers:
            rows.extend(contacts_in_window(row, readers, tracks, skel, cfg, nodes, use_gpu=use_gpu))
    return rows


def detect_contacts(tracks, readers, candidates, cfg, use_gpu=False, n_jobs=1):
    """Contact bouts of every pair, from pose, inside centroid candidate windows.

    Arguments:
        tracks (GroupTracks)
        readers (dict): fly id -> PoseReader (see tracks.make_pose_readers)
        candidates (pd.DataFrame): centroid.proximity_bouts(tracks, cfg.gate_mm, 0, cfg.merge_gap_s)
        cfg (ContactConfig)
        n_jobs (int): candidate windows are split into contiguous time blocks processed in
            parallel; each worker reopens the .h5 files.

    Returns one row per pair-contact bout (columns suffixed _a/_b like the proximity
    bouts), with start_idx/end_idx on the tracks grid.
    """
    nodes = next(iter(readers.values())).nodes if readers else NODES
    candidates = candidates.sort_values("start_fn").reset_index(drop=True)
    n_jobs = max(1, min(n_jobs if n_jobs > 0 else joblib.cpu_count(), len(candidates)))
    batches = np.array_split(np.arange(len(candidates)), n_jobs)
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_detect_batch)(candidates.iloc[batch], readers, tracks, cfg, nodes, use_gpu)
        for batch in batches
    )
    contacts = pd.DataFrame([r for batch in results for r in batch])
    if contacts.empty:
        logger.warning("%s: no contacts detected", tracks.experiment)
        return contacts
    contacts["start_idx"] = tracks.to_index(contacts["start_fn"])
    contacts["end_idx"] = tracks.to_index(contacts["end_fn"] - tracks.stride) + 1
    contacts["t_start"] = tracks.t[contacts["start_idx"]]
    contacts["t_end"] = tracks.t[contacts["end_idx"] - 1] + 1 / tracks.fps
    return contacts.sort_values(["start_fn", "fly_a", "fly_b"]).reset_index(drop=True)
