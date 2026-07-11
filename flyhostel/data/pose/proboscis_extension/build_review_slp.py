"""
build_review_slp.py   —   SLEAP environment (python 3.7 / sleap 1.3.x)

Turn the per-(fly,tier) record feathers from proboscis_candidates.py into one review
.slp PER FLY PER TIER: each suggestion frame carries the raw predicted instance (from
the source .h5) placed on the chunk video at its LOCAL index, so the GUI shows the
prediction and you just drag the proboscis if it's wrong.

UNIFORM PROCESSING: every tier goes through the IDENTICAL frame-selection pipeline
(bout-collapse -> optional quiescence gate -> cap). Nothing branches on tier. That is
what keeps 'likely' a valid positive control of 'potential' — if a frame would be
treated differently in one tier vs another, the control proves nothing.

Run:  python build_review_slp.py --fly EXP__NN_CHUNK --tier {potential|likely|confident_rejected}
"""
import os
import glob
import time
import logging
import numpy as np
import pandas as pd
import h5py
from tqdm.auto import tqdm
import joblib
import sleap
from sleap import Labels, Video, Skeleton, PredictedInstance, LabeledFrame
logger=logging.getLogger(__name__)

try:
    from sleap.io.dataset import SuggestionFrame
except ImportError:
    from sleap.gui.suggestions import SuggestionFrame

from flyhostel.data.pose.constants import skeleton as FH_EDGES        # [(a,b), ...] node-index edges
from flyhostel.data.pose.constants import body_parts_chosen as NODE_NAMES
from flyhostel.utils import get_framerate, get_pixels_per_mm, get_chunksize

# ==========================================================================
# ||||||||||||||||||  MANUAL PARAMETERS (not data-derived)  ||||||||||||||||||
# ==========================================================================
TRACK          = 0      # single-animal .h5 -> one track (index).

# --- bout collapse (identical for every tier) ---
BOUT_MAX_GAP_S = 0.10   # bridge dropouts shorter than this within one extension bout.
                        #   unit: seconds. (repairs held-dropouts; never merges across real gaps.)
MIN_BOUT_FRAMES = 1     # drop bouts shorter than this many frames before collapsing.
                        #   unit: frames. 1 = keep every event (incl. isolated confident hits,
                        #   which is what the 'likely' control needs); raise to cut flicker.

# --- review-volume cap (identical for every tier) ---
MAX_REVIEW_FRAMES = np.inf # cap frames per fly per tier; if exceeded, take a random sample.
                        #   unit: frames. A control/rate estimate needs a sample, not all of them.
SAMPLE_SEED    = 0      # RNG seed for the cap (reproducible sample).  unit: none.

# --- optional quiescence gate (applied IDENTICALLY to all tiers if on) ---
QUIESCENCE_GATE  = False # keep only bouts during sustained body stillness (sleep-ish).
                         #   NOTE: gating the control on quiescence changes what it tests
                         #   (detector-correctness -> detector-correctness-during-sleep), so it
                         #   is applied to BOTH tiers or NEITHER. Default off for a pure control.
QUIESCENCE_S     = 1.0   # min seconds since last body movement to count as quiescent.  unit: s.
MOVE_THRESH_MM_S = 1.0   # thorax speed above which the fly is "moving".  unit: mm/s
                         #   (CALIBRATE per rig — 1 px of jitter is a different mm/s on each).

OUT_DIR = "per_fly_slp"


# ==========================================================================
def build_skeleton():
    skel = Skeleton()
    for n in NODE_NAMES:
        skel.add_node(n)
    for a, b in FH_EDGES:
        skel.add_edge(NODE_NAMES[int(a)], NODE_NAMES[int(b)])
    return skel


def read_h5_meta(h5_path, chunksize):
    with h5py.File(h5_path, "r") as f:
        node_names = [n.decode() for n in f["node_names"][:]]
        files = [e.decode() for e in f["files"][:]]
        n_frames = f["tracks"].shape[-1]
    first_chunk = int(os.path.basename(files[0]).split(".")[0])
    return node_names, first_chunk * chunksize, n_frames


def read_pose_rows(h5_path, rows):
    """rows: sorted unique h5 row indices. Contiguous read (fast), subset in RAM."""
    with h5py.File(h5_path, "r") as f:
        tracks = f["tracks"][TRACK]        # (2, n_nodes, n_frames)
        scores = f["point_scores"][TRACK]  # (n_nodes, n_frames)
    xy = np.transpose(tracks[:, :, rows], (2, 1, 0))   # (len, n_nodes, 2)
    sc = np.transpose(scores[:, rows], (1, 0))         # (len, n_nodes)
    return xy, sc


def collapse_to_bouts(g, fps):
    """Collapse consecutive candidate frames into extension bouts and return one
    representative (peak-extension) frame per bout. IDENTICAL for every tier."""
    d = g.sort_values("frame_number").reset_index(drop=True)
    fn = d["frame_number"].to_numpy()
    max_gap = max(1, round(BOUT_MAX_GAP_S * fps))
    d["bout"] = np.concatenate([[0], (np.diff(fn) > max_gap).cumsum()])
    grp = d.groupby("bout")
    peak = grp.apply(lambda x: x.loc[x["dist_mm"].idxmax(), "frame_number"])
    size = grp["frame_number"].size()
    keep = size >= MIN_BOUT_FRAMES
    return np.unique(peak[keep].to_numpy().astype(int))


def quiescence_since_move(h5_path, ppm, fps):
    """Per-frame seconds since the BODY (thorax, not proboscis) last moved."""
    ti = NODE_NAMES.index("thorax")
    with h5py.File(h5_path, "r") as f:
        xy = f["tracks"][TRACK, :, ti, :].T                 # (n_frames, 2)
    speed = np.concatenate([[0.0], np.linalg.norm(np.diff(xy, axis=0), axis=1) / ppm * fps])
    speed = np.nan_to_num(speed, nan=np.inf)                # missing thorax -> "moving"
    moving = speed > MOVE_THRESH_MM_S
    since = np.empty(len(moving), np.int64)
    last = -1
    for i in range(len(moving)):
        if moving[i]:
            last = i
        since[i] = (i - last) if last >= 0 else i + 1
    return since / fps


def _emit(gframes, out_path, h5_path, chunksize, first_fn, n_frames,
          skel, perm, gf_to_video, gf_to_local):
    """Build one .slp from a set of global frame numbers. Shared by every caller so
    PE / non-PE / tier files use byte-identical instance construction.

    gf_to_video / gf_to_local are PER-FRAME maps (frame_number -> chunk video / local
    frame index) produced in stage 1. Using the stored local_frame — instead of
    recomputing gf % chunksize here — is what keeps chunk-straddling bouts correct:
    each frame is placed in the exact video and local index stage 1 resolved for it."""
    rows_all = gframes - first_fn
    keep = (rows_all >= 0) & (rows_all < n_frames)
    gframes, rows_all = gframes[keep], rows_all[keep]
    if gframes.size == 0:
        print(f"  {os.path.basename(out_path)}: no reviewable frames"); return None

    rows_sorted = np.unique(rows_all)
    t0 = time.time()
    xy, sc = read_pose_rows(h5_path, rows_sorted)
    xy, sc = xy[:, perm, :], sc[:, perm]
    t1 = time.time()
    row_to_col = {int(r): i for i, r in enumerate(rows_sorted)}

    video_cache, videos, lfs, suggestions, seen = {}, [], [], [], set()
    for gf in gframes:
        src = os.path.abspath(gf_to_video[int(gf)])
        local = int(gf_to_local[int(gf)])                 # per-frame, straddle-safe
        # sanity: the resolved local index must belong to the resolved video's chunk
        if int(gf) // chunksize != int(os.path.basename(src).split(".")[0]):
            continue
        if not os.path.exists(src):
            print(f"  missing {src}"); continue
        if src not in video_cache:
            video_cache[src] = Video.from_filename(src); videos.append(video_cache[src])
        video = video_cache[src]
        if (src, local) in seen:
            continue
        seen.add((src, local))

        col = row_to_col[int(gf) - first_fn]
        pts, scr = xy[col], sc[col]
        iscore = float(np.nanmean(scr)) if np.isfinite(scr).any() else 0.0
        inst = PredictedInstance.from_numpy(points=pts, point_confidences=scr,
                                            instance_score=iscore, skeleton=skel)
        lfs.append(LabeledFrame(video=video, frame_idx=local, instances=[inst]))
        suggestions.append(SuggestionFrame(video, local))

    if not lfs:
        return None
    labels = Labels(labeled_frames=lfs, videos=videos, skeletons=[skel])
    labels.suggestions = suggestions
    t2 = time.time(); labels.save(out_path); t3 = time.time()
    print(f"  {out_path}: read {t1-t0:.1f}s build {t2-t1:.1f}s "
          f"save {t3-t2:.1f}s ({len(lfs)} frames)")
    return out_path

def _h5_for(fly_):
    experiment, identity = fly_.split("__")
    identity=int(identity)
    loader=FlyHostelLoader(experiment, identity)
    return loader.get_pose_file_h5py("raw")


def build_labeled_review_slp(fly, out_dir=OUT_DIR,
                             cap_nonpe=MAX_REVIEW_FRAMES):
    """Two review .slp per fly, sharing the SAME instance construction as
    build_per_fly_slp:
        {fly}__PE.slp     : bout peaks pe_features labelled 'pe'      (review precision)
        {fly}__nonPE.slp  : bout peaks labelled feed/groom/walk/...   (review recall)
    PE peaks are all emitted; non-PE peaks are capped (there are far more of them,
    and you only need a representative sample to estimate the false-negative rate)."""
    
    feather_path=f"pe_bouts/{fly}_pe_bouts.feather"
    if not os.path.exists(feather_path):
        logger.warning("%s not found", feather_path)
        return []
    

    bouts = pd.read_feather(feather_path)


    bouts = bouts[bouts["fly"] == fly]
    if bouts.empty:
        print(f"no scored bouts for {fly}"); return []

    h5_path   = bouts["h5_path"].iloc[0]
    chunksize = int(bouts["chunksize"].iloc[0])

    node_names, first_fn, n_frames = read_h5_meta(h5_path, chunksize)
    skel = build_skeleton()
    assert set(node_names) == set(NODE_NAMES), \
        f"node set mismatch: h5={node_names} vs constants={list(NODE_NAMES)}"
    perm = np.array([node_names.index(n) for n in NODE_NAMES])

    # PER-FRAME maps (frame_number -> chunk video / local index), resolved in stage 1.
    gf_to_video = dict(zip(bouts["frame_number"].astype(int), bouts["video_file"]))
    gf_to_local = dict(zip(bouts["frame_number"].astype(int), bouts["local_frame"].astype(int)))

    os.makedirs(out_dir, exist_ok=True)
    written = []

    is_pe = bouts["label"] == "pe"
    pe_frames    = np.unique(bouts.loc[is_pe,  "frame_number"].astype(int).to_numpy())
    nonpe_frames = np.unique(bouts.loc[~is_pe, "frame_number"].astype(int).to_numpy())

    # PE: emit ALL (you want every predicted PE reviewable -> precision)
    out = _emit(pe_frames, os.path.join(out_dir, f"{fly}__PE.slp"),
                h5_path, chunksize, first_fn, n_frames, skel, perm, gf_to_video, gf_to_local)
    if out: written.append(out)

    # non-PE: cap by random sample (many more; a sample estimates recall / missed PEs)
    if nonpe_frames.size > cap_nonpe:
        rng = np.random.default_rng(SAMPLE_SEED)
        nonpe_frames = np.sort(rng.choice(nonpe_frames, cap_nonpe, replace=False))
    out = _emit(nonpe_frames, os.path.join(out_dir, f"{fly}__nonPE.slp"),
                h5_path, chunksize, first_fn, n_frames, skel, perm, gf_to_video, gf_to_local)
    if out: written.append(out)

    print(f"{fly}: PE={pe_frames.size}  nonPE(sampled)={nonpe_frames.size}")
    return written


def build_per_fly_slp(fly, tier, out_dir=OUT_DIR):
    experiment = fly.split("__")[0]
    feathers = glob.glob(f"records/{tier}/{fly}_*records.feather")
    if not feathers:
        print(f"no records for {fly}/{tier}"); return []
    g = pd.concat([pd.read_feather(f) for f in feathers], ignore_index=True)

    # TODO
    g.loc[(g["dist_mm"].isna()) & (g["prob_conf"]==0), "dist_mm"]=0
    ###

    os.makedirs(out_dir, exist_ok=True)

    h5_path = g["h5_path"].iloc[0]
    chunksize = int(g["chunksize"].iloc[0])
    fps = get_framerate(experiment)

    node_names, first_fn, n_frames = read_h5_meta(h5_path, chunksize)
    skel = build_skeleton()
    assert set(node_names) == set(NODE_NAMES), \
        f"node set mismatch: h5={node_names} vs constants={list(NODE_NAMES)}"
    perm = np.array([node_names.index(n) for n in NODE_NAMES])   # h5 order -> NODE_NAMES order

    # ---------------- UNIFORM frame selection (no tier branching) ----------------
    gframes = collapse_to_bouts(g, fps)                          # 1) bouts -> peak frames

    if QUIESCENCE_GATE:                                          # 2) optional stillness gate
        ppm = get_pixels_per_mm(experiment)
        quiet = quiescence_since_move(h5_path, ppm, fps)
        rows = gframes - first_fn
        ok = (rows >= 0) & (rows < len(quiet))
        gframes = gframes[ok][quiet[rows[ok]] >= QUIESCENCE_S]

    cap = int(min(MAX_REVIEW_FRAMES, gframes.size))

    if gframes.size > cap:                         # 3) cap by random sample
        rng = np.random.default_rng(SAMPLE_SEED)
        gframes = np.sort(rng.choice(gframes, cap, replace=False))
    # -----------------------------------------------------------------------------

    # PER-FRAME maps (frame_number -> chunk video / local index) from the records.
    gf_to_video = dict(zip(g["frame_number"].astype(int), g["video_file"]))
    gf_to_local = dict(zip(g["frame_number"].astype(int), g["local_frame"].astype(int)))
    out = _emit(gframes, os.path.join(out_dir, f"{fly}__{tier}.slp"),
                h5_path, chunksize, first_fn, n_frames, skel, perm, gf_to_video, gf_to_local)
    return [out]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", required=False, default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--tier", required=True,
                    choices=["potential", "likely", "confident_rejected"])
    args = ap.parse_args()
    if args.fly is None:
        with open("files.txt", "r") as handle:
            files=handle.readlines()
        flies=[os.path.basename(file).split(".")[0] for file in files]
        joblib.Parallel(n_jobs=args.n_jobs)(
            joblib.delayed(
                build_labeled_review_slp
            )(
                fly, args.tier
            )
            for fly in tqdm(flies)
        )
        
            # build_labeled_review_slp(fly, args.tier)

    else:
        build_labeled_review_slp(args.fly, args.tier)
