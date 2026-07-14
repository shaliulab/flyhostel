# extract_burst_traces.py  — NORMAL env (numpy / h5py / flyhostel)
#
# Parallelism is OPTIONAL via `n_jobs` (default 1 = serial). The expensive per-fly work
# (load_arrays + compute_geometry) happens ONCE in the parent; only the independent
# per-BURST frame-table construction is parallelized. Each burst is handed just the
# slice of the distance/confidence arrays its frames span, so the pickle payload per
# task is tiny (a few hundred rows), not the whole recording.
#
# NOTE on expected speedup: the shared load dominates for a single fly, so per-burst
# parallelism helps most when a fly has many bursts. If you process MANY flies, the
# bigger win is to parallelize across flies (run main() per fly under joblib) — the
# per-burst axis here is orthogonal to that and they compose.
import logging
import os
import numpy as np
import pandas as pd

from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.utils import get_pixels_per_mm, get_framerate, get_chunksize
from flyhostel.utils.pose_export import load_arrays
from .proboscis_candidates import (
    compute_geometry, get_first_frame_number, resolve_video_paths,
)


ROOT_DIR = "."
TRACK = 0
PAD_S = 2          # seconds of retracted context to show on each side of a burst

logger = logging.getLogger(__name__)


def _process_burst_trace(bid, grp, gframes, dist_slice, prob_slice,
                         fly, fps, peak_frames, bout_lookup):
    """Build the per-FRAME table for ONE burst. Pure function of its arguments
    (picklable), so it runs identically serial or under joblib.

    `dist_slice`/`prob_slice` are the distance/confidence values for exactly the frames
    in `gframes` (same length, same order). `peak_frames` is the set of global peak
    frame numbers; `bout_lookup` maps (burst_id, bout_in_burst) -> bout_uid.
    """
    # tag each frame with the bout it falls in (NaN between bouts / in the pad)
    bout_in_burst = np.full(gframes.size, np.nan)
    for _, b in grp.iterrows():
        m = (gframes >= int(b["start_fn"])) & (gframes <= int(b["end_fn"]))
        bout_in_burst[m] = b["bout_in_burst"]

    frame_df = pd.DataFrame({
        "fly": fly,
        "burst_id": int(bid),
        "n_in_burst": int(grp["n_in_burst"].iloc[0]),
        "is_solitary": bool(grp["is_solitary"].iloc[0]),
        "bout_in_burst": bout_in_burst,
        "frame_number": gframes,
        "t_s": (gframes - gframes[0]) / fps,
        "dist_mm": dist_slice,
        "prob_conf": prob_slice,
        "is_peak": np.isin(gframes, list(peak_frames)),
    })

    frame_df = frame_df.merge(
        bout_lookup,
        on=["burst_id", "bout_in_burst"],
        how="left",
    )
    return frame_df


def extract_burst_traces(bouts, h5_path, out_feather="burst_traces.feather",
                         gated=False, n_jobs=1):
    """Use the per-bout table (one fly) as an INDEX to pull the real head->proboscis
    distance from the .h5, expanded to one row per FRAME, for every burst that contains
    at least one PE bout.

    Output columns: burst_id, bout_in_burst, bout_uid, frame_number, local_frame,
    video_file, t_s, dist_mm, is_peak  — long format, ready for ggplot.

    A burst spanning a chunk boundary correctly gets multiple video_file values across
    its frames (resolution is per-frame, via resolve_video_paths).

    `n_jobs` parallelizes the per-burst frame-table construction (default 1 = serial).
    """
    bursts = bouts.loc[bouts["label"] == "pe", "burst_id"].drop_duplicates().values
    idx = bouts.loc[bouts["burst_id"].isin(bursts)]

    if idx.empty:
        raise ValueError("no PE bouts in the CSV")

    fly = idx["fly"].iloc[0]
    experiment, identity = fly.split("__")
    ppm = get_pixels_per_mm(experiment)
    fps = get_framerate(experiment)
    chunksize = get_chunksize(experiment)
    first_fn = get_first_frame_number(h5_path, chunksize)

    # distance for EVERY frame, computed exactly as the pipeline does (ONCE, in parent)
    locs, sc, nodes, inst = load_arrays(h5_path)
    g = compute_geometry(locs, sc, nodes, inst)
    dist = g["dist"][:, TRACK] / ppm                       # raw head->proboscis, mm
    prob_conf = g["prob_conf"][:, TRACK]                   # per-frame proboscis confidence
    if gated:
        raise NotImplementedError("wire the plausibility mask here if you want the gated trace")
    n_frames = dist.size

    pad = int(round(PAD_S * fps))
    peak_frames = set(idx["frame_number"].astype(int))
    bout_lookup = idx[["burst_id", "bout_in_burst", "bout_uid"]].drop_duplicates()

    # ---- build per-burst tasks in the PARENT: slice the arrays so workers get only
    #      their frames, never the whole recording. ----
    tasks = []
    for bid, grp in idx.groupby("burst_id"):
        s = int(grp["start_fn"].min()) - pad
        e = int(grp["end_fn"].max()) + pad
        r0, r1 = max(0, s - first_fn), min(n_frames, e - first_fn + 1)
        if r1 <= r0:
            continue
        gframes = (np.arange(r0, r1) + first_fn).astype(int)
        tasks.append((int(bid), grp.copy(), gframes,
                      dist[r0:r1].copy(), prob_conf[r0:r1].copy()))

    if not tasks:
        raise ValueError("no burst frames in range")

    # ---- run per-burst construction, serial or parallel ----
    if n_jobs == 1:
        out = [_process_burst_trace(bid, grp, gframes, dist_slice, prob_slice,
                                    fly, fps, peak_frames, bout_lookup)
               for (bid, grp, gframes, dist_slice, prob_slice) in tasks]
    else:
        from joblib import Parallel, delayed
        out = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_burst_trace)(bid, grp, gframes, dist_slice, prob_slice,
                                          fly, fps, peak_frames, bout_lookup)
            for (bid, grp, gframes, dist_slice, prob_slice) in tasks)

    traces = pd.concat(out, ignore_index=True)

    # ---- attach every bout-/burst-wise metric to each frame of that bout ----
    # frames map to a bout via (burst_id, bout_in_burst); frames between bouts or in the
    # padding have bout_in_burst = NaN and get NaN metrics (correct: not in a bout).
    frame_level = {"frame_number", "t_s", "dist_mm", "prob_conf", "local_frame",
                   "video_file", "is_peak", "chunk", "identity", "local_identity"}
    metric_cols = [c for c in idx.columns
                   if c not in frame_level and c not in ("burst_id", "bout_in_burst", "bout_uid")]
    bout_metrics = idx[["burst_id", "bout_in_burst"] + metric_cols].drop_duplicates(
        subset=["burst_id", "bout_in_burst"])

    traces = traces.merge(bout_metrics, on=["burst_id", "bout_in_burst"],
                          how="left", suffixes=("", "_bout"))

    # per-FRAME video_file + local_frame (straddle-safe), same resolver as the pipeline.
    # Done ONCE on the full concatenated table (it's a per-fly path resolution).
    traces = resolve_video_paths(traces, experiment, int(identity))

    os.makedirs(os.path.dirname(out_feather) or ".", exist_ok=True)
    traces.to_feather(out_feather)
    print(f"{traces['burst_id'].nunique()} bursts, {len(traces)} frames, "
          f"{len(metric_cols)} bout metrics -> {out_feather}")
    return traces


def get_bouts_file(fly):
    return f"{ROOT_DIR}/pe_bouts/{fly}_pe_bouts.feather"


def main(fly, output=None, n_jobs=1):
    experiment, identity = fly.split("__")
    identity = int(identity)
    if output is None:
        output = "."

    loader = FlyHostelLoader(experiment, identity)
    fly = loader.datasetnames[0]
    path = loader.get_pose_file_h5py("raw")
    feather_file = get_bouts_file(fly)
    out_feather = f"{output}/{fly}_traces.feather"

    if os.path.exists(feather_file):
        bouts = pd.read_feather(feather_file).sort_values("pe_score", ascending=False)
        extract_burst_traces(bouts, path, out_feather=out_feather, n_jobs=n_jobs)
        return out_feather
    else:
        logger.warning("%s not found", feather_file)
        return None


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="per-burst parallelism (1 = serial; -1 = all cores)")
    args = ap.parse_args()
    main(args.fly, output=args.output, n_jobs=args.n_jobs)