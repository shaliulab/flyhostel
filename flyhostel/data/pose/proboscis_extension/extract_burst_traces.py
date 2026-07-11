# extract_burst_traces.py  — NORMAL env (numpy / h5py / flyhostel)
import logging
import joblib
import os
import numpy as np
import pandas as pd

from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.utils import get_pixels_per_mm, get_framerate, get_chunksize
from flyhostel.utils.pose_export import load_arrays
from .proboscis_candidates import (
    compute_geometry, get_first_frame_number, resolve_video_paths,
)


ROOT_DIR="."
TRACK = 0
PAD_S = 2          # seconds of retracted context to show on each side of a burst

logger=logging.getLogger(__name__)

def extract_burst_traces(bouts, h5_path, out_csv="burst_traces.csv", gated=False):
    """Use the per-bout table (one fly, label==pe) as an INDEX to pull the real
    head->proboscis distance from the .h5, expanded to one row per FRAME.

    Output columns: burst_id, bout_in_burst, bout_uid, frame_number, local_frame,
    video_file, t_s, dist_mm, is_peak  — long format, ready for ggplot.

    A burst spanning a chunk boundary correctly gets multiple video_file values
    across its frames (resolution is per-frame, via resolve_video_paths)."""
    
    bursts = bouts.loc[bouts["label"] == "pe", "burst_id"].drop_duplicates().values
    idx=bouts.loc[bouts["burst_id"].isin(bursts)]

    if idx.empty:
        raise ValueError("no PE bouts in the CSV")

    fly = idx["fly"].iloc[0]
    experiment, identity = fly.split("__")
    ppm = get_pixels_per_mm(experiment)
    fps = get_framerate(experiment)
    chunksize = get_chunksize(experiment)
    first_fn = get_first_frame_number(h5_path, chunksize)

    # distance for EVERY frame, computed exactly as the pipeline does
    locs, sc, nodes, inst = load_arrays(h5_path)
    g = compute_geometry(locs, sc, nodes, inst)
    dist = g["dist"][:, TRACK] / ppm                       # raw head->proboscis, mm
    prob_conf = g["prob_conf"][:, TRACK]                   # per-frame proboscis confidence
    if gated:
        # optional: NaN out frames the detector would reject (matches the bout signal)
        max_ext_px = None  # keep raw by default; set gated=True only if you want the
        raise NotImplementedError("wire the plausibility mask here if you want the gated trace")
    n_frames = dist.size

    pad = int(round(PAD_S * fps))
    peak_frames = set(idx["frame_number"].astype(int))
    out = []

    bout_lookup = (
        idx[["burst_id", "bout_in_burst", "bout_uid"]]
        .drop_duplicates()
    )

    for bid, grp in idx.groupby("burst_id"):
        s = int(grp["start_fn"].min()) - pad
        e = int(grp["end_fn"].max()) + pad
        r0, r1 = max(0, s - first_fn), min(n_frames, e - first_fn + 1)
        if r1 <= r0:
            continue
        gframes = (np.arange(r0, r1) + first_fn).astype(int)

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
            "dist_mm": dist[r0:r1],
            "prob_conf": prob_conf[r0:r1], 
            "is_peak": np.isin(gframes, list(peak_frames)),
        })
                
        frame_df = frame_df.merge(
            bout_lookup,
            on=["burst_id", "bout_in_burst"],
            how="left",
        )

        out.append(frame_df)

    traces = pd.concat(out, ignore_index=True)

    # ---- attach every bout-/burst-wise metric to each frame of that bout ----
    # frames map to a bout via (burst_id, bout_in_burst); frames between bouts or
    # in the padding have bout_in_burst = NaN and get NaN metrics (correct: not in a bout).
    frame_level = {"frame_number", "t_s", "dist_mm", "prob_conf", "local_frame",
                   "video_file", "is_peak", "chunk", "identity", "local_identity"}
    metric_cols = [c for c in idx.columns
                   if c not in frame_level and c not in ("burst_id", "bout_in_burst", "bout_uid")]
    bout_metrics = idx[["burst_id", "bout_in_burst"] + metric_cols].drop_duplicates(
        subset=["burst_id", "bout_in_burst"])

    traces = traces.merge(bout_metrics, on=["burst_id", "bout_in_burst"],
                          how="left", suffixes=("", "_bout"))

    # per-FRAME video_file + local_frame (straddle-safe), same resolver as the pipeline
    traces = resolve_video_paths(traces, experiment, int(identity))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    traces.to_csv(out_csv, index=False)
    print(f"{traces['burst_id'].nunique()} bursts, {len(traces)} frames, "
          f"{len(metric_cols)} bout metrics -> {out_csv}")
    return traces


def get_bouts_file(fly):
    feather_file=f"{ROOT_DIR}/pe_bouts/{fly}_pe_bouts.feather"
    return feather_file


def main(fly, output=None):
    experiment, identity = fly.split("__")
    identity=int(identity)
    if output is None:
        output = "."

    loader=FlyHostelLoader(experiment, identity)
    fly=loader.datasetnames[0]
    path=loader.get_pose_file_h5py("raw")
    feather_file=get_bouts_file(fly)
    out_csv=f"{output}/{fly}_traces.csv"

    if os.path.exists(feather_file):
        bouts=pd.read_feather(feather_file).sort_values("pe_score", ascending=False)
        traces=extract_burst_traces(
            bouts,
            path,
            out_csv=out_csv
        )
        return out_csv
    else:
        logger.warning("%s not found", feather_file)
        return None

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    args=ap.parse_args()

    main(args.fly)