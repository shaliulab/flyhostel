"""
make_burst_clips.py   —   NORMAL environment

Pre-render, per PE burst:
    * a whole-burst RAW clip (no pose burned in)   -> videos/{fly}_burst_{bid}.mp4
    * a whole-burst pose JSON                       -> videos/{fly}_burst_{bid}.pose.json
    * per-bout pose JSON (no clip)                  -> videos/{fly}_burst_{bid}_bout_{uid}.pose.json

The app overlays the pose on the video in the browser, so alignment is exact per
presented frame and the skeleton can be toggled/recolored without re-encoding.

Each pose-JSON frame now also carries `conf`: the per-node SLEAP point score, aligned
index-for-index with `pts`, so the browser can map node confidence to brightness/alpha.

PARALLELISM
-----------
`n_jobs` controls burst-level parallelism (default 1 = serial). The full pose array AND
the per-node score array are loaded ONCE in the parent; each burst is handed only the
small slice of `xy`/`sc` its frames span, together with a shifted origin
(`first_fn_global`) so the per-frame row lookup `gf - first_fn_global` indexes the slice
correctly. This keeps the pickle payload tiny (a few hundred rows per task) whether
serial or parallel.
"""
import os
import logging
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import h5py
import cv2
from tqdm.auto import tqdm

try:
    from flyhostel.data.pose.constants import skeleton as FH_EDGES
    from flyhostel.data.pose.constants import body_parts_chosen as NODE_NAMES
except Exception:
    FH_EDGES, NODE_NAMES = [], []

from flyhostel.data.pose.main import FlyHostelLoader

TRACK = 0
FOURCC = cv2.VideoWriter_fourcc(*"avc1")
PER_BOUT = False

logger=logging.getLogger(__name__)

def load_pose(h5_path):
    with h5py.File(h5_path, "r") as f:
        tracks = f["tracks"][:]                 # (n_tracks, 2, n_nodes, n_frames)
        nodes  = [n.decode() for n in f["node_names"][:]]
        files  = [e.decode() for e in f["files"][:]]
        scores = f["point_scores"][:] if "point_scores" in f else None   # (n_tracks, n_nodes, n_frames)
    xy = tracks[TRACK].transpose(2, 1, 0)       # (n_frames, n_nodes, 2)
    sc = scores[TRACK].transpose(1, 0) if scores is not None else None    # (n_frames, n_nodes)
    first_chunk = int(os.path.basename(files[0]).split(".")[0])
    return xy, nodes, first_chunk, sc


def edges_in_node_order(nodes):
    """Map FH_EDGES (indices into NODE_NAMES) to indices into THIS h5's node order."""
    if not (FH_EDGES and NODE_NAMES):
        return []
    name_to_i = {n: i for i, n in enumerate(nodes)}
    out = []
    for a, b in FH_EDGES:
        na, nb = NODE_NAMES[int(a)], NODE_NAMES[int(b)]
        if na in name_to_i and nb in name_to_i:
            out.append([name_to_i[na], name_to_i[nb]])
    return out


def make_clip_and_pose(grp, xy, nodes, chunksize, fps,
                       mp4_path, json_path, upscale=1,
                       include_video=True, include_pose=True,
                       first_fn_global=0, sc=None):
    """Write the raw clip and/or the matching pose JSON for one burst or bout.

    `xy` may be a SLICE of the full pose array; `sc` (optional) is the matching slice of
    per-node SLEAP scores, same origin as `xy`. `first_fn_global` is the global frame
    number that corresponds to row 0 of xy/sc, so the per-frame lookup stays correct:
        r = global_frame - first_fn_global   ->   xy[r], sc[r]
    Frame order in the JSON is IDENTICAL to the clip: clip frame k <-> frames[k].
    """
    grp = grp.sort_values("frame_number")
    prob_idx = nodes.index("proboscis") if "proboscis" in nodes else -1

    cap_cache = {}
    writer, size, start_frame = None, None, None
    json_frames = []
    target_uids = set(grp["bout_uid"].dropna().unique().tolist())
    last_local_frame=None
    last_video_file=None

    for _, row in grp.iterrows():
        gf = int(row["frame_number"])
        r = gf - first_fn_global

        img = None
        if include_video:
            src = row["video_file"]
            if last_video_file is not None and src != last_video_file:
                logger.debug("src = %s", src)
            last_video_file=src
    
            local = int(row["local_frame"])
            if not os.path.exists(src):
                continue
            if src not in cap_cache:
                cap_cache[src] = cv2.VideoCapture(src)
            cap = cap_cache[src]
            
            if last_local_frame is not None and local == last_local_frame+1:
                pass
            else:
                logger.debug("setting %s to frame %s", src, local)
                cap.set(cv2.CAP_PROP_POS_FRAMES, local)
            last_local_frame=local
            ok, img = cap.read()
            if not ok:
                continue
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if upscale != 1:
                img = cv2.resize(img, None, fx=upscale, fy=upscale,
                                 interpolation=cv2.INTER_NEAREST)
            if writer is None:
                size = (img.shape[1], img.shape[0])
                logger.debug("initializing video writer @ %s with fps = %s and size = %s", mp4_path, fps, size)
                writer = cv2.VideoWriter(mp4_path, FOURCC, fps, size, isColor=True)
                start_frame = gf
            logger.debug("Writing frame of shape %s", img.shape)
            writer.write(img)

        if include_pose:
            pts, conf = None, None
            if 0 <= r < xy.shape[0]:
                p = xy[r] * upscale
                pts = [[round(float(x), 1), round(float(y), 1)]
                       if np.isfinite(x) and np.isfinite(y) else None
                       for x, y in p]
                if sc is not None and 0 <= r < sc.shape[0]:
                    conf = [None if not np.isfinite(c) else round(float(c), 3)
                            for c in sc[r]]
            if start_frame is None:
                start_frame = gf
            json_frames.append({
                "f": gf,
                "peak": bool(row.get("is_peak", False)),
                "in_bout": bool(pd.notna(row.get("bout_uid")) and
                                row.get("bout_uid") in target_uids),
                "pts": pts,
                "conf": conf,           # per-node SLEAP confidence, same index as pts
            })

    for c in cap_cache.values():
        c.release()

    wrote_anything = False
    if include_video and writer is not None:
        writer.release()
        wrote_anything = True
    if include_pose and json_frames:
        with open(json_path, "w") as fh:
            json.dump({
                "fly": str(grp["fly"].iloc[0]) if "fly" in grp.columns else None,
                "burst_id": int(grp["burst_id"].iloc[0]),
                "fps": fps, "start_frame": start_frame,
                "nodes": nodes, "edges": edges_in_node_order(nodes),
                "prob_idx": prob_idx, "upscale": upscale,
                "frames": json_frames,
            }, fh)
        wrote_anything = True
    return wrote_anything


def _process_burst(bid, burst, xy_slice, sc_slice, slice_origin, nodes, fly,
                   chunksize, fps, out_dir, pad, upscale):
    """All artifacts for one burst. Pure function of its arguments (picklable), so it
    runs identically serial or under joblib.

    Returns 1 if the whole-burst clip+pose was written, else 0. (Per-bout artifacts,
    when PER_BOUT is on, are reported separately via prints and do NOT inflate this
    count, so `sum(results)/n_bursts` in main() stays a clean bursts-succeeded ratio.)
    """
    # ---- whole-burst clip + pose ----
    b_start = burst["frame_number"].min() - pad
    b_end   = burst["frame_number"].max() + pad
    burst_grp = (burst[(burst["frame_number"] >= b_start) &
                       (burst["frame_number"] <= b_end)]
                 .sort_values("frame_number").copy())
    burst_stem = f"{fly}_burst_{int(bid)}"
    ok_b = make_clip_and_pose(
        burst_grp, xy_slice, nodes, chunksize, fps,
        os.path.join(out_dir, burst_stem + ".mp4"),
        os.path.join(out_dir, burst_stem + ".pose.json"),
        upscale=upscale, include_video=True, include_pose=True,
        first_fn_global=slice_origin, sc=sc_slice)
    n = int(ok_b)                          # <-- count the whole-burst artifact (0 or 1)
    if not ok_b:
        print(f"  burst {bid}: whole-burst clip failed")

    # ---- per-bout pose json (no clip) ----
    if PER_BOUT:
        for bout_uid, grp in burst.groupby("bout_uid", sort=True):
            start = grp["frame_number"].min() - pad
            end   = grp["frame_number"].max() + pad
            grp_pad = (burst[(burst["frame_number"] >= start) &
                            (burst["frame_number"] <= end)]
                    .sort_values("frame_number").copy())
            stem = f"{fly}_burst_{int(bid)}_bout_{int(bout_uid)}"
            ok = make_clip_and_pose(
                grp_pad, xy_slice, nodes, chunksize, fps,
                os.path.join(out_dir, stem + ".mp4"),
                os.path.join(out_dir, stem + ".pose.json"),
                upscale=upscale, include_video=False, include_pose=True,
                first_fn_global=slice_origin, sc=sc_slice)
            if not ok:
                print(f"  burst {bid} bout {bout_uid}: no frames")

    return n



def _build_tasks(d, xy, sc, first_fn_global, pad):
    """Slice xy (and sc) per burst in the PARENT; each task carries only its own rows +
    slices. sc may be None (no point_scores in the h5) -> every task gets None."""
    tasks = []
    for bid in dict.fromkeys(d["burst_id"].tolist()):
        burst = d[d["burst_id"] == bid].copy()
        f_lo = int(burst["frame_number"].min()) - pad
        f_hi = int(burst["frame_number"].max()) + pad
        row_lo = max(0, f_lo - first_fn_global)
        row_hi = min(xy.shape[0], f_hi - first_fn_global + 1)
        xy_slice = xy[row_lo:row_hi].copy()                      # small contiguous copy
        sc_slice = sc[row_lo:row_hi].copy() if sc is not None else None
        slice_origin = first_fn_global + row_lo                  # global frame of row 0
        tasks.append((bid, burst, xy_slice, sc_slice, slice_origin))
    return tasks


def main(fly, upscale=1, output=None, n_jobs=1, burst_id=None):
    experiment, identity = fly.split("__")
    experiment = experiment.replace("/", "_")
    identity = int(identity)
    loader = FlyHostelLoader(experiment, identity)

    if output is None:
        output = os.path.join(loader.basedir, "flyhostel", "proboscis_extensions")

    out_dir = os.path.join(output, "videos")

    d = pd.read_feather(f"{output}/{fly}_traces.feather")

    xy, nodes, first_chunk, sc = load_pose(loader.get_pose_file_h5py("raw"))  # load ONCE
    chunksize, fps = loader.chunksize, loader.framerate
    first_fn_global = first_chunk * chunksize
    pad = int(round(fps))

    os.makedirs(out_dir, exist_ok=True)

    tasks = _build_tasks(d, xy, sc, first_fn_global, pad)
    n_bursts = len(tasks)

    def run(t):
        bid, burst, xy_slice, sc_slice, slice_origin = t
        return _process_burst(bid, burst, xy_slice, sc_slice, slice_origin, nodes, fly,
                              chunksize, fps, out_dir, pad, upscale)
    
    if burst_id is not None:
        tasks=[task for task in tasks if task[0]==burst_id]
        with open("tasks.pkl", "wb") as handle:
            pickle.dump(tasks, handle)

    if n_jobs == 1:
        results = [run(t) for t in tqdm(tasks, desc="Making burst clips")]
    else:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_process_burst)(
                bid, burst, xy_slice, sc_slice, slice_origin, nodes, fly,
                chunksize, fps, out_dir, pad, upscale)
            for bid, burst, xy_slice, sc_slice, slice_origin in tqdm(tasks, desc="Making burst clips"))

    print(f"wrote {sum(results)}/{n_bursts} bursts' bout json -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", required=True)
    ap.add_argument("--upscale", type=int, default=1)
    ap.add_argument("--output", default=None)
    ap.add_argument("--n-jobs", type=int, default=1,
                    help="burst-level parallelism (1 = serial; -1 = all cores)")
    args = ap.parse_args()
    main(args.fly, upscale=args.upscale, output=args.output, n_jobs=args.n_jobs)