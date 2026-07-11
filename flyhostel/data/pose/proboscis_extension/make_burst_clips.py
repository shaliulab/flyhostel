"""
make_burst_clips.py   —   NORMAL environment

Pre-render, per PE burst:
    * a RAW crop clip (no pose burned in)      -> videos/{fly}__burst_{bid}.mp4
    * a pose JSON (skeleton coords per frame)  -> videos/{fly}__burst_{bid}.pose.json

The app overlays the pose on the video in the browser (canvas over <video>), so
alignment is exact per presented frame (via requestVideoFrameCallback) and the
skeleton can be toggled/recolored without re-encoding.

Output lives UNDER THE EXPERIMENT'S OWN TREE:
    {basedir}/flyhostel/proboscis_extensions/videos/
where basedir is derived from the experiment name (FlyHostelN/2X/DATE).

pose JSON schema (compact, for the browser):
    { "fly","burst_id","fps","start_frame","nodes","edges","prob_idx","upscale",
      "frames": [ {"f": <global frame>, "peak": bool, "pts": [[x,y]|null, ...]}, ... ] }
clip frame k  <->  frames[k]  (identical order).
"""
import os
import json
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
# FOURCC = cv2.VideoWriter_fourcc(*"mp4v")
FOURCC = cv2.VideoWriter_fourcc(*"avc1")
MAKE_BOUT_VIDEOS=False
def load_pose(h5_path):
    with h5py.File(h5_path, "r") as f:
        tracks = f["tracks"][:]                 # (n_tracks, 2, n_nodes, n_frames)
        nodes  = [n.decode() for n in f["node_names"][:]]
        files  = [e.decode() for e in f["files"][:]]
    xy = tracks[TRACK].transpose(2, 1, 0)       # (n_frames, n_nodes, 2)
    first_chunk = int(os.path.basename(files[0]).split(".")[0])
    return xy, nodes, first_chunk


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


def make_clip_and_pose(grp, xy, nodes, first_chunk, chunksize, fps,
                       mp4_path, json_path, upscale=1, include_video=True, include_pose=True):
    """Write the raw clip and the matching pose JSON for one burst.
    Frame order in the JSON is IDENTICAL to the clip: clip frame k <-> frames[k]."""
    grp = grp.sort_values("frame_number")
    first_fn_global = first_chunk * chunksize
    prob_idx = nodes.index("proboscis") if "proboscis" in nodes else -1

    cap_cache = {}
    writer, size, start_frame = None, None, None
    json_frames = []
    target_uid = grp["bout_uid"].dropna().iloc[0]

    for _, row in grp.iterrows():
        src = row["video_file"]
        local = int(row["local_frame"])
        if not os.path.exists(src):
            continue
        if src not in cap_cache:
            cap_cache[src] = cv2.VideoCapture(src)
        cap = cap_cache[src]
        cap.set(cv2.CAP_PROP_POS_FRAMES, local)
        ok, img = cap.read()
        if not ok:
            continue
        if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if upscale != 1:
            img = cv2.resize(img, None, fx=upscale, fy=upscale,
                             interpolation=cv2.INTER_NEAREST)

        if include_video:
        
            if writer is None:
                size = (img.shape[1], img.shape[0])
                writer = cv2.VideoWriter(mp4_path, FOURCC, fps, size)
                start_frame = int(row["frame_number"])
            writer.write(img)

        gf = int(row["frame_number"])
        r = gf - first_fn_global
        pts = None
        if 0 <= r < xy.shape[0]:
            p = xy[r] * upscale
            pts = [[round(float(x), 1), round(float(y), 1)]
                   if np.isfinite(x) and np.isfinite(y) else None
                   for x, y in p]
            
        json_frames.append({
            "f": gf,
            "peak": bool(row.get("is_peak", False)),
            "in_bout": bool(pd.notna(row["bout_uid"]) and row["bout_uid"] == target_uid),
            "pts": pts,
        })

    for c in cap_cache.values():
        c.release()
    if writer is None and include_video:
        return False
    
    if include_video: writer.release()

    if include_pose:
        with open(json_path, "w") as fh:
            json.dump({
                "fly": str(grp["fly"].iloc[0]),
                "burst_id": int(grp["burst_id"].iloc[0]),
                "fps": fps, "start_frame": start_frame,
                "nodes": nodes, "edges": edges_in_node_order(nodes),
                "prob_idx": prob_idx, "upscale": upscale,
                "frames": json_frames,
            }, fh)
    return True


def main(fly, upscale=1, output=None):

    experiment, identity = fly.split("__")
    experiment=experiment.replace("/", "_")
    identity=int(identity)
    
    loader=FlyHostelLoader(experiment, identity)
    csv_path=f"./{fly}_traces.csv"
    d = pd.read_csv(csv_path)
    
    xy, nodes, first_chunk = load_pose(loader.get_pose_file_h5py("raw"))

    if output is None:
        output=os.path.join(loader.basedir, "flyhostel", "proboscis_extensions")
    else:
        output="."


    out_dir = os.path.join(output, "videos")
    os.makedirs(out_dir, exist_ok=True)

    burst_ids = list(dict.fromkeys(d["burst_id"].tolist()))
    n = 0
    pad = int(round(loader.framerate))   # one second

    for bid in tqdm(burst_ids, desc="Making burst clips"):
        burst = d[d["burst_id"] == bid]

        # ---- whole-burst clip (start of first bout - pad .. end of last + pad) ----
        b_start = burst["frame_number"].min() - pad
        b_end   = burst["frame_number"].max() + pad
        burst_grp = (burst[(burst["frame_number"] >= b_start) &
                            (burst["frame_number"] <= b_end)]
                        .sort_values("frame_number").copy())
        burst_stem = f"{fly}_burst_{int(bid)}"                    # matches trace_stem
        ok_b = make_clip_and_pose(
            burst_grp, xy, nodes, first_chunk,
            loader.chunksize, loader.framerate,
            os.path.join(out_dir, burst_stem + ".mp4"),
            os.path.join(out_dir, burst_stem + ".pose.json"),
            upscale=upscale)
        if not ok_b:
            print(f"  burst {bid}: whole-burst clip failed")
            
            
        for bout_uid, grp in burst.groupby("bout_uid", sort=True):
            start = grp["frame_number"].min() - pad
            end   = grp["frame_number"].max() + pad

            grp_pad = (
                burst[
                    (burst["frame_number"] >= start) &
                    (burst["frame_number"] <= end)
                ]
                .sort_values("frame_number")
                .copy()
            )

            stem = f"{fly}_burst_{int(bid)}_bout_{int(bout_uid)}"

            ok = make_clip_and_pose(
                grp_pad,
                xy,
                nodes,
                first_chunk,
                loader.chunksize,
                loader.framerate,
                os.path.join(out_dir, stem + ".mp4"),
                os.path.join(out_dir, stem + ".pose.json"),
                upscale=upscale,
                include_video=False,
                include_pose=True
            )

            n += ok
            if not ok:
                print(f"  burst {bid} bout {bout_uid}: no frames")

        if not ok:
            print(f"  burst {bid}: no frames (missing crops?)")
    print(f"wrote {n}/{len(burst_ids)} clips + pose json -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fly", required=True)
    ap.add_argument("--upscale", type=int, default=1)
    args = ap.parse_args()
    main(args.fly,  args.upscale)