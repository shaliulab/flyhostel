import os.path

import numpy as np
import h5py
from flyhostel.data.pose.loaders.roi import arena_calib
from flyhostel.utils.pose_export import load_frame_numbers
from .synthesize import synthesize, FLIP_Y, N
TARGET_FPS=30

def _interpolate_gaps(xy, warn_gap=None, label=""):
    """Linearly interpolate NaN gaps in (2, n_nodes, T), per coord & node, over time.

    - Interior gaps: linear interpolation.
    - Leading/trailing NaNs: held constant at the nearest valid value (np.interp default).
    - A node that is never tracked (all-NaN) is left all-NaN and surfaces later as a
      dropped agent, rather than being silently invented.
    Returns a float64 copy.
    """
    xy = xy.astype(np.float64, copy=True)
    _, n_nodes, T = xy.shape
    t = np.arange(T)
    worst = 0
    for c in range(2):
        for j in range(n_nodes):
            v = xy[c, j]
            good = np.isfinite(v)
            ng = int(good.sum())
            if ng == T or ng == 0:
                continue
            valid = np.flatnonzero(good)
            if warn_gap is not None and valid.size > 1:
                worst = max(worst, int((np.diff(valid) - 1).max()))  # longest interior gap
            v[~good] = np.interp(t[~good], valid, v[valid])
    if warn_gap is not None and worst > warn_gap:
        print(f"WARN {label}: longest interior gap {worst} frames (> {warn_gap}); "
              f"long linear fills are physically dubious")
    return xy

def _decimate_indices(t, target_fps=30):
    """Frame indices resampling to ~target_fps using MEASURED timestamps `t`.

    For each evenly-spaced target time (target_fps apart, spanning the real
    recording duration), pick the source frame whose timestamp is nearest.
    Robust to dropped/jittered frames because it selects on actual time, not
    on an assumed constant framerate.

    Args:
        t: (T,) array of per-frame timestamps in SECONDS, monotonically increasing.
    Returns:
        strictly increasing unique source-frame indices.
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    T = t.shape[0]
    if T <= 1:
        return np.arange(T)

    duration = t[-1] - t[0]
    src_fps_eff = (T - 1) / duration if duration > 0 else np.inf
    if src_fps_eff <= target_fps:          # already at/below target: keep all
        return np.arange(T)

    n_out = int(np.floor(duration * target_fps)) + 1
    if n_out <= 1:
        return np.arange(T)

    # evenly spaced target times across the REAL span
    t_target = t[0] + np.arange(n_out) / target_fps
    # nearest source frame to each target time (searchsorted + neighbor compare)
    j = np.searchsorted(t, t_target)
    j = np.clip(j, 1, T - 1)
    left_closer = (t_target - t[j - 1]) <= (t[j] - t_target)
    idx = np.where(left_closer, j - 1, j)
    return np.unique(idx)                                # dedupe if rounding collides


def read_fly(path, xy_offset_center, px_per_mm, target_fps=30, warn_gap=None, return_indices=False):
    cx, cy = xy_offset_center
    print(f"Opening {path}")
    with h5py.File(path, 'r') as f:
        xy     = f['tracks'][0].astype(np.float64)     # (2,18,T) crop px
        anchor = f['anchor'][:].astype(np.float64)     # (T,2)
        t      = f['t'][:].astype(np.float64)          # (T,) measured timestamps (s)
        names  = [n.decode() for n in f['node_names'][:]]

    keep   = _decimate_indices(t, target_fps)
    xy     = xy[:, :, keep]
    anchor = anchor[keep]

    xy[0] += anchor[:, 0][None, :]; xy[1] += anchor[:, 1][None, :]   # -> full-frame px (absolute)
    xy = _interpolate_gaps(xy, warn_gap=warn_gap, label=os.path.basename(path))
    xy[0] -= cx; xy[1] -= cy                                          # -> arena-centered px
    if FLIP_Y: xy[1] = -xy[1]
    xy /= px_per_mm                                                   # -> mm
    nd = {role: xy[:, names.index(nm), :] for role, nm in N.items()}
    X=synthesize(nd)
    return (X, keep) if return_indices else X

def load_t(path):

    with h5py.File(path, "r") as f:
        t = f["t"][:]
    return t

def build_group(paths, loader, vi, id0, target_fps=TARGET_FPS):
    """Convert one group (one arena, one experiment) into an APF `data` fragment.

    All flies in a group are frame-aligned and share the arena, so calibration
    is resolved once here and reused across every fly in the group.

    Args:
        paths:  list of per-fly .h5 paths belonging to this group
        loader: FlyHostelLoader for this experiment (gives px_per_mm, dbfile)
        vi:     videoidx to assign to every frame of this group
        id0:    first global agent id to hand out (ids are unique across groups)

    Returns:
        (data_fragment, next_id0) where data_fragment has APF's keys for this
        group and next_id0 is id0 advanced past the ids used here.
    """
    # --- 1. Resolve arena geometry once for the whole group ---
    arena_cx, arena_cy, arena_r_px = arena_calib(loader.dbfile, loader.px_per_mm)

    # --- 2. Read + convert each fly to APF keypoints (19, 2, T_i) in mm, arena-centered ---
    paths = sorted(paths)
    out   = [read_fly(p, (arena_cx, arena_cy), loader.px_per_mm,
                      target_fps=target_fps, return_indices=True) for p in paths]
    flies = [o[0] for o in out]
    sels  = [o[1] for o in out]
    n_agents = len(flies)

    # --- 3. Truncate to the shortest fly so all agents share one frame axis ---
    #     Flies within a group are frame-aligned, so this only clips ragged tails.
    n_frames = min(fly.shape[2] for fly in flies)
    flies = [fly[:, :, :n_frames] for fly in flies]

    # --- 4. Stack flies along the agent axis -> X is (n_keypoints, 2, T, n_agents) ---
    X = np.stack(flies, axis=3)

    # --- 5. One stable global id per agent, constant across all frames of the group ---
    #     (never -1 mid-span; process_test_data asserts no isstart inside an id run)
    agent_ids = np.arange(n_agents) + id0
    ids = np.broadcast_to(agent_ids[None, :], (n_frames, n_agents)).copy()

    # --- 6. Bookkeeping: REAL frame numbers, decimated the same way as X ---
    fns = [load_frame_numbers(p, loader.chunksize) for p in paths]

    # (2) all flies in a group were analyzed on the same frames
    for p, fn in zip(paths[1:], fns[1:]):
        assert np.array_equal(fns[0], fn), (
            f"frame numbers differ between {paths[0]} and {p}: "
            f"{len(fns[0])} vs {len(fn)} entries")
    for p, s in zip(paths[1:], sels[1:]):
        assert np.array_equal(sels[0], s), (
            f"decimation indices differ between {paths[0]} and {p} — "
            f"timestamps are not frame-aligned across flies")

    # APF derives isstart from consecutive `frames` (apf/data.py:536) —
    # must be 0,1,2,... per fragment or every frame reads as a sequence start.
    frames = np.arange(n_frames)[:, None]

    # Raw (undecimated) frame numbers, for mapping windows back to video time.
    # APF ignores this key; load_raw_npz_data reads a fixed key list.
    fns = [load_frame_numbers(p, loader.chunksize) for p in paths]
    for p, fn in zip(paths[1:], fns[1:]):
        assert np.array_equal(fns[0], fn), f"frame numbers differ: {paths[0]} vs {p}"
    for p, s in zip(paths[1:], sels[1:]):
        assert np.array_equal(sels[0], s), f"decimation indices differ: {paths[0]} vs {p}"
    frame_numbers = np.asarray(fns[0])[sels[0]][:n_frames][:, None]
    assert frame_numbers.shape[0] == n_frames
    assert np.all(np.diff(frame_numbers[:, 0]) > 0)

    videoidx = np.full((n_frames, 1), vi)
    y        = np.zeros((1, n_frames, n_agents), np.float32)


    # --- ZT encoding: sin/cos of time-of-day, so the model sees circadian phase ---
    # t[i] = seconds since ZT0 for the i-th (undecimated) pose
    ts = [load_t(p) for p in paths]                      # your accessor for the 't' dataset
    for p, tt in zip(paths[1:], ts[1:]):
        assert np.allclose(ts[0], tt), f"timestamps differ: {paths[0]} vs {p}"
    t_sel = np.asarray(ts[0])[sels[0]][:n_frames]        # (T,) seconds since ZT0
    phase = 2 * np.pi * (t_sel % 86400.0) / 86400.0
    zt = np.stack([np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)  # (T, 2)
    assert zt.shape == (n_frames, 2)


    data_fragment = dict(
        X=X, ids=ids, frames=frames,
        frame_numbers=frame_numbers,
        videoidx=videoidx, y=y,
        zt = zt
    )
    
    return data_fragment, id0 + n_agents