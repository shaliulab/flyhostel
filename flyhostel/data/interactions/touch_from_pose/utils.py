import os.path
import numpy as np
import pandas as pd
import xarray as xr
import json
import logging
from typing import Sequence, Hashable, Tuple, Union
import h5py
logger=logging.getLogger(__name__)

def detect_touch_pairs(
    ds: xr.Dataset,
    *,
    individuals: Tuple[Hashable, Hashable] = ("ind_0", "ind_1"),
    thorax: Union[Hashable, int] = "thorax",      # body anchor
    abdomen: Union[Hashable, int] = "abdomen",    # kept for BL reporting
    BODY: Sequence[Union[Hashable, int]] = ("head","thorax","abdomen","wing_left","wing_right"),
    APP:  Sequence[Union[Hashable, int]] = ("LF_tarsus","LM_tarsus","LH_tarsus",
                                            "RF_tarsus","RM_tarsus","RH_tarsus","proboscis"),
    persist_N: int     = 6,
    median_kernel: int = 5,
    merge_gap_sec: float = 0.0,     # merge gaps <= this duration (seconds)
    parallel_tol: float = 1e-9,
    app_app_thresh_px: float | None = None,   # <-- NEW: pixel threshold
    app_app_thresh_BL: float | None = None,   # <-- NEW: body-length threshold
) -> Tuple[xr.DataArray, pd.DataFrame]:

    """
    Detect inter-fly contact based on **segment crossings** between body→appendage lines.

    This detector defines "touch" for a pair of flies as a **planar segment intersection**
    between the line from fly A's body anchor (default: thorax) to any of its appendage tips,
    and the line from fly B's body anchor to any of its appendage tips, evaluated per frame.

    If segments cross (i.e., the intersection parameters t and u both lie in [0, 1]),
    the frame is marked as "touch" (True). If they don't, the function computes the
    infinite-line intersection and reports how far each segment would need to be extended
    to reach that point; the distance metric is then the **negative** of the smaller
    required extension (so non-crossing frames have negative metrics; parallel lines
    are treated as `-inf`).

    The final touch mask is the raw crossing mask filtered by a **persistence** rule
    (minimum run length in frames) and optionally **gap merging** (fill brief False gaps).

    Parameters
    ----------
    ds : xarray.Dataset
        Movement dataset with dims **(time, space, keypoints, individuals)**.
        `space` is expected to contain `["x", "y"]` (order enforced if labeled).
        The first data variable with these dims is treated as the pose tensor.
    individuals : tuple of hashable, optional
        Labels (or integer positions) of the two individuals to compare,
        e.g. `("ind_0", "ind_1")`.
    thorax : hashable or int, optional
        Body anchor keypoint name or index (default "thorax"). Used as the body
        endpoint of each segment. (For a midpoint anchor, precompute and pass in.)
    abdomen : hashable or int, optional
        Abdomen keypoint (default "abdomen"). Used only for reporting mean body
        length in the output dataframe (not for the crossing test itself).
    BODY : sequence of hashable or int, optional
        Unused in this crossing-based definition (kept for API compatibility).
    APP : sequence of hashable or int, optional
        Appendage keypoints to test against (default: 6 tarsi + proboscis).
        May be names or indices; all are tested pairwise across flies.
    persist_N : int, optional
        Minimum number of **consecutive frames** required to keep a touch bout
        (after raw crossing). Shorter runs are suppressed. Default is 6.
    median_kernel : int, optional
        Temporal median filter window (in frames) applied to poses before analysis.
        Must be odd; set to 1 to disable. Default is 5.
    merge_gap_sec : float, optional
        If > 0, **merge** adjacent touch bouts separated by False gaps whose
        duration is ≤ `merge_gap_sec` seconds. Sampling rate is estimated from
        `ds.time`. Default 0.0 (no gap merge).
    parallel_tol : float, optional
        Absolute tolerance for treating R×S≈0 as parallel/colinear (no finite
        intersection). Default 1e-9.

    Returns
    -------
    touch : xarray.DataArray
        Boolean mask over `time` (and `frame_number` coord if present) where True
        indicates frames classified as touch **after** persistence and gap merging.
        `.attrs` records the key parameters.
    df : pandas.DataFrame
        Per-frame diagnostics with columns:
            - `frame_number` : int (if available in `ds`)
            - `BL`           : float; mean body length of the two flies (px; NaN-safe)
            - `metric_cross` : float; crossing distance metric:
                * **> 0** : segments cross; value is `max(|X - appA|, |X - appB|)`
                  where X is the crosspoint.
                * **< 0** : no crossing; value is `-min(extension_A, extension_B)`,
                  the smallest extension (in px) needed for a crossing.
                * **-inf** : segments parallel/colinear under `parallel_tol`.
                * **NaN** : frame skipped (see Missing data handling).
            - `touch_raw`    : bool; raw crossing (True iff t,u∈[0,1]) AND frame valid
            - `touch`        : bool; final mask after persistence and gap merge
            - `t_param`      : float; intersection parameter along A’s segment (NaN if invalid)
            - `u_param`      : float; intersection parameter along B’s segment (NaN if invalid)
            - `kpA_idx`/`kpB_idx` : int; indices of the winning appendage pair
            - `kpA`/`kpB`    : str; names of the winning appendage pair
            - `valid_frame`  : bool; whether frame met validity checks
            - `thoraxA_ok`/`thoraxB_ok` : bool; thorax present for A/B
            - `any_appA_ok`/`any_appB_ok` : bool; at least one appendage present for A/B

    Definition (Geometry)
    ---------------------
    For each frame and each appendage pair (a∈APP of A, b∈APP of B), define segments
    `A: P + t R` with `P = bodyA`, `R = appA - bodyA` and `B: Q + u S` with
    `Q = bodyB`, `S = appB - bodyB`. The infinite-line intersection satisfies:
        t = cross(Q - P, S) / cross(R, S)
        u = cross(Q - P, R) / cross(R, S)
    A **segment crossing** occurs iff `t∈[0,1]` and `u∈[0,1]`. The per-frame metric is:
        * crossing:  `max( ||X - appA||, ||X - appB|| )`, X = intersection point
        * no cross: `-min(extra_A, extra_B)`, where `extra_*` is how much beyond [0,1]
          (in absolute length units) the segment must be extended to reach X.
        * parallel: `-inf`.

    Missing Data Handling
    ---------------------
    A frame is **skipped** (i.e., `metric_cross = NaN`, `touch=False`) if **either**
    fly has a missing thorax position **or** lacks any valid appendage tip position
    (both x and y must be finite). All computations are NaN-safe otherwise.

    Assumptions
    -----------
    * `ds` is regularly sampled or close to it; `merge_gap_sec` uses the median `Δtime`.
    * Coordinates are in the **same pixel space** for both flies (absolute coordinates).
    * `space` is `["x","y"]` (reordered if labeled).

    Examples
    --------
    >>> touch, df = detect_touch_pairs(
    ...     ds_pair,
    ...     individuals=("ind_0","ind_1"),
    ...     thorax="thorax",
    ...     APP=("LF_tarsus","LM_tarsus","LH_tarsus","RF_tarsus","RM_tarsus","RH_tarsus","proboscis"),
    ...     persist_N=6,
    ...     median_kernel=5,
    ...     merge_gap_sec=0.2,
    ... )
    >>> df.loc[df["touch"]].head()[["frame_number","kpA","kpB","metric_cross"]]

    Notes
    -----
    * To treat **near-crossings** as touch, you can adjust the decision to
      `metric_cross >= -ε` with a small ε (e.g., 1–2 px).
    * To use a different body anchor (e.g., midpoint of thorax/abdomen), precompute
      it into the dataset or adapt the body selection before building segments.
    """
    

    # ---------- 0) basic checks / get pose ----------
    required_dims = ("time", "space", "keypoints", "individuals")
    for d in required_dims:
        if d not in ds.dims:
            raise ValueError(f"Expected dims {required_dims}, found {tuple(ds.dims)}")

    vars_match = [v for v in ds.data_vars
                  if all(dim in ds[v].dims for dim in required_dims)]
    if not vars_match:
        raise ValueError("Couldn't find a data variable with dims (time, space, keypoints, individuals).")
    pos = ds[vars_match[0]]
    if "space" in pos.coords and set(pos.coords["space"].values) >= {"x","y"}:
        pos = pos.sel(space=["x","y"])
    if median_kernel and median_kernel > 1:
        pos = pos.rolling(time=median_kernel, center=True, min_periods=1).median()

    indA, indB = individuals
    posA_all = pos.sel(individuals=indA).transpose("time","space","keypoints")
    posB_all = pos.sel(individuals=indB).transpose("time","space","keypoints")

    kp_labels = np.asarray(ds.coords["keypoints"].values)

    def _kp_to_idx(k):
        if isinstance(k, (int, np.integer)): return int(k)
        m = np.where(kp_labels == k)[0]
        if not len(m): raise KeyError(f"keypoint '{k}' not found")
        return int(m[0])

    thor_i = _kp_to_idx(thorax)
    abd_i  = _kp_to_idx(abdomen)
    APP_idx = np.array([_kp_to_idx(k) for k in APP], dtype=int)


    # Anchors and appendages
    A_body = posA_all.sel(keypoints=thorax).transpose("time","space").values        # (T,2)
    B_body = posB_all.sel(keypoints=thorax).transpose("time","space").values        # (T,2)
    A_app  = posA_all.transpose("time","keypoints","space").isel(keypoints=APP_idx).values  # (T,KA,2)
    B_app  = posB_all.transpose("time","keypoints","space").isel(keypoints=APP_idx).values  # (T,KB,2)
    T, KA = A_app.shape[0], A_app.shape[1]
    KB    = B_app.shape[1]

    # ---- VALIDITY MASKS (skip frames if thorax missing OR no appendage found) ----
    valid_thorA = np.isfinite(A_body).all(axis=1)                 # (T,)
    valid_thorB = np.isfinite(B_body).all(axis=1)                 # (T,)
    valid_appA_any = np.isfinite(A_app).all(axis=-1).any(axis=1)  # any APP in A is finite -> (T,)
    valid_appB_any = np.isfinite(B_app).all(axis=-1).any(axis=1)  # any APP in B is finite -> (T,)
    valid_frame = valid_thorA & valid_thorB & valid_appA_any & valid_appB_any  # (T,)

    # Body length (optional reporting)
    BL_A = np.linalg.norm(
        posA_all.sel(keypoints=thorax).transpose("time","space").values
        - posA_all.sel(keypoints=abdomen).transpose("time","space").values, axis=-1)
    BL_B = np.linalg.norm(
        posB_all.sel(keypoints=thorax).transpose("time","space").values
        - posB_all.sel(keypoints=abdomen).transpose("time","space").values, axis=-1)
    BL_mean = np.nanmean(np.stack([BL_A, BL_B], axis=1), axis=1)

    n_missing_bls=np.isnan(BL_mean).sum()
    if n_missing_bls > 0:
        logger.warning("Cannot find mean BL in %s frames", n_missing_bls)
        BL_mean[np.isnan(BL_mean)]=np.nanmean(BL_mean)

    ###
    # --- Pairwise appendage-appendage distances (T, KB, KA) ---
    # diff = A_app (T,KA,2) vs B_app (T,KB,2) -> broadcast to (T,KB,KA,2)
    diff_app = A_app[:, None, :, :] - B_app[:, :, None, :]
    d_appapp = np.linalg.norm(diff_app, axis=-1)  # (T, KB, KA)



    # Build threshold grid if enabled
    use_app_prox = (app_app_thresh_px is not None) or (app_app_thresh_BL is not None)
    if use_app_prox:
        # pixel threshold (scalar) -> (T,KB,KA)
        th_px = None if app_app_thresh_px is None else np.full_like(d_appapp, float(app_app_thresh_px), dtype=float)
        # BL threshold -> per-frame scalar then broadcast
        if app_app_thresh_BL is not None:
            th_bl = (app_app_thresh_BL * BL_mean).reshape(-1, 1, 1)
            th_bl = np.broadcast_to(th_bl, d_appapp.shape)
        else:
            th_bl = None
    
        # If both set, be conservative and use the **max** (i.e., allow either criterion)
        if th_px is not None and th_bl is not None:
            th = np.maximum(th_px, th_bl)
        else:
            th = th_px if th_px is not None else th_bl
    
        # Proximity metric: positive means "close enough"
        app_metric_pairs = th - d_appapp        # (T,KB,KA)
        # Invalidate frames we skip anyway
        app_metric_pairs[~valid_frame, :] = -np.inf
    else:
        # Disabled -> won't influence max
        app_metric_pairs = np.full_like(d_appapp, -np.inf, dtype=float)
    
        
    ###
    # ---------- 1) segment geometry ----------
    R = A_app - A_body[:, None, :]    # (T,KA,2)
    S = B_app - B_body[:, None, :]    # (T,KB,2)

    P = A_body[:, None, None, :]                  # (T,1,1,2)
    Q = B_body[:, None, None, :]                  # (T,1,1,2)
    Rpair = R[:, None, :, :]                      # (T,1,KA,2)
    Spair = S[:, :, None, :]                      # (T,KB,1,2)

    def cross2(a, b):
        return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]

    rxs = cross2(Rpair, Spair)                    # (T,KB,KA)
    qp   = (Q - P)                                # (T,KB,KA,2)

    t_param = cross2(qp, Spair) / rxs             # (T,KB,KA)
    u_param = cross2(qp, Rpair) / rxs             # (T,KB,KA)

    parallel = np.isfinite(rxs) & (np.abs(rxs) < parallel_tol)
    t_param = np.where(parallel, np.nan, t_param)
    u_param = np.where(parallel, np.nan, u_param)

    crosses = (t_param >= 0.0) & (t_param <= 1.0) & (u_param >= 0.0) & (u_param <= 1.0)

    X = P + t_param[..., None] * Rpair            # (T,KB,KA,2)

    A_app_pair = A_app[:, None, :, :]             # (T,1,KA,2)
    B_app_pair = B_app[:, :, None, :]             # (T,KB,1,2)

    dXA = np.linalg.norm(X - A_app_pair, axis=-1) # (T,KB,KA)
    dXB = np.linalg.norm(X - B_app_pair, axis=-1) # (T,KB,KA)

    metric_cross_pairs = np.where(crosses, np.maximum(dXA, dXB), -np.inf)  # (T,KB,KA)

    lenR = np.linalg.norm(Rpair, axis=-1)         # (T,1,KA)
    lenS = np.linalg.norm(Spair, axis=-1)         # (T,KB,1)
    extraA = np.where(t_param < 0, -t_param * lenR, np.where(t_param > 1, (t_param - 1) * lenR, 0.0))
    extraB = np.where(u_param < 0, -u_param * lenS, np.where(u_param > 1, (u_param - 1) * lenS, 0.0))
    extraA = np.where(parallel, np.inf, extraA)
    extraB = np.where(parallel, np.inf, extraB)
    neg_metric_pairs = -np.maximum(extraA, extraB)

    ###
    # --- Colinear overlap detection (treat as crossing) ---
    # Colinear if parallel AND (Q-P) colinear with R and with S (within tol)
    col_qp_R = np.abs(cross2(qp, Rpair)) < parallel_tol
    col_qp_S = np.abs(cross2(qp, Spair)) < parallel_tol
    colinear = parallel & col_qp_R & col_qp_S                           # (T,KB,KA)

    # Segment lengths & squared lengths
    Rlen  = np.linalg.norm(Rpair, axis=-1)                               # (T,1,KA)
    Slen  = np.linalg.norm(Spair, axis=-1)                               # (T,KB,1)
    Rlen2 = np.where(Rlen > 0, Rlen * Rlen, np.nan)                      # guard zero-length
    Slen2 = np.where(Slen > 0, Slen * Slen, np.nan)

    # Project B segment endpoints onto A's R-axis: t for (Q-P) and (Q+S-P)
    # t = dot(V, R) / |R|^2, where V is the vector from P to the point
    # qp = (Q - P); Spair = S
    tB0 = np.nansum(qp * Rpair, axis=-1) / Rlen2                         # (T,KB,KA)
    tB1 = np.nansum((qp + Spair) * Rpair, axis=-1) / Rlen2               # (T,KB,KA)

    # Overlap along A's parameter t in [0,1]
    tBmin = np.minimum(tB0, tB1)
    tBmax = np.maximum(tB0, tB1)
    # intersection of [0,1] with [tBmin, tBmax]
    left  = np.maximum(0.0, tBmin)
    right = np.minimum(1.0, tBmax)
    overlap_scalar = np.maximum(0.0, right - left)                        # (T,KB,KA)

    # Overlap length in pixels (use A's segment length)
    overlap_len = overlap_scalar * Rlen                                   # (T,KB,KA)

    # Positive metric for true colinear overlap; else −inf (so it won't win)
    col_metric = np.where(colinear & (overlap_len > 0), overlap_len, -np.inf)

    # # --- Combine metrics: point-cross/extension vs colinear-overlap ---
    # combined_metric = np.maximum(
    #     np.where(crosses, metric_cross_pairs, neg_metric_pairs),          # existing logic
    #     col_metric                                                       # new colinear overlap
    # )

    # # Pair-level cross mask (either a point-cross or a colinear-overlap)
    # crosses_any = crosses | (colinear & (overlap_len > 0))
    
    ###
    # Existing metrics:
    # - metric_cross_pairs (point crossings -> positive)
    # - neg_metric_pairs   (no crossing -> negative extension)
    # - col_metric         (colinear overlap -> positive overlap length)
    base_metric = np.where(crosses, metric_cross_pairs, neg_metric_pairs)
    base_metric = np.maximum(base_metric, col_metric)

    # NEW: unite with appendage proximity metric
    combined_metric = np.maximum(base_metric, app_metric_pairs)
    
    # Crossing decision per pair: any of the three conditions
    crosses_any = crosses | (colinear & (overlap_len > 0)) | (app_metric_pairs > 0)

    ###
    # combined_metric[~valid_frame, ...] = -np.inf
    flat = combined_metric.reshape(T, KB * KA)
    
    best_flat_idx = np.nanargmax(flat, axis=1)
    best_metric   = flat[np.arange(T), best_flat_idx]
    best_kb, best_ka = np.divmod(best_flat_idx, KA)
    # best_metric[~valid_frame] = np.nan


    # params at winner (still meaningful for point-cross; NaN for colinear case)
    t_best = np.full(T, np.nan, dtype=float)
    u_best = np.full(T, np.nan, dtype=float)

    valid_idx = np.where(valid_frame)[0]
    if valid_idx.size:
        t_best[valid_idx] = t_param[valid_idx, best_kb[valid_idx], best_ka[valid_idx]]
        u_best[valid_idx] = u_param[valid_idx, best_kb[valid_idx], best_ka[valid_idx]]

    # crossing decision for the winning pair
    cross_best = np.zeros(T, dtype=bool)
    if valid_idx.size:
        cross_best[valid_idx] = crosses_any[valid_idx, best_kb[valid_idx], best_ka[valid_idx]]

    # winner appendage labels
    kpA_idx = APP_idx[best_ka]
    kpB_idx = APP_idx[best_kb]
    kpA_names = kp_labels[kpA_idx]
    kpB_names = kp_labels[kpB_idx]
    # for invalid frames, names/idx still filled; that's fine—metric is NaN and touch False.

    # ---------- 3) boolean mask, persistence, gap merge ----------
    raw_touch = cross_best  & valid_frame  # require frame validity

    def persist_mask(mask: np.ndarray, N: int) -> np.ndarray:
        if N <= 1: return mask.copy()
        m = mask.astype(bool)
        edges = np.flatnonzero(np.diff(np.concatenate(([False], m, [False]))))
        starts, ends = edges[::2], edges[1::2]
        out = np.zeros_like(m, dtype=bool)
        for s, e in zip(starts, ends):
            if (e - s) >= N: out[s:e] = True
        return out

    touch = persist_mask(raw_touch, persist_N)

    if merge_gap_sec and merge_gap_sec > 0:
        m = touch.astype(bool)
        tvals = ds["time"].values
        if np.issubdtype(tvals.dtype, np.timedelta64):
            t_s = tvals.astype("timedelta64[ns]").astype(np.int64) / 1e9
        elif np.issubdtype(tvals.dtype, np.datetime64):
            t_s = (tvals - tvals[0]).astype("timedelta64[ns]").astype(np.int64) / 1e9
        else:
            t_s = tvals.astype(float)
        dt = np.median(np.diff(t_s)) if len(t_s) > 1 else 0.0
        fps = (1.0 / dt) if dt > 0 else 50.0
        gap_frames = int(round(merge_gap_sec * fps))
        if gap_frames > 0:
            edges = np.flatnonzero(np.diff(np.concatenate(([False], m, [False]))))
            starts, ends = edges[::2], edges[1::2]
            if len(starts):
                merged = []
                cur_s, cur_e = starts[0], ends[0]
                for s, e in zip(starts[1:], ends[1:]):
                    if (s - cur_e) <= gap_frames:
                        cur_e = e
                    else:
                        merged.append((cur_s, cur_e)); cur_s, cur_e = s, e
                merged.append((cur_s, cur_e))
                m2 = np.zeros_like(m, dtype=bool)
                for s, e in merged: m2[s:e] = True
                touch = m2

    # ---------- 4) outputs ----------
    out = xr.DataArray(
        touch,
        dims=["time"],
        coords={"time": ds["time"]},
        name="touch_frames",
        attrs=dict(
            definition="segment_cross_body_to_appendage",
            persist_frames=persist_N,
            merge_gap_sec=merge_gap_sec,
            individuals=str(individuals),
            skip_rule="thorax missing OR no appendage present in either fly -> metric=np.nan, touch=False",
        ),
    )
    if "frame_number" in ds.coords:
        out = out.assign_coords(frame_number=("time", ds["frame_number"].data))

    app_dist_best = np.full(T, np.nan)

    if valid_idx.size:
        app_dist_best[valid_idx] = d_appapp[valid_idx, best_kb[valid_idx], best_ka[valid_idx]]
    
    # And whether proximity criterion was the one that fired (for the winner)
    won_by_proximity = np.full(T, False)
    if valid_idx.size:
        won_by_proximity[valid_idx] = (app_metric_pairs[valid_idx, best_kb[valid_idx], best_ka[valid_idx]] > 
                                       base_metric[valid_idx, best_kb[valid_idx], best_ka[valid_idx]])

    df = pd.DataFrame({
        "frame_number": ds["frame_number"].data if "frame_number" in ds.coords else np.arange(T),
        "t": ds["time"].data,
        "BL": BL_mean,
        "metric_cross": best_metric,      # np.nan on skipped frames
        "touch_raw": raw_touch,
        "touch": np.asarray(out.values, dtype=bool),
        "t_param": t_best,
        "u_param": u_best,
        "kpA_idx": APP_idx[best_ka], "kpB_idx": APP_idx[best_kb],
        "kpA": kpA_names,   "kpB": kpB_names,
        "valid_frame": valid_frame,
        "thoraxA_ok": valid_thorA, "thoraxB_ok": valid_thorB,
        "any_appA_ok": valid_appA_any, "any_appB_ok": valid_appB_any,
        "app_dist_best": app_dist_best,
        "won_by_proximity": won_by_proximity,
        
    })

    return out, df

def stack_individuals(
    ds1: xr.Dataset,
    ds2: xr.Dataset,
    ind_names=None,
    join="outer",
) -> xr.Dataset:
    """
    Stack two single-individual movement datasets along 'individuals',
    assuming each already has individuals dim of length 1 (e.g., 'individual_0').

    Relabels each to a unique name before concatenation and excludes
    'individuals' from alignment to avoid unintended expansion.
    """
    # 0) sanity checks
    for i, ds in enumerate((ds1, ds2), start=1):
        if "individuals" not in ds.dims:
            raise ValueError(f"ds{i} has no 'individuals' dim.")
        if ds.sizes["individuals"] != 1:
            raise ValueError(f"ds{i} must have size 1 along 'individuals'.")

    if ind_names is not None:       
        # 1) relabel the single label of each dataset to a unique name
        ds1 = ds1.assign_coords(individuals=[ind_names[0]])
        ds2 = ds2.assign_coords(individuals=[ind_names[1]])

    # 2) align everything EXCEPT the 'individuals' dim
    ds1a, ds2a = xr.align(ds1, ds2, join=join, exclude=["individuals"])

    # 3) concatenate along 'individuals' (now the labels are unique)
    out = xr.concat([ds1a, ds2a], dim="individuals")

    # 4) keep attrs from the first input (optional)
    out.attrs.update(ds1.attrs)
    return out


def add_centroid_offset_single(
    ds: xr.Dataset,
    cx: np.ndarray,          # shape (T,)
    cy: np.ndarray,          # shape (T,)
    space_labels=("x","y"),
    var_name=None            # name of the pose var; auto-detect if None
) -> xr.Dataset:
    """
    ds dims: (time, space, keypoints, individuals) with individuals==1.
    cx, cy: centroid coords per frame (same length as ds.sizes['time']).
    Returns a copy where pose is translated from centroid-relative to image-absolute.
    """
    # pick the pose variable with expected dims
    if var_name is None:
        cand = [v for v in ds.data_vars if all(d in ds[v].dims for d in ("time","space","keypoints","individuals"))]
        if not cand:
            raise ValueError("No pose variable with dims (time, space, keypoints, individuals) found.")
        var_name = cand[0]
    pos = ds[var_name]

    # enforce space ordering (x,y)
    if "space" in pos.coords and set(pos.coords["space"].values) >= {"x","y"}:
        pos = pos.sel(space=["x","y"])
    else:
        # assume current order already matches space_labels
        pass

    # build centroid as DataArray with dims subset of pose: (time, space, individuals)
    cent = xr.DataArray(
        np.stack([cx, cy], axis=1),  # (T, 2)
        dims=("time","space"),
        coords={"time": ds["time"], "space": list(space_labels)},
    ).expand_dims({"individuals": ds["individuals"]})

    pos_abs = pos + cent  # broadcasts over keypoints
    return ds.assign({var_name: pos_abs})


def get_first_video(pose_file):
    with h5py.File(pose_file) as f:
        first_video=os.path.basename(f["files"][0].decode())
    
    return first_video

def get_last_video(pose_file):
    with h5py.File(pose_file) as f:
        last_video=os.path.basename(f["files"][-1].decode())
    
    return last_video

def video_to_chunk(filename):
    chunk=int(filename.split(".")[0])
    return chunk


def get_first_chunk(pose_file):
    return video_to_chunk(get_first_video(pose_file))

def get_last_chunk(pose_file):
    return video_to_chunk(get_last_video(pose_file))

def get_frames(pose_file, chunksize):
    return np.arange(
        get_first_chunk(pose_file)*chunksize,
        (get_last_chunk(pose_file)+1)*chunksize,
        1
    )

def load_params(model):
    params_json=os.path.join(model, "params.json")
    assert os.path.exists(params_json)
    with open(params_json, "r") as handle:
        params=json.load(handle)
    
    return params