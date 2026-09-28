"""
segmentation.py  —  NORMAL env
 
ONE segmentation method, used by the GUI, the benchmark and pnf_splitter:
 
  visible : one bout per episode in which the proboscis is VISIBLE, from the per-frame
            proboscis confidence.
              * enter when confidence rises above HI, leave only when it falls below LO
                (hysteresis: a confidence wobbling around 0.5 stays in one state)
              * fill short NaN gaps
              * merge episodes separated by less than BRIDGE_S  — real PE intervals
                are >= ~2 s, so bridging below that cannot merge two genuine PEs
              * drop episodes shorter than MIN_S (single-frame flickers)
 
  pipeline : the pipeline's own bouts (bout_in_burst in the trace). Not a proposal —
             only the BASELINE the benchmark compares against.
 
Why confidence and not distance: the question is "is the proboscis visible", which is
what the detector's confidence measures. The distance only exists while the proboscis
is detected, carries the dips that shattered bouts, and needs a per-fly mm threshold.
Confidence is unitless, so the same parameters apply on 47 and 150 fps rigs.
One assumption to verify on your data: a RETRACTED proboscis must get low confidence.
If some flies show a retracted tip with confidence above HI, whole bursts will read as
one long visible episode — the ground truth will show it immediately.
 
STORAGE (the shared pe_annotations.db; frames GLOBAL, end EXCLUSIVE; keyed on `fly`)
  auto_runs / auto_segments   algorithm output, for display in the GUI
  gt_windows / gt_segments    human ground truth
 
CLI
  python -m segmentation write --experiments A B                 # method 'visible'
  python -m segmentation write --experiments A B --method pipeline
  python -m segmentation benchmark
"""
import os
import json
import sqlite3
import argparse
import datetime
import itertools
import numpy as np
import pandas as pd
 
DB_PATH = os.environ.get("RESEG_DB",
                         os.environ.get("PE_DB", "/flyhostel_data/videos/pe_annotations.db"))
PROBOSCIS_EXTENSIONS_FOLDER = "proboscis_extensions"
 
VISIBLE_DEFAULTS = dict(hi=0.50, lo=0.30, nanfill_s=0.10, bridge_s=0.40, min_s=0.05)
METHODS = ["visible", "pipeline"]
 
 
# ==========================================================================
# the method
# ==========================================================================
def _runs(mask):
    """Contiguous True runs of a boolean mask, as index intervals.
 
    Parameters
    ----------
    mask : np.ndarray of bool
        One entry per frame. Must have dtype bool: the function reinterprets it as
        int8 (``mask.view(np.int8)``), which only works for 1-byte booleans.
 
    Returns
    -------
    list of (int, int)
        ``(start, end)`` for each run, with ``end`` EXCLUSIVE, so ``mask[start:end]``
        is exactly the run. Empty list if the mask has no True entries.
 
    Example
    -------
    >>> _runs(np.array([0, 1, 1, 0, 1], bool))
    [(1, 3), (4, 5)]
    """
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))
 
 
def _bridge(mask, n):
    """Merge True runs separated by short False gaps.
 
    A gap is filled only when it lies BETWEEN two True runs and is at most ``n``
    frames long. Leading and trailing False stretches are never filled, so bridging
    cannot extend an episode past its first or last visible frame.
 
    Parameters
    ----------
    mask : np.ndarray of bool
        One entry per frame.
    n : int
        Longest gap, in frames, that gets filled.
 
    Returns
    -------
    np.ndarray of bool
        A modified copy; the input is left untouched.
 
    Example
    -------
    >>> _bridge(np.array([1, 0, 0, 1, 0, 0, 0, 1], bool), 2).astype(int)
    array([1, 1, 1, 1, 0, 0, 0, 1])
    """
    m = mask.copy()
    d = np.diff(np.concatenate([[0], m.view(np.int8), [0]]))
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    for e, s in zip(en[:-1], st[1:]):
        if s - e <= n:
            m[e:s] = True
    return m
 
 
def _fill_short_nans(x, max_len):
    """Linearly interpolate short NaN gaps; leave long ones as NaN.
 
    Meant for brief tracking dropouts inside an otherwise continuous signal: a run
    of at most ``max_len`` NaNs is replaced by the straight line between its two
    neighbours. Longer runs are kept as NaN, because a long absence is information
    (the proboscis is genuinely not detected), not a glitch.
 
    Only INTERIOR runs are filled (``interpolate(limit_area="inside")``): NaNs at the
    very start or end of ``x`` have a neighbour on one side only and stay NaN even
    if short.
 
    Parameters
    ----------
    x : array-like of float
        One value per frame; NaN marks missing frames.
    max_len : int
        Longest NaN run, in frames, that gets filled. ``<= 0`` disables filling and
        returns ``x`` unchanged (the same object, not a copy).
 
    Returns
    -------
    np.ndarray of float
    """
    if max_len <= 0:
        return x
    s = pd.Series(x)
    isn = s.isna()
    run_len = isn.groupby((isn != isn.shift()).cumsum()).transform("size")
    return np.where(isn & (run_len <= max_len), s.interpolate(limit_area="inside"), x)
 
 
def seg_visible(conf, fps, hi=0.50, lo=0.30, nanfill_s=0.10, bridge_s=0.40, min_s=0.05):
    """Segment a confidence trace into episodes in which the proboscis is VISIBLE.
 
    One episode per continuous period of visibility, regardless of what the
    proboscis is doing (sleep PE, feeding, grooming...). Designed so that one real
    episode is not shattered into several evenly spaced fragments, which would give
    a falsely low coefficient of variation downstream.
 
    Steps
    -----
    1. Fill NaN runs of up to ``nanfill_s`` seconds by interpolation (dropouts).
    2. Hysteresis (a Schmitt trigger) on the confidence:
         conf >  hi              -> visible
         conf <  lo, or NaN      -> not visible
         lo <= conf <= hi        -> keep the previous state
       so a confidence wobbling around 0.5 does not flip the state back and forth.
       Frames at the start of the trace that fall between ``lo`` and ``hi`` have no
       previous state and count as not visible.
    3. Merge episodes separated by gaps of up to ``bridge_s`` seconds. Real sleep-PE
       intervals are around 2 s or more, so this cannot join two genuine PEs.
    4. Drop episodes shorter than ``min_s`` seconds (single-frame flickers).
 
    Durations are converted to frames with ``round(seconds * fps)``; ``bridge_s``
    and ``min_s`` are floored at 1 frame. At 47 fps the defaults are: dropout fill
    5 frames, bridge 19 frames, minimum episode 2 frames.
 
    Parameters
    ----------
    conf : array-like of float
        Per-frame proboscis confidence for ONE contiguous window (e.g. one burst's
        trace). Values are compared directly against ``hi``/``lo``.
    fps : float
        Frame rate of the recording.
    hi : float
        Confidence above which an episode starts.
    lo : float
        Confidence below which an episode ends. Must be < ``hi``.
    nanfill_s : float
        Longest NaN gap (s) interpolated in step 1.
    bridge_s : float
        Longest gap (s) merged in step 3.
    min_s : float
        Shortest episode (s) kept in step 4.
 
    Returns
    -------
    list of (int, int)
        ``(start, end)`` indices into ``conf`` with ``end`` EXCLUSIVE, in time order.
        Empty list if the proboscis is never visible.
    """
    x = _fill_short_nans(np.asarray(conf, float), round(nanfill_s * fps))
    state = np.where(x > hi, 1.0, np.where((x < lo) | ~np.isfinite(x), 0.0, np.nan))
    vis = pd.Series(state).ffill().fillna(0).to_numpy().astype(bool)
    vis = _bridge(vis, max(1, round(bridge_s * fps)))
    mn = max(1, round(min_s * fps))
    return [(s, e) for s, e in _runs(vis) if e - s >= mn]
 
 
def seg_pipeline(bib):
    """Recover the PE pipeline's own bouts from a trace window. Baseline only.
 
    The pipeline's segmentation is not recomputed here; it is read back from the
    ``bout_in_burst`` column of the traces feather, which holds each frame's bout
    number within its burst and NaN outside bouts. This includes the effect of the
    pipeline's plausibility masks, which the raw trace does not carry, so it is the
    correct baseline to benchmark ``seg_visible`` against.
 
    A bout is a run of consecutive frames sharing the same non-NaN value. Two bouts
    that touch with no NaN between them are still kept apart, because their values
    differ.
 
    Parameters
    ----------
    bib : array-like of float
        ``bout_in_burst`` for one contiguous window, one value per frame.
 
    Returns
    -------
    list of (int, int)
        ``(start, end)`` indices into ``bib`` with ``end`` EXCLUSIVE, in time order.
    """
    b = pd.Series(np.asarray(bib, dtype=float))
    ok = b.notna().to_numpy()
    run = ((b != b.shift()) | (b.notna() != b.shift().notna())).cumsum().to_numpy()
    out = []
    for r in np.unique(run[ok]):
        idx = np.flatnonzero(ok & (run == r))
        out.append((int(idx.min()), int(idx.max()) + 1))
    return sorted(out)
 
 
def run_method(method, w, params):
    """Run a segmentation method on one trace window.
 
    Single entry point used by ``write_auto`` and the benchmark, so both always
    apply a method the same way.
 
    Parameters
    ----------
    method : {"visible", "pipeline"}
        ``"visible"`` runs ``seg_visible`` on the confidence; ``"pipeline"`` reads
        back the pipeline's own bouts with ``seg_pipeline``.
    w : dict
        The window's per-frame data, as built by ``window_arrays``:
        ``conf`` (confidence array), ``bib`` (bout_in_burst array), ``fps``.
    params : dict
        Keyword arguments for ``seg_visible`` (hi, lo, nanfill_s, bridge_s, min_s).
        Ignored for ``"pipeline"``.
 
    Returns
    -------
    list of (int, int)
        ``(start, end)`` indices into the window, ``end`` EXCLUSIVE.
 
    Raises
    ------
    ValueError
        If ``method`` is not recognised.
    """
    if method == "visible":
        return seg_visible(w["conf"], w["fps"], **params)
    if method == "pipeline":
        return seg_pipeline(w["bib"])
    raise ValueError(f"unknown method {method}")
 
 
# ==========================================================================
# per-burst features over visible episodes
# ==========================================================================
# ONE implementation, used by pnf_splitter (to build/validate the rule on the audit)
# and by label_overrides (to apply it fleet-wide), so the deployed rule sees exactly
# the numbers it was validated on.
PROD_THRESH = 0.50      # per-frame confidence gate used for the distance (amplitude)
PLATEAU_S = 3.0         # an episode longer than this is a held extension
 
 
def _cv(a):
    """Coefficient of variation (std / mean, population std) of the finite values.
 
    NaN when fewer than 2 finite values remain or the mean is 0, because a CV from a
    single value is meaningless.
    """
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(a.std() / a.mean()) if len(a) >= 2 and a.mean() else np.nan
 
 
def burst_event_features(g, fps, plateau_s=PLATEAU_S):
    """Segment one burst's trace into visible episodes and summarise them.
 
    Parameters
    ----------
    g : pd.DataFrame
        The burst's rows from the traces feather: ``frame_number``, ``prob_conf``,
        ``dist_mm``. One row per frame; need not be sorted.
    fps : float
        Frame rate of the recording.
    plateau_s : float
        Episodes longer than this count as held extensions, not PE candidates.
 
    Returns
    -------
    dict or None
        None if the burst has no visible episode. Otherwise:
          ``features`` : dict of burst-level ``rs_*`` values
              rs_n_events, rs_n_short_events   episode counts (all / <= plateau_s)
              rs_max_event_dur_s, rs_median_event_dur_s, rs_mean_event_dur_s
              rs_frac_extended                 time visible / window length
              rs_med_period_s                  median onset-to-onset interval
              rs_cv_period, rs_cv_dur, rs_cv_amp   regularity (NaN if < 2 values)
              rs_stereotyped                   >= 3 short episodes, CVs <= 0.5
          ``ev_start``, ``ev_end`` : global frame numbers, ``ev_end`` EXCLUSIVE
          ``dur`` : episode durations in seconds
    """
    g = g.sort_values("frame_number")
    fr = g["frame_number"].to_numpy()
    conf = g["prob_conf"].to_numpy(dtype=float)
    dist = np.where(conf >= PROD_THRESH, g["dist_mm"].to_numpy(dtype=float), np.nan)
 
    ev = seg_visible(conf, fps, **VISIBLE_DEFAULTS)
    if not ev:
        return None
    ev_start = np.array([fr[s] for s, _ in ev])
    ev_end = np.array([fr[e - 1] + 1 for _, e in ev])
    dur = (ev_end - ev_start) / fps
    amp = np.array([np.nanmax(dist[s:e]) if np.isfinite(dist[s:e]).any() else np.nan
                    for s, e in ev])
    period = np.diff(ev_start) / fps
    win = (fr[-1] - fr[0] + 1) / fps
    short = dur <= plateau_s
 
    features = dict(
        rs_n_events=len(ev),
        rs_n_short_events=int(short.sum()),
        rs_max_event_dur_s=float(dur.max()),
        rs_median_event_dur_s=float(np.median(dur)),
        rs_mean_event_dur_s=float(dur.mean()),
        rs_frac_extended=float(dur.sum() / win),
        rs_med_period_s=float(np.median(period)) if len(period) else np.nan,
        rs_cv_period=_cv(period),
        rs_cv_dur=_cv(dur),
        rs_cv_amp=_cv(amp),
        rs_stereotyped=float(short.sum() >= 3
                             and _cv(np.diff(ev_start) / fps) <= 0.5
                             and _cv(dur) <= 0.5),
    )
    return dict(features=features, ev_start=ev_start, ev_end=ev_end, dur=dur)
 
 
def link_bouts_to_events(pb, ev_start, ev_end, dur, fps):
    """Attach each pipeline bout to the visible episode it overlaps most.
 
    Parameters
    ----------
    pb : pd.DataFrame
        The burst's pipeline bouts, with ``start_fn`` and ``end_fn``. Its index is
        returned as ``_idx`` so the caller can join the rows back.
    ev_start, ev_end, dur, fps
        As returned by ``burst_event_features``.
 
    Returns
    -------
    list of dict
        One per bout that overlaps an episode: ``_idx``, ``rs_event_dur_s``,
        ``rs_bouts_in_event`` (how many pipeline bouts that episode was split into),
        ``rs_gap_prev_s``, ``rs_gap_next_s`` (gaps to the neighbouring EPISODES).
        Bouts overlapping no episode are omitted.
    """
    n_per_event = np.zeros(len(ev_start), int)
    links = []
    for r in pb.itertuples():
        ov = np.minimum(ev_end, r.end_fn) - np.maximum(ev_start, r.start_fn)
        j = int(ov.argmax())
        if ov[j] > 0:
            n_per_event[j] += 1
            links.append((r.Index, j))
    return [dict(_idx=idx,
                 rs_event_dur_s=float(dur[j]),
                 rs_bouts_in_event=int(n_per_event[j]),
                 rs_gap_prev_s=float((ev_start[j] - ev_end[j - 1]) / fps) if j > 0
                               else np.nan,
                 rs_gap_next_s=float((ev_start[j + 1] - ev_end[j]) / fps)
                               if j + 1 < len(ev_start) else np.nan)
            for idx, j in links]
 
 
# ==========================================================================
# traces
# ==========================================================================
def traces_for_fly(fly):
    from flyhostel.utils import get_basedir
    experiment, _ = fly.rsplit("__", 1)
    return (f"{get_basedir(experiment)}/flyhostel/{PROBOSCIS_EXTENSIONS_FOLDER}/"
            f"{fly}_traces.feather")
 
 
def load_traces(fly):
    p = traces_for_fly(fly)
    if not os.path.exists(p):
        return None
    cols = ["burst_id", "frame_number", "dist_mm", "prob_conf", "bout_in_burst"]
    return pd.read_feather(p, columns=cols).sort_values(["burst_id", "frame_number"])
 
 
def fps_for(fly):
    from flyhostel.utils import get_framerate
    return float(get_framerate(fly.rsplit("__", 1)[0]))
 
 
def window_arrays(g, fps):
    return dict(conf=g["prob_conf"].to_numpy(dtype=float),
                bib=g["bout_in_burst"].to_numpy(dtype=float), fps=fps)
 
 
# ==========================================================================
# database
# ==========================================================================
SCHEMA = """
CREATE TABLE IF NOT EXISTS auto_runs (
    method TEXT NOT NULL, fly TEXT NOT NULL, params TEXT NOT NULL,
    ext_min_mm REAL, written_at TEXT NOT NULL,
    PRIMARY KEY (method, fly));
CREATE TABLE IF NOT EXISTS auto_segments (
    method TEXT NOT NULL, fly TEXT NOT NULL,
    start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL,
    PRIMARY KEY (method, fly, start_frame));
CREATE TABLE IF NOT EXISTS gt_windows (
    fly TEXT NOT NULL, win_start INTEGER NOT NULL, win_end INTEGER NOT NULL,
    annotator TEXT, done_at TEXT NOT NULL,
    PRIMARY KEY (fly, win_start, win_end));
CREATE TABLE IF NOT EXISTS gt_segments (
    fly TEXT NOT NULL, start_frame INTEGER NOT NULL, end_frame INTEGER NOT NULL,
    annotator TEXT, created_at TEXT NOT NULL,
    PRIMARY KEY (fly, start_frame, end_frame));
"""
 
 
def init_db(path=DB_PATH):
    with sqlite3.connect(path) as c:
        c.executescript(SCHEMA)
 
 
def write_auto(flies, method="visible", params=None, db_path=DB_PATH):
    init_db(db_path)
    if method == "visible" and params is None:
        params = dict(VISIBLE_DEFAULTS)
    params = params or {}
    now = datetime.datetime.utcnow().isoformat()
    for fly in flies:
        tr = load_traces(fly)
        if tr is None:
            print(f"  no traces feather: {fly}")
            continue
        fps = fps_for(fly)
        rows = []
        for _, g in tr.groupby("burst_id"):
            fr = g["frame_number"].to_numpy()
            for s, e in run_method(method, window_arrays(g, fps), params):
                rows.append((method, fly, int(fr[s]), int(fr[e - 1]) + 1))
        with sqlite3.connect(db_path) as c:
            c.execute("DELETE FROM auto_segments WHERE method=? AND fly=?", (method, fly))
            c.executemany("INSERT OR REPLACE INTO auto_segments VALUES (?,?,?,?)", rows)
            c.execute("INSERT OR REPLACE INTO auto_runs VALUES (?,?,?,?,?)",
                      (method, fly, json.dumps(params, sort_keys=True), None, now))
        print(f"  {fly}: {len(rows)} segments  (fps {fps:g})")
 
 
# ==========================================================================
# benchmark
# ==========================================================================
def match_events(gt, pr, iou_thr=0.5):
    """One-to-one greedy matching by IoU. gt/pr: lists of (start, end_exclusive)."""
    pairs = []
    for i, (gs, ge) in enumerate(gt):
        for j, (ps, pe) in enumerate(pr):
            inter = min(ge, pe) - max(gs, ps)
            if inter > 0:
                pairs.append((inter / (max(ge, pe) - min(gs, ps)), i, j))
    pairs.sort(reverse=True)
    ug, up, matched = set(), set(), []
    for iou, i, j in pairs:
        if iou < iou_thr:
            break
        if i in ug or j in up:
            continue
        ug.add(i); up.add(j); matched.append((i, j))
    ov = lambda a, b: min(a[1], b[1]) - max(a[0], b[0]) > 0
    frag = [sum(ov(g, p) for p in pr) for g in gt]
    merge = [sum(ov(p, g) for g in gt) for p in pr]
    return dict(tp=len(matched), fp=len(pr) - len(matched), fn=len(gt) - len(matched),
                frag=[f for f in frag if f > 0], merge=[m for m in merge if m > 0],
                onset=[abs(pr[j][0] - gt[i][0]) for i, j in matched],
                offset=[abs(pr[j][1] - gt[i][1]) for i, j in matched])
 
 
def load_gt_windows(db_path=DB_PATH):
    with sqlite3.connect(db_path) as c:
        win = pd.read_sql("SELECT fly, win_start, win_end FROM gt_windows", c)
        seg = pd.read_sql("SELECT fly, start_frame, end_frame FROM gt_segments", c)
    if win.empty:
        raise SystemExit("no gt_windows marked done — segment some windows first")
    out = []
    for fly, wf in win.groupby("fly"):
        tr = load_traces(fly)
        if tr is None:
            print(f"  no traces feather for GT fly {fly} — skipped")
            continue
        fps = fps_for(fly)
        tr = tr.drop_duplicates("frame_number").set_index("frame_number")
        for w in wf.itertuples():
            sub = tr.loc[w.win_start:w.win_end - 1]
            if len(sub) < w.win_end - w.win_start:
                print(f"  {fly} window {w.win_start}-{w.win_end}: trace incomplete — skipped")
                continue
            ev = seg[(seg["fly"] == fly) & (seg["end_frame"] > w.win_start)
                     & (seg["start_frame"] < w.win_end)]
            d = window_arrays(sub.reset_index(), fps)
            d.update(fly=fly, gt=[(max(s, w.win_start) - w.win_start,
                                   min(e, w.win_end) - w.win_start)
                                  for s, e in zip(ev["start_frame"], ev["end_frame"])])
            out.append(d)
    return out
 
 
def configs(quick=False):
    yield ("pipeline", {})                                        # the baseline
    grid = dict(hi=[0.4, 0.5, 0.6], lo=[0.2, 0.3, 0.4],
                nanfill_s=[0.10], bridge_s=[0.10, 0.25, 0.40, 0.60], min_s=[0.02, 0.05])
    if quick:
        grid = dict(hi=[0.5], lo=[0.3], nanfill_s=[0.10], bridge_s=[0.25, 0.40],
                    min_s=[0.05])
    keys = list(grid)
    for vals in itertools.product(*grid.values()):
        p = dict(zip(keys, vals))
        if p["lo"] < p["hi"]:
            yield ("visible", p)
 
 
def score(results):
    tp = sum(r["tp"] for r in results); fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    P = tp / (tp + fp) if tp + fp else np.nan
    R = tp / (tp + fn) if tp + fn else np.nan
    F = 2 * P * R / (P + R) if (P and R) else np.nan
    cat = lambda k: [x for r in results for x in r[k]]
    ms = lambda k: (float(np.median([x * 1000 / r["fps"] for r in results for x in r[k]]))
                    if cat(k) else np.nan)
    return dict(P=P, R=R, F1=F, tp=tp, fp=fp, fn=fn,
                frag=float(np.mean(cat("frag"))) if cat("frag") else np.nan,
                merge=float(np.mean(cat("merge"))) if cat("merge") else np.nan,
                onset_ms=ms("onset"), offset_ms=ms("offset"))
 
 
def run_config(windows, method, params, iou_thr):
    res = []
    for w in windows:
        r = match_events(w["gt"], run_method(method, w, params), iou_thr)
        r["fps"], r["fly"] = w["fps"], w["fly"]
        res.append(r)
    return res
 
 
def benchmark(db_path=DB_PATH, iou_thr=0.5, quick=False, top=15):
    windows = load_gt_windows(db_path)
    flies = sorted({w["fly"] for w in windows})
    print(f"{len(windows)} GT windows | {sum(len(w['gt']) for w in windows)} GT events "
          f"| {len(flies)} flies")
 
    table, per_cfg = [], {}
    for k, (method, params) in enumerate(configs(quick)):
        res = run_config(windows, method, params, iou_thr)
        per_cfg[k] = (method, params, res)
        table.append(dict(cfg=k, method=method,
                          params=json.dumps(params, sort_keys=True), **score(res)))
    t = pd.DataFrame(table).sort_values("F1", ascending=False)
    cols = ["cfg", "method", "F1", "P", "R", "frag", "merge", "onset_ms", "offset_ms", "params"]
 
    print(f"\n=== in-sample leaderboard (IoU >= {iou_thr}) — a MAP, not the result ===")
    print(t[cols].head(top).round(3).to_string(index=False))
    print("\npipeline baseline:")
    print(t[t["method"] == "pipeline"][cols].round(3).to_string(index=False))
 
    if len(flies) >= 3:
        pooled, picks = [], []
        cands = {k: v for k, v in per_cfg.items() if v[0] == "visible"}
        for f in flies:
            best, bestF = None, -1
            for k, (m, p, res) in cands.items():
                F = score([r for r in res if r["fly"] != f])["F1"]
                if F == F and F > bestF:
                    best, bestF = k, F
            picks.append(best)
            pooled += [r for r in cands[best][2] if r["fly"] == f]
        s = score(pooled)
        print(f"\n=== 'visible', parameters chosen leave-one-fly-out (the honest score) ===")
        print(f"  F1 {s['F1']:.3f}  P {s['P']:.3f}  R {s['R']:.3f}  "
              f"frag {s['frag']:.2f}  merge {s['merge']:.2f}")
        for k, n in pd.Series(picks).value_counts().items():
            print(f"    x{n}  {json.dumps(per_cfg[k][1], sort_keys=True)}")
    else:
        print("\n(need >= 3 flies with GT for leave-one-fly-out selection)")
    return t

 

from .utils import read_experiment_list

# ==========================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("write", help="run the method and store its segments for the GUI")
    w.add_argument("--experiments", nargs="*", default=None)
    w.add_argument("--flies", nargs="*", default=None)
    w.add_argument("--method", choices=METHODS, default="visible")
    w.add_argument("--params", default=None,
                   help='JSON overrides, e.g. \'{"bridge_s": 0.25, "lo": 0.2}\'')
    w.add_argument("--db", default=DB_PATH)

    b = sub.add_parser("benchmark", help="compare against human ground truth")
    b.add_argument("--db", default=DB_PATH)
    b.add_argument("--iou", type=float, default=0.5)
    b.add_argument("--quick", action="store_true")
    b.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    if args.cmd == "write":
        if args.flies:
            flies = args.flies
        elif args.experiments:
            experiments = (read_experiment_list(args.experiments[0])
                       if len(args.experiments) == 1
                          and args.experiments[0].endswith(".txt")
                       else args.experiments)
            
            from flyhostel.utils import get_identities
            flies = [f"{e}__{str(i).zfill(2)}" for e in experiments
                     for i in get_identities(e)]
        else:
            raise SystemExit("pass --flies or --experiments")
        params = None
        if args.method == "visible":
            params = dict(VISIBLE_DEFAULTS)
            if args.params:
                params.update(json.loads(args.params))
        write_auto(flies, args.method, params, args.db)
    else:
        benchmark(args.db, args.iou, args.quick, args.top)