"""
label_overrides.py  —  NORMAL env

Rescue true PE bouts that the pipeline labelled `pe_near_food`, using the regularity
of the burst's visible episodes, and record the rescue as a small table NEXT TO the
feather instead of editing the feather:

    pe_bouts/{fly}_pe_bouts.feather            <- pipeline output, never modified
    pe_bouts/{fly}_pe_bouts.overrides.csv      <- fly,burst_id,start_fn,end_fn,label

Readers call `read_pe_bouts(path)`, which applies the table on the fly.

THE RULE
--------
A burst qualifies when its visible episodes (segmentation.burst_event_features) form a
regular train at a sleep-PE period:
    rs_cv_dur      <= MAX_CV_DUR          episodes of consistent duration
    rs_cv_period   <= MAX_CV_PERIOD       evenly spaced
    rs_med_period_s > MIN_MED_PERIOD_S    at a PE-like pace, not rapid food contacts
    rs_n_events    >= MIN_EVENTS          enough episodes that a low CV is not chance
    rs_mean_event_dur_s <= MAX_MEAN_DUR_S (optional)
    frac_pe_or_target  >= MIN_FRAC_PE_OR_TARGET
                                          most of the burst is already PE or would
                                          be rescued (see label_composition)
Every bout in a qualifying burst whose pipeline label is in OVERRIDE_TARGET_LABELS is
relabelled OVERRIDE_LABEL. The rule was validated only on `pe_near_food`, so other
labels are left alone.

Why MIN_EVENTS: with only 3 episodes (2 intervals), randomly timed episodes give
CV <= 0.5 half the time and CV <= 0.3 about 30% of the time. With 5 episodes those
drop to ~20% and ~4%. Genuine trains pass at any count, so this costs little.

WHY THE TABLE IS KEYED ON FRAMES
--------------------------------
Overrides are matched to bouts by (start_fn, end_fn), not burst_id. burst_id is
regenerated whenever grouping changes; frame numbers only change if segmentation does.
If the feather is regenerated and a bout no longer exists, its override simply fails
to match (and is reported) instead of silently relabelling a different bout.
burst_id is still written to the table, for reading it by eye.
"""
import os
import logging
import numpy as np
import pandas as pd

from .segmentation import burst_event_features

logger = logging.getLogger(__name__)

# ---- the rule (set these from the audit; see the check in the chat) -------
OVERRIDE_TARGET_LABELS = ("pe_near_food",)
OVERRIDE_LABEL = "pe"
MAX_CV_DUR = 0.20
MAX_CV_PERIOD = 0.20
MIN_MED_PERIOD_S = 2.5
MIN_EVENTS = 4
MAX_MEAN_DUR_S = None        # e.g. 2.0 to also cap mean episode duration; None = off
MIN_FRAC_PE_OR_TARGET = 0.80 # share of the burst's bouts that are pe or target labels
FRAC_BASIS = "count"         # "count": share of bouts; "duration": share of bout time

OVERRIDE_COLUMNS = ["fly", "burst_id", "start_fn", "end_fn", "label"]
TRACE_COLUMNS = ["burst_id", "frame_number", "dist_mm", "prob_conf"]


def label_composition(bouts):
    """How much of each burst the pipeline already calls PE or would have rescued.

    Computed from the pipeline's RAW labels in the pe_bouts feather (never from a
    feather with overrides applied). For each burst:

        n_bouts   every pipeline bout in the burst, whatever its label  (denominator)
        n_pe      bouts already labelled OVERRIDE_LABEL
        n_target  bouts labelled one of OVERRIDE_TARGET_LABELS — in a qualifying burst,
                  every one of these would be overridden
        frac_pe_or_target      (n_pe + n_target) / n_bouts
        frac_pe_or_target_dur  the same share, but of summed bout DURATION (dur_s).
                               Less sensitive to fragmentation: an episode the pipeline
                               split into 10 bouts adds 10 to n_bouts, but only its own
                               duration to the time total.

    Parameters
    ----------
    bouts : pd.DataFrame
        Pipeline bouts with ``burst_id``, ``label`` and ideally ``dur_s``. May hold one
        fly or several; with a ``fly`` column the grouping is per (fly, burst_id).

    Returns
    -------
    pd.DataFrame
        One row per burst, keyed by ``burst_id`` (and ``fly`` when present).
    """
    keys = [k for k in ("fly", "burst_id") if k in bouts.columns]
    yes = bouts["label"].isin((OVERRIDE_LABEL,) + tuple(OVERRIDE_TARGET_LABELS))
    dur = (pd.to_numeric(bouts["dur_s"], errors="coerce") if "dur_s" in bouts.columns
           else pd.Series(np.nan, index=bouts.index))
    b = bouts[keys].assign(
        _pe=(bouts["label"] == OVERRIDE_LABEL).astype(int),
        _target=bouts["label"].isin(OVERRIDE_TARGET_LABELS).astype(int),
        _dur=dur, _yes_dur=dur.where(yes, 0.0))
    comp = (b.groupby(keys)
              .agg(n_bouts=("_pe", "size"), n_pe=("_pe", "sum"),
                   n_target=("_target", "sum"),
                   _dur_all=("_dur", "sum"), _dur_yes=("_yes_dur", "sum"))
              .reset_index())
    comp["frac_pe_or_target"] = (comp["n_pe"] + comp["n_target"]) / comp["n_bouts"]
    comp["frac_pe_or_target_dur"] = np.where(comp["_dur_all"] > 0,
                                             comp["_dur_yes"] / comp["_dur_all"], np.nan)
    return comp.drop(columns=["_dur_all", "_dur_yes"])


def rule_mask(df):
    """Boolean Series: which rows (bursts, or bouts carrying burst-level columns)
    satisfy the rule. NaN features never pass, because NaN comparisons are False.

    Needs the rs_* features AND the label fraction from ``label_composition``. The
    fraction is REQUIRED rather than skipped when missing: silently dropping it would
    mean validating one rule on the audit and deploying another.
    """
    frac_col = ("frac_pe_or_target" if FRAC_BASIS == "count"
                else "frac_pe_or_target_dur")
    if frac_col not in df.columns:
        raise KeyError(f"rule needs '{frac_col}' — merge label_composition(bouts) first")
    m = ((df["rs_cv_dur"] <= MAX_CV_DUR)
         & (df["rs_cv_period"] <= MAX_CV_PERIOD)
         & (df["rs_med_period_s"] > MIN_MED_PERIOD_S)
         & (df["rs_n_events"] >= MIN_EVENTS)
         & (df[frac_col] >= MIN_FRAC_PE_OR_TARGET))
    if MAX_MEAN_DUR_S is not None:
        m &= df["rs_mean_event_dur_s"] <= MAX_MEAN_DUR_S
    return m.fillna(False).astype(bool)


def overrides_path(bouts_feather):
    """pe_bouts/{fly}_pe_bouts.feather -> pe_bouts/{fly}_pe_bouts.overrides.csv"""
    root, _ = os.path.splitext(bouts_feather)
    return root + ".overrides.csv"


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------
def compute_overrides(bouts, traces, fly, fps):
    """Apply the rule to one fly.

    Parameters
    ----------
    bouts : pd.DataFrame
        The RAW pipeline feather (never one with overrides already applied).
    traces : pd.DataFrame
        The fly's traces feather, at least TRACE_COLUMNS.
    fly : str
    fps : float

    Returns
    -------
    (overrides, burst_features)
        overrides : one row per relabelled bout, OVERRIDE_COLUMNS
        burst_features : the rs_* features of every candidate burst plus a
                         ``qualifies`` column, for inspection
    """
    cand = bouts[bouts["label"].isin(OVERRIDE_TARGET_LABELS)]
    bursts = set(cand["burst_id"].astype(int))
    tr = traces[traces["burst_id"].isin(bursts)]

    rows = []
    for bid, g in tr.groupby("burst_id"):
        r = burst_event_features(g, fps)
        if r is not None:
            rows.append(dict(burst_id=int(bid), **r["features"]))
    if not rows:
        return pd.DataFrame(columns=OVERRIDE_COLUMNS), pd.DataFrame()

    fb = pd.DataFrame(rows)
    comp = label_composition(bouts).drop(columns=["fly"], errors="ignore")
    fb = fb.merge(comp, on="burst_id", how="left")
    fb["qualifies"] = rule_mask(fb)
    ok = set(fb.loc[fb["qualifies"], "burst_id"])

    out = cand[cand["burst_id"].astype(int).isin(ok)][["burst_id", "start_fn", "end_fn"]]
    out = out.astype({"burst_id": int, "start_fn": int, "end_fn": int})
    out.insert(0, "fly", fly)
    out["label"] = OVERRIDE_LABEL
    return out[OVERRIDE_COLUMNS].reset_index(drop=True), fb


def write_overrides(fly, output="."):
    """Compute and write the override table for one fly.

    Uses the same layout as extract_burst_traces:
        {output}/pe_bouts/{fly}_pe_bouts.feather
        {output}/{fly}_traces.feather
    The table is ALWAYS written, even when empty, so its presence tells readers the
    rule ran for this fly; a missing file means it never ran.
    """
    from flyhostel.utils import get_framerate

    bouts_path = os.path.join(output, "pe_bouts", f"{fly}_pe_bouts.feather")
    traces_path = os.path.join(output, f"{fly}_traces.feather")
    for p in (bouts_path, traces_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} — run pe_features and extract_burst_traces first")

    fps = float(get_framerate(fly.rsplit("__", 1)[0]))
    bouts = pd.read_feather(bouts_path)                       # RAW, never overridden
    traces = pd.read_feather(traces_path, columns=TRACE_COLUMNS)

    out, fb = compute_overrides(bouts, traces, fly, fps)
    path = overrides_path(bouts_path)
    out.to_csv(path, index=False)

    n_cand = int(bouts["label"].isin(OVERRIDE_TARGET_LABELS).sum())
    n_b = int(fb["qualifies"].sum()) if len(fb) else 0
    print(f"  [overrides] {fly}: {len(out)}/{n_cand} {'/'.join(OVERRIDE_TARGET_LABELS)} "
          f"bouts -> {OVERRIDE_LABEL} ({n_b} bursts) -> {path}")
    return path


# --------------------------------------------------------------------------
# read
# --------------------------------------------------------------------------
def apply_overrides(df, ov):
    """Return a copy of ``df`` with the override labels applied.

    Adds ``label_pipeline`` (the original label, preserved) and ``label_source``
    ("override" or "pipeline"). Matching is on (start_fn, end_fn).

    Returns
    -------
    (df, n_unmatched)
        n_unmatched : override rows that match no bout (a stale table).
    """
    df = df.copy()
    if "label_pipeline" not in df.columns:
        df["label_pipeline"] = df["label"]
    key = list(zip(df["start_fn"].astype(int), df["end_fn"].astype(int)))
    new = dict(zip(zip(ov["start_fn"].astype(int), ov["end_fn"].astype(int)),
                   ov["label"]))
    hit = np.array([k in new for k in key], dtype=bool)
    if "label_reason" in df.columns:
        df.loc[hit, "label_reason"] = [
            f"rescued by label_overrides: regular train of visible episodes at a PE "
            f"pace (pipeline said {orig})"
            for orig in df.loc[hit, "label_pipeline"]]
    df.loc[hit, "label"] = [new[k] for k, h in zip(key, hit) if h]
    df["label_source"] = np.where(hit, "override", "pipeline")
    n_unmatched = len(set(new) - set(key))
    return df, n_unmatched


def read_pe_bouts(path, apply=True):
    """Read a pe_bouts feather, applying its override table when one exists.

    Use this wherever labels feed the BIOLOGY or the GUI. Do NOT use it (or pass
    ``apply=False``) when building training data for pnf_splitter or the audits:
    those must see the pipeline's own labels, otherwise rescued bouts drop out of
    the `pe_near_food` population the rule is evaluated on.

    Note: burst-level columns derived from labels (n_pe, frac_pe, burst_label,
    burst_pe_score) are NOT recomputed; they still describe the pipeline labels.
    """
    df = pd.read_feather(path)
    if not apply:
        return df
    op = overrides_path(path)
    if not os.path.exists(op):
        return df
    ov = pd.read_csv(op)
    if ov.empty:
        df = df.copy()
        df["label_pipeline"] = df["label"]
        df["label_source"] = "pipeline"
        return df
    df, n_unmatched = apply_overrides(df, ov)
    if n_unmatched:
        logger.warning("%s: %d override rows match no bout — the table predates the "
                       "current feather; rerun the override step", op, n_unmatched)
    return df