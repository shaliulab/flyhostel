"""
label_overrides.py  —  NORMAL env

Rescue true PE bouts that the pipeline labelled `pe_near_food`, using the regularity
of the burst's visible episodes, and record the rescue as a small table NEXT TO the
feather instead of editing the feather:

    pe_bouts/{fly}_pe_bouts.feather            <- pipeline output, never modified
    pe_bouts/{fly}_pe_bouts.overrides.csv      <- one row per CANDIDATE bout

The table lists EVERY bout that could have been rescued (pipeline label in
OVERRIDE_TARGET_LABELS), not only the rescued ones, so you can see why each one was or
wasn't. Columns:
    fly, burst_id, start_fn, end_fn, label   label = what the bout BECOMES if it passes
    pass                                     True only if every criterion passes
    has_episodes                             the burst had >= 1 visible episode
    pass_cv_dur, pass_cv_period, pass_med_period, pass_n_events, pass_frac,
    pass_mean_dur                            one boolean per criterion
    rs_n_events, rs_cv_dur, rs_cv_period, rs_med_period_s, rs_mean_event_dur_s,
    n_bouts, n_pe, n_target, frac_pe_or_target, frac_pe_or_target_dur
                                             the values the criteria were applied to

Readers call `read_pe_bouts(path)`, which applies ONLY the rows with pass == True.

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
MAX_CV_DUR = 0.30
MAX_CV_PERIOD = 0.30
MIN_MED_PERIOD_S = 2.5
MIN_EVENTS = 4
MAX_MEAN_DUR_S = None        # e.g. 2.0 to also cap mean episode duration; None = off
MIN_FRAC_PE_OR_TARGET = 0.80 # share of the burst's bouts that are pe or target labels
FRAC_BASIS = "count"         # "count": share of bouts; "duration": share of bout time

CRITERIA = ["pass_cv_dur", "pass_cv_period", "pass_med_period", "pass_n_events",
            "pass_frac", "pass_mean_dur"]
VALUE_COLUMNS = ["rs_n_events", "rs_cv_dur", "rs_cv_period", "rs_med_period_s",
                 "rs_mean_event_dur_s", "n_bouts", "n_pe", "n_target",
                 "frac_pe_or_target", "frac_pe_or_target_dur"]
OVERRIDE_COLUMNS = (["fly", "burst_id", "start_fn", "end_fn", "label", "pass",
                     "has_episodes"] + CRITERIA + VALUE_COLUMNS)
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


def rule_criteria(df):
    """Evaluate each criterion of the rule separately.

    Parameters
    ----------
    df : pd.DataFrame
        Bursts (or bouts carrying burst-level columns) with the rs_* features from
        ``burst_event_features`` and the fractions from ``label_composition``.

    Returns
    -------
    pd.DataFrame of bool, index aligned with ``df``, one column per CRITERIA entry:
        pass_cv_dur      rs_cv_dur       <= MAX_CV_DUR
        pass_cv_period   rs_cv_period    <= MAX_CV_PERIOD
        pass_med_period  rs_med_period_s  > MIN_MED_PERIOD_S
        pass_n_events    rs_n_events     >= MIN_EVENTS
        pass_frac        frac (FRAC_BASIS) >= MIN_FRAC_PE_OR_TARGET
        pass_mean_dur    rs_mean_event_dur_s <= MAX_MEAN_DUR_S  (always True when
                         MAX_MEAN_DUR_S is None, i.e. the criterion is switched off)
    A NaN value FAILS its criterion (NaN comparisons are False): e.g. a burst with a
    single episode has no period, so pass_cv_period and pass_med_period are False.

    The fraction is REQUIRED rather than skipped when missing: silently dropping it
    would mean validating one rule on the audit and deploying another.
    """
    frac_col = ("frac_pe_or_target" if FRAC_BASIS == "count"
                else "frac_pe_or_target_dur")
    if frac_col not in df.columns:
        raise KeyError(f"rule needs '{frac_col}' — merge label_composition(bouts) first")
    c = pd.DataFrame(index=df.index)
    c["pass_cv_dur"] = df["rs_cv_dur"] <= MAX_CV_DUR
    c["pass_cv_period"] = df["rs_cv_period"] <= MAX_CV_PERIOD
    c["pass_med_period"] = df["rs_med_period_s"] > MIN_MED_PERIOD_S
    c["pass_n_events"] = df["rs_n_events"] >= MIN_EVENTS
    c["pass_frac"] = df[frac_col] >= MIN_FRAC_PE_OR_TARGET
    c["pass_mean_dur"] = (True if MAX_MEAN_DUR_S is None
                          else df["rs_mean_event_dur_s"] <= MAX_MEAN_DUR_S)
    return c[CRITERIA].fillna(False).astype(bool)


def rule_mask(df):
    """Boolean Series: rows passing EVERY criterion of ``rule_criteria``."""
    return rule_criteria(df).all(axis=1)


def overrides_path(bouts_feather):
    """pe_bouts/{fly}_pe_bouts.feather -> pe_bouts/{fly}_pe_bouts.overrides.csv"""
    root, _ = os.path.splitext(bouts_feather)
    return root + ".overrides.csv"


# --------------------------------------------------------------------------
# write
# --------------------------------------------------------------------------
def compute_overrides(bouts, traces, fly, fps):
    """Evaluate the rule for every candidate bout of one fly.

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
    (table, burst_table)
        table : one row per CANDIDATE bout (label in OVERRIDE_TARGET_LABELS), with
                OVERRIDE_COLUMNS. ``pass`` marks the ones to relabel.
        burst_table : the same values and criteria at one row per candidate burst,
                      for inspection.
    """
    cand = bouts[bouts["label"].isin(OVERRIDE_TARGET_LABELS)]
    cand = cand[["burst_id", "start_fn", "end_fn"]].astype(int)
    if cand.empty:
        return pd.DataFrame(columns=OVERRIDE_COLUMNS), pd.DataFrame()

    # rs_* features for every candidate burst that has at least one visible episode
    bursts = set(cand["burst_id"])
    rows = []
    for bid, g in traces[traces["burst_id"].isin(bursts)].groupby("burst_id"):
        r = burst_event_features(g, fps)
        if r is not None:
            rows.append(dict(burst_id=int(bid), **r["features"]))
    feats = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["burst_id"])

    # one row per candidate burst: features (NaN if no episodes) + label composition
    bt = pd.DataFrame({"burst_id": sorted(bursts)})
    bt["has_episodes"] = bt["burst_id"].isin(set(feats["burst_id"]))
    bt = bt.merge(feats, on="burst_id", how="left")
    comp = label_composition(bouts).drop(columns=["fly"], errors="ignore")
    bt = bt.merge(comp, on="burst_id", how="left")
    for col in VALUE_COLUMNS:                      # bursts without episodes -> NaN
        if col not in bt.columns:
            bt[col] = np.nan

    crit = rule_criteria(bt)
    bt = pd.concat([bt, crit], axis=1)
    bt["pass"] = crit.all(axis=1)

    table = cand.merge(bt[["burst_id", "pass", "has_episodes"] + CRITERIA + VALUE_COLUMNS],
                       on="burst_id", how="left")
    table.insert(0, "fly", fly)
    table["label"] = OVERRIDE_LABEL
    return table[OVERRIDE_COLUMNS].reset_index(drop=True), bt


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

    table, bt = compute_overrides(bouts, traces, fly, fps)
    path = overrides_path(bouts_path)
    table.to_csv(path, index=False)

    n_pass = int(table["pass"].sum()) if len(table) else 0
    n_bp = int(bt["pass"].sum()) if len(bt) else 0
    print(f"  [overrides] {fly}: {n_pass}/{len(table)} "
          f"{'/'.join(OVERRIDE_TARGET_LABELS)} bouts pass -> {OVERRIDE_LABEL} "
          f"({n_bp}/{len(bt)} bursts) -> {path}")
    if len(bt):
        fails = {c: int((~bt[c]).sum()) for c in CRITERIA}
        print(f"  [overrides]   bursts failing each criterion: {fails}")
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


def _passing(ov, path=""):
    """Keep only the rows to apply: ``pass == True``.

    Parsed as text so a column read back as strings can't slip through
    (``bool("False")`` is True). A table without a ``pass`` column predates this
    format, when only passing rows were written, so all its rows apply.
    """
    if "pass" not in ov.columns:
        logger.info("%s: no 'pass' column (old format) — applying every row", path)
        return ov
    return ov[ov["pass"].astype(str).str.strip().str.lower() == "true"]


def read_pe_bouts(path, apply=True):
    """Read a pe_bouts feather, applying its override table when one exists.

    Only rows with ``pass == True`` are applied; the other rows are there to
    explain the decision, not to relabel anything.

    Use this wherever labels feed the BIOLOGY or the GUI. Do NOT use it (or pass
    ``apply=False``) when building training data for pnf_splitter or the audits:
    those must see the pipeline's own labels, otherwise rescued bouts drop out of
    the `pe_near_food` population the rule is evaluated on.

    Note: burst-level columns derived from labels (n_pe, frac_pe, burst_label,
    burst_pe_score) are NOT recomputed; they still describe the pipeline labels.
    """
    df = pd.read_feather(path)
    assert "t" in df.columns, f"t missing in {path}"

    if not apply:
        return df
    op = overrides_path(path)
    if not os.path.exists(op):
        return df
    ov = _passing(pd.read_csv(op), op)
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