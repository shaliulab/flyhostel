"""
annotations.py

Verdicts given to the PE bouts of a fly in the GUI (idtrackerai-validator-server, pe_validation.py). They are kept in
the shared pe_annotations.db, one row per bout:

    pe_annotations(experiment, identity, start_frame, end_frame, burst_id, bout_uid, verdict, pe_score, reviewer, reviewed_at)

experiment is the folder of the experiment relative to the videos folder (FlyHostel1/2X/2025-02-04_16-00-00, or
FlyHostel1/4X/2026-10-13_17-00-00/arenas/2 for an arena) and (start_frame, end_frame) are the (start_fn, end_fn) of
the bout in the pe_bouts feather, so verdicts are matched to bouts like the override table (label_overrides.py).

The verdict of the user has the last word: read_pe_bouts applies the overrides, apply_annotations then applies the
verdicts on top.
"""
import logging
import os
import sqlite3

import numpy as np
import pandas as pd

from flyhostel.utils.experiment import experiment_to_relpath

logger = logging.getLogger(__name__)

# the GUI reads PE_ANNOTATIONS_DB, segmentation.py PE_DB
ANNOTATIONS_DB = os.environ.get("PE_ANNOTATIONS_DB") or os.environ.get("PE_DB", "/flyhostel_data/videos/pe_annotations.db")
# verdicts that relabel the bout; merge and unsure keep the label it had
RELABELLING_VERDICTS = ("pe", "feed", "groom", "walk", "other")


def read_annotations(fly, db=None):
    """Verdicts of the bouts of a fly (experiment__NN), with columns start_fn, end_fn, verdict.

    Empty if the database does not exist (the GUI was never used) or has no verdicts for the fly.
    """
    db = db or ANNOTATIONS_DB
    columns = ["start_fn", "end_fn", "verdict"]
    if not os.path.exists(db):
        logger.warning("%s not found: no verdicts of the GUI are applied", db)
        return pd.DataFrame(columns=columns)

    experiment, identity = fly.rsplit("__", 1)
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as con:
        verdicts = pd.read_sql_query(
            "SELECT start_frame AS start_fn, end_frame AS end_fn, verdict FROM pe_annotations "
            "WHERE experiment=? AND identity=?",
            con, params=(experiment_to_relpath(experiment), int(identity)),
        )
    return verdicts[columns]


def apply_annotations(df, fly, db=None):
    """Return a copy of a pe_bouts table (ideally from read_pe_bouts) with the verdicts of the GUI applied.

    Adds label_pipeline (the label before any override or verdict) if missing and sets label_source to "gui"
    where a verdict relabelled the bout.
    """
    df = df.copy()
    if "label_pipeline" not in df.columns:
        df["label_pipeline"] = df["label"]
    if "label_source" not in df.columns:
        df["label_source"] = "pipeline"

    verdicts = read_annotations(fly, db=db)
    verdicts = verdicts.loc[verdicts["verdict"].isin(RELABELLING_VERDICTS)]
    if verdicts.empty:
        return df

    new = dict(zip(zip(verdicts["start_fn"].astype(int), verdicts["end_fn"].astype(int)), verdicts["verdict"]))
    key = list(zip(df["start_fn"].astype(int), df["end_fn"].astype(int)))
    hit = np.array([k in new for k in key], dtype=bool)
    df.loc[hit, "label"] = [new[k] for k, h in zip(key, hit) if h]
    df.loc[hit, "label_source"] = "gui"

    n_unmatched = len(set(new) - set(key))
    if n_unmatched:
        logger.warning("%s: %d verdicts of the GUI match no bout — the bouts were segmented again after the review",
                       fly, n_unmatched)
    logger.info("%s: %d bouts relabelled by the GUI", fly, hit.sum())
    return df
