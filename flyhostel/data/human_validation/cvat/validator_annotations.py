"""
Human annotations made in idtrackerai-validator's "Annotate" tab instead of CVAT

The validator keeps them in <folder>/validator_annotations.db (folder = <basedir>/flyhostel/validation):

    boxes  (frame_number, x0, y0, x1, y1, label)   coordinates of the original frame
    frames (frame_number, author, updated, n_boxes)

label is a local identity ("1".."N"), a tag of the CVAT vocabulary (FMB, NEW-FRAGMENT, CROSSING)
or "" (not assigned: ignored, like UNASSIGNED boxes in CVAT).

load_validator_annotations returns the same as cvat_integration.get_annotations, so the rest of
integrate_human_annotations does not know where the boxes come from: each box becomes the
rectangle contour CVAT exports for a box, already in original resolution (block size 1).
"""
import datetime
import logging
import os.path
import sqlite3

import numpy as np
import pandas as pd

from flyhostel.data.human_validation.cvat.contour_utils import contour_to_centroid
from flyhostel.data.human_validation.cvat.utils import assign_in_frame_indices
from flyhostel.utils.utils import get_chunksize, get_number_of_animals, get_experiment_identifier

logger = logging.getLogger(__name__)

DB_NAME = "validator_annotations.db"


def validator_db(folder):
    return os.path.join(folder, DB_NAME)


def validator_mtime(folder):
    """When the annotations last changed (datetime, UTC)"""
    path = validator_db(folder)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        updated = conn.execute("SELECT MAX(updated) FROM frames").fetchone()[0]
    if updated is None:
        updated = os.path.getmtime(path)
    return datetime.datetime.fromtimestamp(updated, tz=datetime.timezone.utc)


def load_validator_annotations(experiment, basedir, folder):
    """
    Returns (annotations_df, contours, mtime) as cvat_integration.get_annotations
    """
    path = validator_db(folder)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: no boxes were saved in the validator for {experiment}")

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        boxes = pd.read_sql_query(
            "SELECT b.rowid AS contour_id, b.frame_number, b.x0, b.y0, b.x1, b.y1, b.label "
            "FROM boxes b JOIN frames f ON f.frame_number = b.frame_number "
            "ORDER BY b.frame_number, b.rowid",
            conn,
        )
    logger.info("%s boxes in %s frames read from %s", len(boxes), boxes["frame_number"].nunique(), path)
    chunksize = get_chunksize(experiment)

    parsed_annot = []
    contours = []
    seen_lids = {}
    i = 0
    for row in boxes.itertuples(index=False):
        label = row.label or ""
        if label == "":
            continue
        try:
            local_identity = int(label)
            text = None
        except ValueError:
            local_identity = None
            text = label

        frame_number = int(row.frame_number)
        if local_identity is not None:
            seen = seen_lids.setdefault(frame_number, set())
            if local_identity in seen:
                continue
            seen.add(local_identity)

        polygon = [(row.x0, row.y0), (row.x1, row.y0), (row.x1, row.y1), (row.x0, row.y1)]
        contour = np.array(polygon).reshape((-1, 1, 2)).astype(np.int32)
        x, y = contour_to_centroid(contour)
        # same fields as cvat_integration.load_task_annotations, with one frame per image
        parsed_annot.append((i, frame_number, x, y, local_identity, int(row.contour_id), text, frame_number, 0, 1, 0, 0))
        contours.append(contour)
        i += 1

    annotations_df = pd.DataFrame.from_records(
        parsed_annot,
        columns=["idx", "frame_number", "x", "y", "local_identity", "contour_id", "text",
                 "frame_number0", "block", "block_size", "panel", "frame_idx_in_block"],
    )
    annotations_df["frame_number"] = annotations_df["frame_number"].astype(int)
    annotations_df["chunk"] = annotations_df["frame_number"] // chunksize
    annotations_df["x"] = annotations_df["x"].astype(float)
    annotations_df["y"] = annotations_df["y"].astype(float)
    annotations_df["task"] = 0

    number_of_animals = get_number_of_animals(get_experiment_identifier(basedir))
    annotations_df = assign_in_frame_indices(annotations_df, number_of_animals, experiment=experiment)
    annotations_df["fragment"] = np.nan

    return annotations_df, contours, validator_mtime(folder)


def validator_annotate_url(experiment, frame_number):
    """Link opening the frame in the validator's Annotate tab (None without idtrackerai_validator_server
    or experiment). Base url: $VALIDATOR_PUBLIC_URL (set by the validator for the jobs it starts), else this host."""
    if experiment is None:
        return None
    try:
        from idtrackerai_validator_server.links import annotate_url
    except ImportError:
        return None
    return annotate_url(experiment, frame_number)


def validator_problem_blocks(problems):
    """Runs of consecutive frames with problems: first_frame_number, last_frame_number, n_frames, problems
    (each problem with its number of frames), as report.find_problem_blocks for CVAT frames"""
    columns = ["first_frame_number", "last_frame_number", "n_frames", "problems"]
    if problems.shape[0] == 0:
        return pd.DataFrame(columns=columns)
    frames = problems.groupby("frame_number")["problem"].apply(list).reset_index()
    block_id = (frames["frame_number"].diff() != 1).cumsum()
    blocks = []
    for _, block in frames.groupby(block_id):
        summary = pd.Series([p for ps in block["problem"] for p in ps]).value_counts(sort=False)
        blocks.append((
            int(block["frame_number"].iloc[0]), int(block["frame_number"].iloc[-1]), block.shape[0],
            ", ".join(f"{problem} ({count} frame{'' if count == 1 else 's'})" for problem, count in summary.items()),
        ))
    return pd.DataFrame.from_records(blocks, columns=columns)


def validator_problem_report(out, folder, number_of_animals, pixels_per_mm, max_jump_mm=None, experiment=None, chunksize=None):
    """
    Frames annotated in the validator that still have a problem after the integration
    (an identity missing or duplicated, a blob without identity, optionally a jump),
    saved to <folder>/validator_problem_frames.csv (frame_number, problem), which the Annotate tab lists.
    Same checks as report.cvat_problem_report, on the annotated frames instead of the CVAT frames.
    With experiment, prints each run of consecutive frames with problems with the link opening
    its first frame in the validator's Annotate tab (also in <folder>/validator_problem_blocks.csv).
    """
    from flyhostel.data.human_validation.cvat.report import find_problem_frames

    path = validator_db(folder)
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        annotated = set(int(fn) for (fn,) in conn.execute("SELECT frame_number FROM frames"))
    first_frame_number = out["frame_number"].min()
    last_frame_number = out["frame_number"].max()
    annotated = {fn for fn in annotated if first_frame_number <= fn <= last_frame_number}
    frame_numbers = annotated | {fn - 1 for fn in annotated}
    subset = out.loc[out["frame_number"].isin(frame_numbers)]

    max_jump = None if max_jump_mm is None else pixels_per_mm * max_jump_mm
    if subset.shape[0]:
        problems = find_problem_frames(subset, number_of_animals, max_jump=max_jump)
        problems = problems.loc[problems["frame_number"].isin(annotated)]
    else:
        problems = pd.DataFrame({"frame_number": [], "problem": []})
    empty = sorted(annotated - set(subset["frame_number"].unique()))
    problems = pd.concat([problems, pd.DataFrame({"frame_number": empty, "problem": "no flies"})], axis=0)
    problems["frame_number"] = problems["frame_number"].astype(np.int64)
    problems = problems.sort_values(["frame_number", "problem"]).reset_index(drop=True)
    problems.to_csv(os.path.join(folder, "validator_problem_frames.csv"), index=False)
    print(f"{problems['frame_number'].nunique()} of the {len(annotated)} frames annotated in the validator still have problems "
          f"(validator_problem_frames.csv)")

    blocks = validator_problem_blocks(problems)
    blocks["url"] = [validator_annotate_url(experiment, fn) for fn in blocks["first_frame_number"]]
    blocks.to_csv(os.path.join(folder, "validator_problem_blocks.csv"), index=False)
    for _, row in blocks.iterrows():
        chunk = "" if chunksize is None else f"chunk {row['first_frame_number'] // chunksize}, "
        print(f"Frames {row['first_frame_number']}-{row['last_frame_number']} "
              f"({chunk}{row['n_frames']} frames): {row['problems']}"
              + (f"\n    {row['url']}" if row["url"] else ""))
    return problems
