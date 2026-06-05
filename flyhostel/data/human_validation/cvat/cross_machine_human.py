"""
flyhostel/data/human_validation/cvat/cross_machine_human.py

Reconcile human CVAT annotations with machine-generated tracking data.
"""
from __future__ import annotations

import time
import pickle
import logging
import math
import os.path
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from imgstore.interface import VideoCapture
from idtrackerai_validator_server.backend import (
    load_idtrackerai_config,
    process_frame,
)
from flyhostel.data.human_validation.cvat.contour_utils import (
    get_contour_list_from_yolo_centroids,
    select_by_contour,
)
from flyhostel.data.human_validation.cvat.utils import annotate_crossings
from flyhostel.utils.utils import get_chunksize, get_experiment_identifier, get_dbfile

logger = logging.getLogger(__name__)

# OpenCV property; avoid the bare magic number cap.set(1, ...)
CAP_PROP_POS_FRAMES = 1
YOLO_CONTOUR_SIZE = 50
DEFAULT_FIRST_CHUNK = 50

# Annotation-text vocabulary used in the CVAT validation GUI
TAG_FRAGMENT_MUST_BREAK = "FMB"
TAG_COPY = "COPY"
TAG_SPATIAL_COPY = "SPATIAL-COPY"
TAG_CROSSING = "CROSSING"
TAG_DONE = "DONE"

# Set this from a config or env var instead of relying on a module-level global
DEBUG = False


# ---------------------------------------------------------------------------
# State carried through the per-frame loop
# ---------------------------------------------------------------------------

@dataclass
class CrossingState:
    """
    Accumulates rows and events produced while walking annotated frames.

    `roi0_rows` and `identity_rows` are lists of tuples that will be turned
    into DataFrames at the end of the loop. The other fields collect
    follow-up actions to apply once the per-frame walk is done.
    """
    roi0_rows: list = field(default_factory=list)
    identity_rows: list = field(default_factory=list)
    fragments_must_break: list = field(default_factory=list)
    annotations_to_copy: set = field(default_factory=set)
    annotations_to_spatial_copy: set = field(default_factory=set)
    crossings: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Annotation-tag handlers
# ---------------------------------------------------------------------------

def _frame_range_for_block(annotation, tag):
    """
    Given a row tagged COPY or SPATIAL-COPY, compute the block frame range.

    Returns (frame_number0, block, block_size, frame_numbers).
    """
    fn0, block, block_size = annotation.loc[
        annotation["text"] == tag, ["frame_number0", "block", "block_size"]
    ].values.flatten()
    start = fn0 + block * block_size
    return fn0, block, block_size, range(start, start + block_size)


def process_text_annotations(rec, annotation, metadata, annot_idx, state: CrossingState):
    """Handle a single non-localized annotation (one whose local_identity is NaN)."""
    frame_number, in_frame_index, local_identity, fragment = metadata
    text = rec["text"]  # was: annotation["text"].iloc[annot_idx]

    if text == TAG_FRAGMENT_MUST_BREAK:
        state.fragments_must_break.append((frame_number, fragment))

    elif text == TAG_COPY:
        # _frame_range_for_block still needs the DataFrame because it does a
        # tag-based filter, not a row-positional lookup.
        _, _, _, frame_numbers = _frame_range_for_block(annotation, TAG_COPY)
        state.annotations_to_copy.update(frame_numbers)

    elif text == TAG_SPATIAL_COPY:
        fn0, _, block_size, frame_numbers = _frame_range_for_block(annotation, TAG_SPATIAL_COPY)
        ref_frame = fn0 + block_size // 2
        state.annotations_to_spatial_copy.update((fn, ref_frame) for fn in frame_numbers)

    elif text == TAG_CROSSING:
        state.crossings.append((frame_number, fragment))

    if text == TAG_DONE:
        return None, None

    roi0_row = (frame_number, in_frame_index, rec["x"], rec["y"], fragment)
    ident_row = (frame_number, in_frame_index, local_identity)
    return roi0_row, ident_row


def parse_overlapping_annotation(df, match_idx, used_indices, next_idx):
    """
    Look up the machine-assigned in_frame_index and fragment for a matched
    contour, falling back to `next_idx` if the slot is already taken.
    Appends to `used_indices` in place.
    """
    fragment = df["fragment"].iloc[match_idx]
    in_frame_index = df["in_frame_index"].iloc[match_idx]
    if in_frame_index in used_indices:
        in_frame_index = next_idx
    used_indices.append(in_frame_index)
    return in_frame_index, fragment


# ---------------------------------------------------------------------------
# COPY / SPATIAL-COPY: replicate annotations from a reference frame
# ---------------------------------------------------------------------------

_MARK_ANNOTATION_COLUMNS = [
    "idx", "frame_number", "x", "y", "local_identity", "contour_id", "text",
    "frame_number0", "block", "block_size", "panel", "task", "frame_idx_in_block",
]


def _build_copy_mark(frame_number, block_size):
    """
    Build the placeholder annotation row inserted after a COPY so that
    later iterations can still find data at this frame.
    """
    row = {col: None for col in _MARK_ANNOTATION_COLUMNS}
    row["frame_number"] = frame_number
    row["text"] = TAG_COPY
    row["block_size"] = block_size
    return pd.DataFrame([row])


def _copy_frame(df, ref_frame_number, target_frame_number):
    """Return rows from `ref_frame_number` rewritten to `target_frame_number`."""
    src = df.loc[df["frame_number"] == ref_frame_number].copy()
    src["frame_number"] = target_frame_number
    return src


def copy_annotations(annotations_df, identity_corrected, roi0_corrected,
                     frame_number, ref_frame_number, block_size):
    """
    Copy all corrected rows from `ref_frame_number` to `frame_number`,
    and insert a marker row in `annotations_df` so that copies-of-copies
    can find a reference in later iterations.
    """
    ref_exists = (annotations_df["frame_number"] == ref_frame_number).any()
    if not ref_exists:
        logger.warning("No annotations for frame %s", ref_frame_number)
        return annotations_df, identity_corrected, roi0_corrected

    identity_corrected = pd.concat(
        [identity_corrected, _copy_frame(identity_corrected, ref_frame_number, frame_number)],
        axis=0,
    ).reset_index(drop=True)

    roi0_corrected = pd.concat(
        [roi0_corrected, _copy_frame(roi0_corrected, ref_frame_number, frame_number)],
        axis=0,
    ).reset_index(drop=True)

    annotations_df = pd.concat(
        [annotations_df.reset_index(drop=True), _build_copy_mark(frame_number, block_size)],
        axis=0,
    )
    return annotations_df, identity_corrected, roi0_corrected


def _unique_block_size(annotations_df):
    sizes = annotations_df["block_size"].dropna().unique()
    assert len(sizes) == 1, f"Expected single block_size, got {sizes!r}"
    return sizes[0]


def _apply_copies(annotations_df, identity_corrected, roi0_corrected,
                  pairs: Iterable[tuple[int, int]]):
    """Shared engine for spatial_copy_annotations and copy_annotations_one_block_back."""
    block_size = _unique_block_size(annotations_df)
    for target_frame, ref_frame in pairs:
        annotations_df, identity_corrected, roi0_corrected = copy_annotations(
            annotations_df, identity_corrected, roi0_corrected,
            target_frame, ref_frame, block_size=block_size,
        )
    identity_corrected.sort_values(["frame_number", "local_identity"], inplace=True)
    roi0_corrected.sort_values("frame_number", inplace=True)
    return identity_corrected, roi0_corrected


def spatial_copy_annotations(annotations_df, identity_corrected, roi0_corrected,
                             annotations_to_copy):
    """Copy rows from a mid-block reference frame to every frame in the block."""
    return _apply_copies(annotations_df, identity_corrected, roi0_corrected,
                         annotations_to_copy)


def copy_annotations_one_block_back(annotations_df, identity_corrected, roi0_corrected,
                                    annotations_to_copy, n_steps=1):
    """
    Replicate the annotations from one block back into the present frame.

    A block is the number of frames packed into a single space-time image
    (9 by default). The COPY tag in the GUI lets a user annotate one block
    and have following blocks reuse those annotations. Use sparingly: with
    every block the positions drift, so accuracy degrades.
    """
    block_size = _unique_block_size(annotations_df)
    pairs = [(fn, fn - block_size * n_steps) for fn in annotations_to_copy]
    return _apply_copies(annotations_df, identity_corrected, roi0_corrected, pairs)


# ---------------------------------------------------------------------------
# Per-frame crossing loop
# ---------------------------------------------------------------------------

def _prepare_machine_data(roi_0_machine, identity_machine, annotations_df, chunksize):
    """Merge ROI and identity machine tables and restrict to annotated frames."""
    df = (
        roi_0_machine.drop("id", axis=1)
        .merge(identity_machine.drop("id", axis=1), on=["frame_number", "in_frame_index"])
    )
    df["chunk"] = df["frame_number"] // chunksize
    return df.loc[df["frame_number"].isin(annotations_df["frame_number"].unique())]


# Module-level sentinel: tracks the position of the *next* frame cap.read()
# will return. After cap.set(POS, N), reading yields frame N and advances
# the internal pointer to N+1. So if next_expected_frame == N, we can skip
# the seek and read directly.

def _read_frame_candidates(cap, frame_number, df, config, cap_state):
    """
    Produce candidate contours for a frame, seeking only when necessary.

    `cap_state` is a small dict carrying the next-frame the capture is
    positioned at, so consecutive frames can be read without re-seeking.
    Mutated in place.
    """
    if (df["modified"] == 1).any():
        # YOLO branch: no video read needed, position is unchanged.
        candidates = get_contour_list_from_yolo_centroids(
            df[["x", "y"]].values, size=YOLO_CONTOUR_SIZE,
        )
        return None, candidates

    # Only seek if we're not already where we want to be. A successful read
    # advances the pointer by one, so "already there" means
    # cap_state["next_frame"] == frame_number.
    if cap_state["next_frame"] != frame_number:
        cap.set(CAP_PROP_POS_FRAMES, frame_number)

    _, frame = cap.read()
    cap_state["next_frame"] = frame_number + 1
    frame = frame[:, :, 0]
    candidates = [np.asarray(c) for c in process_frame(frame, config)]
    return frame, candidates


def _resolve_index_and_fragment(df, human_contour, candidates, used_indices,
                                annot_idx, last_machine_id, frame_number, frame):
    """
    Match the human contour to a machine candidate. Returns
    (in_frame_index, fragment, n_matches). When no machine match is found,
    allocate a fresh in_frame_index past last_machine_id.
    """
    match_idx, n = select_by_contour(human_contour, candidates, debug=False, frame=frame)
    if match_idx is not None:
        in_frame_index, fragment = parse_overlapping_annotation(
            df, match_idx, used_indices, next_idx=annot_idx + last_machine_id,
        )
        return in_frame_index, fragment, n

    logger.debug("Could not select by contour in frame %s", frame_number)
    in_frame_index = annot_idx + last_machine_id
    while in_frame_index in used_indices:
        in_frame_index += 1
    used_indices.append(in_frame_index)
    return in_frame_index, None, n


def _log_winner_count(n, frame_number, local_identity):
    if n == 0:
        logger.debug("De novo annotation detected in frame %s with local identity %s",
                     frame_number, local_identity)
    elif n != 1:
        logger.debug("Multiple winners detected in frame %s with local identity %s",
                     frame_number, local_identity)


def _process_single_annotation(rec, annotation, annot_idx, df, frame, candidates,
                               used_indices, last_machine_id, frame_number,
                               annotated_contours, state):
    """Cross one human annotation row with the machine candidates in this frame."""
    # rec is a numpy record with the fields in _ANNOT_COLS; field access is
    # ~ns vs ~µs for pandas .iloc. We still need `annotation` for the
    # process_text_annotations DataFrame-level lookups.
    human_contour = annotated_contours[rec["idx"]]
    local_identity = rec["local_identity"]

    in_frame_index, fragment, n = _resolve_index_and_fragment(
        df, human_contour, candidates, used_indices,
        annot_idx, last_machine_id, frame_number, frame,
    )

    if fragment is None and DEBUG and not np.isnan(local_identity):
        logger.debug(
            "Fly blob added in frame %s with in_frame_index %s and local_identity %s",
            frame_number, in_frame_index, local_identity,
        )

    if np.isnan(local_identity):
        roi0_row, ident_row = process_text_annotations(
            rec, annotation,
            (frame_number, in_frame_index, local_identity, fragment),
            annot_idx,
            state,
        )
        if roi0_row is not None:
            state.roi0_rows.append(roi0_row)
            state.identity_rows.append(ident_row)
        return

    state.roi0_rows.append((
        frame_number, in_frame_index, rec["x"], rec["y"], fragment,
    ))
    state.identity_rows.append((frame_number, in_frame_index, local_identity))
    _log_winner_count(n, frame_number, local_identity)


def _walk_frames_pickable(basedir, machine_data, annotations_df, annotated_contours,
                        config, last_machine_id):
    """
    wrapper around _walk_frames that creates its own VideoCapture object
    """
    # Open the cap here (not in the caller) so the cached path doesn't pay
    # for VideoCapture init either — which the profile showed was ~1s by itself.
    cap = VideoCapture(os.path.join(basedir, "metadata.yaml"), DEFAULT_FIRST_CHUNK)
    try:
        state = _walk_frames(
            cap, machine_data, annotations_df, annotated_contours,
            config, last_machine_id,
        )
    finally:
        cap.release()

    return state



# Columns accessed per annotation by _process_single_annotation and
# process_text_annotations. Pulling them as a numpy record array once per
# frame avoids ~5 pandas .iloc calls per annotation row.
_ANNOT_COLS = ["idx", "local_identity", "x", "y", "text",
               "frame_number0", "block", "block_size"]


def _walk_frames(cap, machine_data, annotations_df, annotated_contours,
                 config, last_machine_id):
    """Run the per-frame crossing loop and return the accumulated state.

    Two optimizations beyond the original:
      1. annotations_df is pre-grouped by frame_number once, so per-frame
         lookup is O(1) instead of an O(N) scan.
      2. The video capture is only seeked when the desired frame isn't the
         one cap.read() would return next. Consecutive-frame reads skip
         the expensive cv2.VideoCapture.set call entirely.
    """
    state = CrossingState()

    # #1: O(1) per-frame annotation lookup
    annotations_by_frame = dict(tuple(annotations_df.groupby("frame_number")))

    # groupby iterates in ascending key order by default; that ordering is
    # what makes the seek-skipping below worthwhile.
    groups = list(machine_data.groupby("frame_number", sort=True))
    pb = tqdm(total=len(groups), desc="Crossing human annotations and machine data")

    # #2: cap_state["next_frame"] = the frame index cap.read() would return
    # next. Sentinel -1 forces a seek on the first real read.
    cap_state = {"next_frame": -1}

    for frame_number, df in groups:
        annotation = annotations_by_frame.get(frame_number)
        if annotation is None or annotation.shape[0] == 0:
            pb.update(1)
            continue

        frame, candidates = _read_frame_candidates(
            cap, frame_number, df, config, cap_state,
        )

        # Extract the per-annotation fields as a numpy record array once;
        # field access on `rec` inside the loop is ~ns vs ~µs for .iloc.
        annot_records = annotation[_ANNOT_COLS].to_records(index=False)

        used_indices = []
        for annot_idx, rec in enumerate(annot_records):
            _process_single_annotation(
                rec, annotation, annot_idx, df, frame, candidates,
                used_indices, last_machine_id, frame_number,
                annotated_contours, state,
            )
        pb.update(1)

    return state
    

# ---------------------------------------------------------------------------
# Post-loop processing: build DataFrames, merge with originals, clean up
# ---------------------------------------------------------------------------

def _build_corrected_frames(state: CrossingState):
    identity = pd.DataFrame.from_records(
        state.identity_rows,
        columns=["frame_number", "in_frame_index", "local_identity"],
    )
    roi0 = pd.DataFrame.from_records(
        state.roi0_rows,
        columns=["frame_number", "in_frame_index", "x", "y", "fragment"],
    )
    return identity, roi0


def _merge_with_originals(roi0_corrected, identity_corrected, annotations_df):
    """
    Stamp validated/non-modified flags, concatenate corrected rows with the
    originals, drop duplicates on (frame_number, in_frame_index), and sort.
    """
    roi0_corrected = roi0_corrected.copy()
    roi0_corrected["modified"] = False
    roi0_corrected["validated"] = True
    identity_corrected = identity_corrected.copy()
    identity_corrected["validated"] = True

    annotations_df = annotations_df.copy()
    annotations_df["modified"] = False
    annotations_df["validated"] = True

    roi0_cols = ["frame_number", "in_frame_index", "x", "y", "fragment",
                 "modified", "validated"]
    ident_cols = ["frame_number", "in_frame_index", "local_identity", "validated"]

    def merge(corrected, original, cols):
        return (
            pd.concat([corrected[cols], original[cols]], axis=0)
            .reset_index(drop=True)
            .drop_duplicates(["frame_number", "in_frame_index"])
            .sort_values(["frame_number", "in_frame_index"])
        )

    return merge(roi0_corrected, annotations_df, roi0_cols), \
           merge(identity_corrected, annotations_df, ident_cols)


def _apply_fragment_breaks(roi0_corrected, fragments_must_break, chunksize):
    """Re-number fragments after each requested FMB break point within its chunk."""
    if not fragments_must_break:
        return roi0_corrected

    max_frag = roi0_corrected.groupby("chunk").agg({"fragment": np.max}).reset_index()

    for frame_number, fragment in fragments_must_break:
        chunk = frame_number // chunksize
        if fragment is None:
            mask = (roi0_corrected["frame_number"] == frame_number) & \
                   (roi0_corrected["chunk"] == chunk)
            roi0_corrected.loc[mask, "fragment"] = np.nan
            continue

        new_id = max_frag.loc[max_frag["chunk"] == chunk, "fragment"].item() + 1
        logger.warning("Fragment %s after frame number %s becomes fragment %s",
                       fragment, frame_number, new_id)
        mask = (
            (roi0_corrected["frame_number"] > frame_number)
            & (roi0_corrected["chunk"] == chunk)
            & (roi0_corrected["fragment"] == fragment)
        )
        roi0_corrected.loc[mask, "fragment"] = new_id
        max_frag.loc[max_frag["chunk"] == chunk, "fragment"] += 1

    return roi0_corrected


def _drop_duplicate_fragment_assignments(roi0_corrected):
    """A fragment id must be unique per frame; nullify any duplicates."""
    counts = roi0_corrected.groupby(["frame_number", "fragment"]).size().reset_index()
    duplicates = counts.loc[counts[0] > 1]
    for _, row in duplicates.iterrows():
        mask = (
            (roi0_corrected["fragment"] == row["fragment"])
            & (roi0_corrected["frame_number"] == row["frame_number"])
        )
        roi0_corrected.loc[mask, "fragment"] = np.nan
    return roi0_corrected


def _restrict_to_range(df, first_frame, last_frame, chunksize):
    df = df.loc[(df["frame_number"] >= first_frame) & (df["frame_number"] < last_frame)].copy()
    df["chunk"] = df["frame_number"] // chunksize
    return df


def _add_annotation_ids(df):
    df = df.copy()
    df["annotation_id"] = (
        df["frame_number"].astype(str) + "_" + df["in_frame_index"].astype(str)
    )
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def cross_machine_human(basedir, identity_machine, roi_0_machine, annotations_df,
                        annotated_contours, last_machine_id,
                        first_frame_number=0, last_frame_number=math.inf):
    """
    Reconcile human CVAT annotations with machine-generated tracking data.

    `annotations_df` contains the fields exported in `load_task_annotations`.
    Returns (identity_corrected, roi0_corrected, score_dist).
    """
    # Load the idtrackerai preprocessing config (thresholds, ROI, etc.) for this experiment.
    config = load_idtrackerai_config(basedir)
    # Resolve the experiment's SQLite database file and read its chunk size
    # (number of frames per recording chunk; used to scope fragment ids to a chunk).
    chunksize = get_chunksize(dbfile=get_dbfile(basedir))
    score_dist: list = []  # kept for API compatibility; currently unused

    # Join the two machine-side tables (ROI positions + identity assignments) on
    # (frame_number, in_frame_index), tag each row with its chunk, and keep only
    # rows for frames that a human has actually annotated.
    machine_data = _prepare_machine_data(
        roi_0_machine, identity_machine, annotations_df, chunksize,
    )

    # Main loop: for every annotated frame, read the image, get candidate contours
    # (idtrackerai preprocessing or YOLO centroids if the frame was modified), and
    # match each human annotation to one of them. Accumulates corrected rows plus
    # follow-up actions (FMB / COPY / SPATIAL-COPY / CROSSING) into `state`.

    state = _walk_frames_pickable(
        basedir, machine_data, annotations_df, annotated_contours,
        config, last_machine_id,
    )

    # Convert the list-of-tuples accumulators in `state` into two DataFrames:
    # identity_corrected (frame, in_frame_index, local_identity) and
    # roi0_corrected (frame, in_frame_index, x, y, fragment).
    identity_corrected, roi0_corrected = _build_corrected_frames(state)

    # This is deprecated
    # # SPATIAL-COPY: for every (target_frame, ref_frame) pair the user tagged,
    # # copy the corrected rows from ref_frame (the mid-block reference) into
    # # target_frame. Also inserts a placeholder row into annotations_df so that
    # # copies-of-copies can still find a reference on later passes.
    # identity_corrected, roi0_corrected = spatial_copy_annotations(
    #     annotations_df, identity_corrected, roi0_corrected,
    #     sorted(state.annotations_to_spatial_copy),
    # )

    # # COPY: for every frame the user tagged, replicate the annotations from
    # # one block back (frame - block_size) into the present frame. Same machinery
    # # as spatial_copy_annotations, just with a different rule for picking the
    # # reference frame.
    # identity_corrected, roi0_corrected = copy_annotations_one_block_back(
    #     annotations_df, identity_corrected, roi0_corrected,
    #     sorted(state.annotations_to_copy),
    # )


    # Stamp validated/non-modified flags on both the corrected rows and the original
    # annotations, then concatenate them and drop duplicates on (frame, in_frame_index)
    # so the originals act as a fallback wherever the corrected pass produced nothing.
    roi0_corrected, identity_corrected = _merge_with_originals(
        roi0_corrected, identity_corrected, annotations_df,
    )

    # Re-derive the chunk column on the merged table (originals brought in rows without it).
    roi0_corrected["chunk"] = roi0_corrected["frame_number"] // chunksize

    # FMB ("fragment must break"): at each requested break point, split the fragment id
    # so frames after the break get a fresh fragment number within that chunk. If the
    # fragment is None, just nullify the fragment on that single frame.
    roi0_corrected = _apply_fragment_breaks(
        roi0_corrected, state.fragments_must_break, chunksize,
    )

    # CROSSING: mark frames flagged as crossings in identity_corrected (using
    # roi0_corrected to look up which blobs participate). Defined in cvat.utils.
    identity_corrected = annotate_crossings(
        identity_corrected, roi0_corrected, state.crossings,
    )

    # Sanity pass: a fragment id should appear at most once per frame. Where it
    # appears twice (e.g. a copy overlapped an existing assignment), null the
    # fragment so it doesn't confuse downstream tracking.
    roi0_corrected = _drop_duplicate_fragment_assignments(roi0_corrected)

    # Clip both tables to the requested [first_frame_number, last_frame_number)
    # window and recompute the chunk column on the survivors.
    roi0_corrected = _restrict_to_range(roi0_corrected, first_frame_number,
                                        last_frame_number, chunksize)
    identity_corrected = _restrict_to_range(identity_corrected, first_frame_number,
                                            last_frame_number, chunksize)

    # Add a stable "{frame_number}_{in_frame_index}" id used by downstream joins,
    # then return both tables plus the (currently empty) score_dist.
    return _add_annotation_ids(identity_corrected), \
           _add_annotation_ids(roi0_corrected), \
           score_dist