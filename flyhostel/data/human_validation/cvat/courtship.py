import os.path
import math
import subprocess
import logging
import json
import shutil
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from flyhostel.utils import (
    get_basedir,
    get_number_of_animals,
    get_chunksize,
    get_framerate,
    get_resolution,
)


from flyhostel.utils.cvat import (
    get_tasks_for_project,
    get_project_id_from_name,
)

from flyhostel.data.human_validation.cvat.cvat_integration import (
    get_zipfile_for_task
)
INTERVAL_BETWEEN_CHECKPOINTS_IN_SECONDS = 1

logger=logging.getLogger(__name__)

_KEYFRAME_HELP = """
Terminology: a KEYFRAME is a frame where you explicitly drew, moved, or resized
a shape. In CVAT it appears as a solid marker on that object's timeline row.
Frames between two keyframes are interpolated by CVAT — they look annotated in
the UI but carry no data in the export, so this parser cannot see them.

Every chunk a courtship bout spans needs its own keyframe, because engagement
markers are matched to COURTSHIP boxes frame-by-frame, and identities are
resolved per chunk.

To add one: navigate to a frame in the listed range, select the COURTSHIP
object, and either nudge its box or click the keyframe (star/diamond) toggle in
the object sidebar. Then draw the engagement markers on that same frame.
"""


def parse_frame_number(x):
    return int(x.split("_")[0])


def detect_continuous_bouts(annotations, label, isi, framerate, interpolate=True):
    """
    Group annotated events of a given category into temporally continuous bouts,
    one bout per CVAT track. Optionally interpolates bbox coordinates at every
    integer frame within each bout.

    Annotations are first split by `track_id` (the CVAT track functionality
    propagates the same rectangle across frames under one track_id), so two
    rectangles in the same frame belonging to different tracks always produce
    distinct bouts. Within a single track, two annotations belong to the same
    bout if their frame_number distance is at most `isi * framerate` frames;
    a longer gap starts a new bout.
    """
    cols = ["interval_id", "track_id", "frame_number",
            "x1", "y1", "x2", "y2", "is_annotated"]

    matching = [c for c in annotations["categories"] if c["name"] == label]
    if not matching:
        raise ValueError(f"Label {label!r} not found in annotations[categories].")
    label_id = matching[0]["id"]

    fn_by_image_id = {
        img["id"]: parse_frame_number(img["file_name"])
        for img in annotations["images"]
    }

    records = []
    for ann in annotations["annotations"]:
        if ann["category_id"] != label_id:
            continue
        if ann["image_id"] not in fn_by_image_id:
            continue
        x, y, w, h = ann["bbox"]
        records.append({
            "frame_number": fn_by_image_id[ann["image_id"]],
            "track_id": ann["attributes"].get("track_id", -1),
            "x1": x,
            "y1": y,
            "x2": x + w,
            "y2": y + h,
        })

    if not records:
        return pd.DataFrame(columns=cols)

    df = (
        pd.DataFrame(records)
        .sort_values(["track_id", "frame_number"], kind="mergesort")
        .reset_index(drop=True)
    )

    max_gap = isi * framerate
    gaps_within_track = df.groupby("track_id")["frame_number"].diff()
    track_changed = df["track_id"] != df["track_id"].shift()
    new_bout = track_changed | (gaps_within_track > max_gap)
    df["interval_id"] = new_bout.cumsum().astype(int)
    df["is_annotated"] = True

    if not interpolate:
        return df[cols]

    pieces = []
    for interval_id, bout in df.groupby("interval_id", sort=True):
        pieces.append(_interpolate_bout(bout, interval_id))
    out = pd.concat(pieces, ignore_index=True)

    return out[cols]


def _interpolate_bout(bout: pd.DataFrame, interval_id: int) -> pd.DataFrame:
    """Linearly interpolate bbox coords at every integer frame in [first, last]."""
    first = int(bout["frame_number"].iloc[0])
    last = int(bout["frame_number"].iloc[-1])
    track_id = bout["track_id"].iloc[0]

    if first == last:
        return bout.copy()

    annotated = bout.set_index("frame_number")[["x1", "y1", "x2", "y2"]]
    full_range = pd.RangeIndex(first, last + 1, name="frame_number")
    dense = annotated.reindex(full_range)
    dense[["x1", "y1", "x2", "y2"]] = dense[["x1", "y1", "x2", "y2"]].interpolate(
        method="linear", limit_direction="both"
    )

    out = dense.reset_index()
    out["interval_id"] = interval_id
    out["track_id"] = track_id
    out["is_annotated"] = out["frame_number"].isin(annotated.index)
    return out


def mark_courtship(new_data: pd.DataFrame,
                   intervals: pd.DataFrame,
                   x_col: str = "x",
                   y_col: str = "y",
                   frame_col: str = "frame_number") -> pd.DataFrame:
    """
    Add a boolean `courtship` column to `new_data`. A row is True iff there
    exists a bbox in `intervals` at the same frame_number such that (x, y)
    falls inside [x1, x2] x [y1, y2] (inclusive).
    """
    out = new_data.copy()
    out = out.reset_index(drop=False).rename(columns={"index": "_row_id"})

    boxes = intervals[[frame_col, "x1", "y1", "x2", "y2"]]
    candidates = out.merge(boxes, on=frame_col, how="inner")

    inside = (
        (candidates[x_col] >= candidates["x1"])
        & (candidates[x_col] <= candidates["x2"])
        & (candidates[y_col] >= candidates["y1"])
        & (candidates[y_col] <= candidates["y2"])
    )

    matched_ids = set(candidates.loc[inside, "_row_id"])
    out["courtship"] = out["_row_id"].isin(matched_ids)

    out = out.merge(
        intervals[["frame_number", "is_annotated", "interval_id"]],
        on="frame_number", how="left",
    )
    out.loc[out["is_annotated"].isna(), "is_annotated"] = True

    courtship_index = (
        out.groupby("frame_number")
        .agg({"courtship": np.any})
        .reset_index()
        .rename({"courtship": "has_courtship"}, axis=1)
    )
    out = out.merge(courtship_index, on="frame_number", how="outer")
    return out.drop(columns="_row_id")


def mark_ok_labels(annotations, intervals, chunksize):
    """
    For every courtship bout and chunk, store the set of local identities
    that are NOT involved in the courtship (i.e. flies that should remain
    distinguishable in that chunk).

    Returns: {interval_id: {"interval": (start, end),
                            "labels_per_chunk": {chunk: {ok_label, ...}}}}

    The dict is keyed by the actual interval_id value from `intervals`, not
    by row position — so callers should look up by interval_id, never by
    position-in-iteration.
    """
    intervals_index = (
        intervals.groupby(["interval_id", "track_id"])
        .agg({"frame_number": [np.min, np.max]})
        .reset_index()
    )
    intervals_index.columns = ["interval_id", "track_id", "min", "max"]
    intervals_index["chunk"] = intervals_index["min"] // chunksize

    categories_index = {x["id"]: x["name"] for x in annotations["categories"]}
    id_fn_index = {
        x["id"]: int(x["file_name"].split("_")[0])
        for x in annotations["images"]
    }

    all_intervals_ok_labels = {}
    # Iterate as rows, using the actual interval_id value — not the row index.
    for _, row in intervals_index.iterrows():
        interval_id = int(row["interval_id"])
        interval_start = int(row["min"])
        interval_end = int(row["max"])

        labels = {}
        for ann in annotations["annotations"]:
            fn = id_fn_index[ann["image_id"]]
            if not (interval_start <= fn <= interval_end):
                continue

            cat = categories_index[ann["category_id"]]
            try:
                cat = int(cat)
            except ValueError:
                continue

            chunk = fn // chunksize
            labels.setdefault(chunk, set()).add(cat)

        all_intervals_ok_labels[interval_id] = {
            "labels_per_chunk": labels,
            "interval": (interval_start, interval_end),
        }
    return all_intervals_ok_labels

def get_annotations(tasks=None, first_frame_number=None, last_frame_number=None):
    """
    Download and parse COCO annotations for the given CVAT tasks, optionally
    restricting to a frame-number window inferred from image file names.

    Image file names follow `FRAME_NUMBER_*.png`, so the frame number is the
    integer prefix of the file name. When `first_frame_number` and/or
    `last_frame_number` is provided, only images whose frame number falls in
    [first_frame_number, last_frame_number) survive; annotations referencing
    other images are dropped.

    Bounds use [start, end) — `last_frame_number` is exclusive — to match the
    convention used elsewhere in this module (e.g. cross_machine_human).
    Pass `last_frame_number=None` (or `math.inf`) for an open upper bound.
    """
    if os.path.exists("annotations"):
        shutil.rmtree("annotations")

    zip_files = [get_zipfile_for_task(".", task) for task in tasks]
    for zip_file in zip_files:
        process = subprocess.Popen(["unzip", zip_file])
        process.communicate()

    with open("annotations/instances_default.json", "r") as handle:
        annotations = json.load(handle)

    if first_frame_number is None and last_frame_number is None:
        return annotations

    lo = -math.inf if first_frame_number is None else first_frame_number
    hi = math.inf if last_frame_number is None else last_frame_number

    # Keep only images whose frame number is in [lo, hi), then drop
    # annotations that referenced dropped images.
    kept_images = [
        img for img in annotations["images"]
        if lo <= parse_frame_number(img["file_name"]) < hi
    ]
    kept_image_ids = {img["id"] for img in kept_images}

    annotations["images"] = kept_images
    annotations["annotations"] = [
        ann for ann in annotations["annotations"]
        if ann["image_id"] in kept_image_ids
    ]
    return annotations


def load_intervals(experiment, annotations=None, tasks=None):
    assert annotations is not None or tasks is not None
    if annotations is None:
        annotations = get_annotations(tasks=tasks)

    original_width, original_height = get_resolution(experiment)
    framerate = get_framerate(experiment)

    intervals = detect_continuous_bouts(annotations, "COURTSHIP", 1, framerate)
    mult_x = original_width / annotations["images"][0]["width"]
    mult_y = original_height / annotations["images"][0]["height"]
    intervals["x1"] *= mult_x
    intervals["y1"] *= mult_y
    intervals["x2"] *= mult_x
    intervals["y2"] *= mult_y
    return intervals


def replace_courtship_identities(data, all_intervals_ok_labels,
                                 all_intervals_engaged_labels,
                                 intervals, chunksize):
    """
    For each courtship bout, ensure that every engaged local_identity has a
    row at every frame of the bout, with (x, y) set to the bbox centroid.

    Engaged local_identity values come from `all_intervals_engaged_labels`,
    which is populated by mark_engaged_labels + _bin_engaged_to_chunks from
    explicit per-chunk engagement markers in the annotation. These local_ids
    live in the same namespace as data["local_identity"] (per-chunk machine
    ids), so the synthetic rows can claim them legitimately.

    `all_intervals_ok_labels` is unused now (kept for signature compatibility
    with downstream callers; remove once they're migrated).

    Rows touched are flagged synthetic_courtship=True so that
    remove_blobs_associated_to_courtship spares them.
    """
    centroids = (
        intervals.assign(
            cx=(intervals["x1"] + intervals["x2"]) / 2,
            cy=(intervals["y1"] + intervals["y2"]) / 2,
        )
        .set_index(["interval_id", "frame_number"])[["cx", "cy"]]
    )

    if "synthetic_courtship" not in data.columns:
        data = data.copy()
        data["synthetic_courtship"] = False

    new_rows = []

    for interval_id, info in all_intervals_engaged_labels.items():
        engaged_per_chunk = info["engaged_per_chunk"]
        # Pull interval frame range from centroids index for this interval_id.
        # (Cheaper than another lookup; the index is already built.)
        try:
            frames_in_interval = centroids.loc[interval_id].index
        except KeyError:
            continue
        interval_start = int(frames_in_interval.min())
        interval_end = int(frames_in_interval.max())

        for chunk, engaged_ids in engaged_per_chunk.items():
            # (1) Reassign coords on existing rows whose local_identity is
            # an engaged one in this chunk.
            mask = (
                (data["interval_id"] == interval_id)
                & (data["frame_number"] >= interval_start)
                & (data["frame_number"] <= interval_end)
                & ((data["frame_number"] // chunksize) == chunk)
                & (data["local_identity"].isin(engaged_ids))
            )
            if mask.any():
                idx = pd.MultiIndex.from_arrays([
                    data.loc[mask, "interval_id"],
                    data.loc[mask, "frame_number"],
                ])
                coords = centroids.reindex(idx)
                data.loc[mask, "x"] = coords["cx"].to_numpy()
                data.loc[mask, "y"] = coords["cy"].to_numpy()
                data.loc[mask, "synthetic_courtship"] = True

            # (2) Insert rows for engaged local_ids absent from this chunk.
            chunk_start = max(interval_start, chunk * chunksize)
            chunk_end = min(interval_end, (chunk + 1) * chunksize - 1)
            if chunk_start > chunk_end:
                continue

            present_in_chunk = (
                data.loc[
                    (data["interval_id"] == interval_id)
                    & (data["frame_number"].between(chunk_start, chunk_end)),
                    ["frame_number", "local_identity"],
                ]
                .groupby("frame_number")["local_identity"]
                .apply(set)
                .to_dict()
            )

            for frame_number in range(chunk_start, chunk_end + 1):
                key = (interval_id, frame_number)
                if key not in centroids.index:
                    continue
                cx = centroids.loc[key, "cx"]
                cy = centroids.loc[key, "cy"]
                already = present_in_chunk.get(frame_number, set())
                for li in engaged_ids:
                    if li in already:
                        continue
                    new_rows.append({
                        "interval_id": interval_id,
                        "frame_number": int(frame_number),
                        "chunk": int(chunk),
                        "local_identity": li,
                        "x": cx,
                        "y": cy,
                        "synthetic_courtship": True,
                        "courtship": True,
                        "has_courtship": True,
                        "is_a_crossing": True,
                        "class_name": "courtship",
                        "frame_validated": True,
                        "fragment": np.nan,
                        "is_annotated": False,
                    })

    if new_rows:
        data = pd.concat(
            [data, pd.DataFrame(new_rows)],
            axis=0, ignore_index=True,
        )

    return data.sort_values(["frame_number", "local_identity"]).reset_index(drop=True)


def annotate_validated_fragments(data):
    """
    Blobs in courtship frames that belong to non-courting flies are protected
    even if not manually annotated, as long as they belong to a fragment that
    is annotated in some other frame.
    """
    fragment_index = (
        data.loc[data["validated"] > 0]
        .groupby(["chunk", "fragment", "local_identity"])
        .size()
        .reset_index(name="count")
    )
    assert not fragment_index.duplicated(["chunk", "fragment", "local_identity"]).any()
    fragment_index["validated_fragment"] = True
    fragment_index.rename({"local_identity": "fragment_identity"}, axis=1, inplace=True)
    data = data.merge(fragment_index, on=["chunk", "fragment"], how="left")
    data.loc[data["validated_fragment"].isna(), "validated_fragment"] = False
    return data



def remove_blobs_associated_to_courtship(data):
    """
    Remove blobs that are either
      1) a crossing blob produced by a courtship event, or
      2) non-crossing blobs from flies in an ongoing courtship heavy-contact
         that happen to be briefly distinguishable.

    Synthetic-centroid rows produced by replace_courtship_identities are
    spared regardless: they represent the (assumed) positions of courting
    flies and are the whole reason we kept identity information through
    the bout.
    """
    fragment_ok = (
        (data["validated_fragment"] == True)
        | data["fragment"].isna()  # singleton, can't be placed in a fragment
    )
    # Treat missing synthetic_courtship as False so the column is optional.
    synthetic = data.get("synthetic_courtship", pd.Series(False, index=data.index))
    synthetic = synthetic.fillna(False).astype(bool)

    selector = synthetic | (~(data["courtship"]) & (fragment_ok | (~data["has_courtship"])))

    data.loc[~selector].to_csv("courtship_discarded.csv")
    return data.loc[selector]


def remove_courtship_identities_from_local_identity_table(lid_table, all_intervals_ok_labels, chunksize):
    """
    If a good blob has a local_identity associated with courtship in the same
    chunk, ignore it for identity propagation between chunks: the fly isn't
    properly segmented in at least one end of the chunk (possibly both, since
    a 5-minute chunk can be entirely covered by one bout).
    """
    for interval_id in all_intervals_ok_labels:
        interval_start, interval_end = all_intervals_ok_labels[interval_id]["interval"]
        if interval_start // chunksize == interval_end // chunksize:
            continue

        chunks_in_interval = list(all_intervals_ok_labels[interval_id]["labels_per_chunk"])
        for i, chunk in enumerate(chunks_in_interval):
            ok_labels = all_intervals_ok_labels[interval_id]["labels_per_chunk"][chunk]
            same_chunk = lid_table["chunk"] == chunk
            wrong_label = ~lid_table["local_identity"].isin(ok_labels)

            if i == 0:
                drop = same_chunk & (lid_table["position"] == "last") & wrong_label
            elif i == len(chunks_in_interval) - 1:
                drop = same_chunk & (lid_table["position"] == "first") & wrong_label
            else:
                drop = same_chunk & wrong_label
            lid_table = lid_table.loc[~drop]

    return lid_table


def mark_engaged_labels(annotations, intervals):
    """
    Parse engagement markers — small rectangles fully contained inside a
    COURTSHIP rectangle at the same frame — to determine which local_identity
    values in each chunk are engaged in each bout.

    Each marker is a regular integer-labeled rectangle ("1", "2", "3", ...)
    whose bbox is fully enclosed by a COURTSHIP rectangle's bbox at the same
    frame. Containment is checked per frame; markers that aren't fully
    inside any COURTSHIP rectangle are ignored (they're regular checkpoint
    annotations).

    Two markers per (interval_id, chunk) are expected (two engaged flies);
    if a different count appears, the function logs a warning but still
    returns what it found.

    Parameters
    ----------
    annotations : dict (COCO-style)
    intervals : DataFrame from detect_continuous_bouts (one row per integer
        frame in each bout, with x1, y1, x2, y2 of the COURTSHIP rectangle).

    Returns
    -------
    dict[interval_id -> dict["engaged_per_chunk" -> dict[chunk -> set[int]]]]
    """
    # Per-frame index of every COURTSHIP rectangle (in raw CVAT pixel space —
    # the same space the marker bboxes are in, *before* the resolution rescale
    # in load_intervals). We need raw-space bboxes for containment.
    # Easiest: re-derive from annotations rather than reusing rescaled intervals.

    categories_index = {c["id"]: c["name"] for c in annotations["categories"]}
    fn_by_image_id = {
        img["id"]: parse_frame_number(img["file_name"])
        for img in annotations["images"]
    }
    # Reverse lookup: track_id -> interval_id (for the parent COURTSHIP track).
    # We need this so that when a marker is contained in a COURTSHIP at some
    # frame, we know which bout interval_id to attribute it to.
    courtship_cat_id = next(
        c["id"] for c in annotations["categories"] if c["name"] == "COURTSHIP"
    )

    # Build a per-frame list of (track_id, x1, y1, x2, y2) for COURTSHIPs
    courtship_by_frame = {}  # frame_number -> list of (track_id, x1,y1,x2,y2)
    courtship_track_to_interval = {}  # track_id -> interval_id (resolved later)
    for ann in annotations["annotations"]:
        if ann["category_id"] != courtship_cat_id:
            continue
        if ann["image_id"] not in fn_by_image_id:
            continue
        fn = fn_by_image_id[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        track_id = ann["attributes"].get("track_id", -1)
        courtship_by_frame.setdefault(fn, []).append(
            (track_id, x, y, x + w, y + h)
        )

    # Map each COURTSHIP track_id to its interval_id by looking at intervals.
    # `intervals` has one row per (interval_id, track_id) — we built it that way.
    for _, row in intervals[["interval_id", "track_id"]].drop_duplicates().iterrows():
        courtship_track_to_interval[row["track_id"]] = int(row["interval_id"])

    # Walk every annotation. If it's an integer-labeled rectangle whose bbox is
    # fully contained in some COURTSHIP at the same frame, record it as an
    # engagement marker for that bout's chunk.
    engaged = {}  # interval_id -> {"engaged_per_chunk": {chunk: {local_id, ...}}}

    for ann in annotations["annotations"]:
        if ann["category_id"] == courtship_cat_id:
            continue
        if ann["image_id"] not in fn_by_image_id:
            continue
        cat = categories_index[ann["category_id"]]
        try:
            local_id = int(cat)
        except ValueError:
            continue

        fn = fn_by_image_id[ann["image_id"]]
        if fn not in courtship_by_frame:
            continue

        x, y, w, h = ann["bbox"]
        ax1, ay1, ax2, ay2 = x, y, x + w, y + h

        # Containment check: this marker fully inside any COURTSHIP at this frame?
        parent_interval_id = None
        for (track_id, cx1, cy1, cx2, cy2) in courtship_by_frame[fn]:
            if cx1 <= ax1 and cy1 <= ay1 and ax2 <= cx2 and ay2 <= cy2:
                parent_interval_id = courtship_track_to_interval.get(track_id)
                break  # first containing COURTSHIP wins; overlaps are rare
        if parent_interval_id is None:
            continue  # regular checkpoint annotation, not an engagement marker

        # Note: chunksize isn't available here — we'll attribute the marker to
        # its frame's chunk in the caller, who knows chunksize. For now, store
        # by frame_number; the caller bins to chunk.
        slot = engaged.setdefault(parent_interval_id, {})
        slot.setdefault(fn, set()).add(local_id)

    return engaged


def _bin_engaged_to_chunks(engaged_by_frame, chunksize):
    """
    Convert {interval_id: {frame_number: {local_id, ...}}} (output of
    mark_engaged_labels) into {interval_id: {"engaged_per_chunk": {chunk:
    {local_id, ...}}}}, unioning markers across frames within the same chunk.

    Also warns when a chunk has a marker count other than the expected 2
    engaged flies — too few suggests a missing marker, too many suggests a
    typo or a marker that should not have been attributed.
    """
    out = {}
    for interval_id, frame_to_ids in engaged_by_frame.items():
        per_chunk = {}
        for fn, ids in frame_to_ids.items():
            chunk = fn // chunksize
            per_chunk.setdefault(chunk, set()).update(ids)
        for chunk, ids in per_chunk.items():
            if len(ids) != 2:
                logger.warning(
                    "Interval %s, chunk %s: expected 2 engagement markers, got %d (%s)",
                    interval_id, chunk, len(ids), sorted(ids),
                )
        out[interval_id] = {"engaged_per_chunk": per_chunk}
    return out



class EngagementMarkerError(ValueError):
    """Raised when a courtship bout is missing required engagement markers."""


def parse_engagement_markers(annotations, intervals, chunksize,
                             expected_per_chunk=2, tolerance=0.0):
    """
    Parse engagement markers and validate that every (interval_id, chunk)
    in every bout has the expected number of markers.

    An engagement marker is an integer-labeled rectangle whose bbox is fully
    contained in a COURTSHIP rectangle at the same raw-annotated frame.
    Containment requires:
        courtship_x1 - tol <= marker_x1
        courtship_y1 - tol <= marker_y1
        marker_x2 <= courtship_x2 + tol
        marker_y2 <= courtship_y2 + tol
    Partial overlaps don't count.

    Parameters
    ----------
    annotations : dict (COCO-style)
    intervals : DataFrame from detect_continuous_bouts. Used only to map
        track_id -> interval_id.
    chunksize : int
    expected_per_chunk : int, default 2
        Number of engaged flies per bout per chunk. Fewer or more markers
        in any (interval_id, chunk) → raises EngagementMarkerError.
    tolerance : float, default 0.0
        Slack on the containment check, in CVAT-export pixels. Set to a
        small positive value (e.g. 0.5) if annotators draw markers flush
        with COURTSHIP edges and floating-point export causes false
        rejections.

    Returns
    -------
    dict[interval_id -> dict["engaged_per_chunk" -> dict[chunk -> set[int]]]]
        Same shape used by replace_courtship_identities.

    Raises
    ------
    EngagementMarkerError
        If any (interval_id, chunk) lacks the expected number of markers, or
        if a COURTSHIP track has no raw-annotated frame with markers at all.
        Error message lists every offending (interval_id, chunk) so the
        annotator can fix them in one pass.

        Also raises EngagementMarkerError if any chunk that a bout spans (per
            `intervals`) has no raw COURTSHIP keyframe at all. CVAT interpolation
            between keyframes doesn't count: the annotator must place at least one
            keyframe per (bout, chunk) so markers can attach to it.
    """
    
    categories_index = {c["id"]: c["name"] for c in annotations["categories"]}
    fn_by_image_id = {
        img["id"]: parse_frame_number(img["file_name"])
        for img in annotations["images"]
    }

    courtship_match = [c for c in annotations["categories"] if c["name"] == "COURTSHIP"]
    if not courtship_match:
        return {}  # no COURTSHIP category → no bouts → nothing to validate
    courtship_cat_id = courtship_match[0]["id"]

    # Build per-frame list of raw COURTSHIP bboxes with their track_id.
    courtship_by_frame = {}  # frame_number -> [(track_id, x1,y1,x2,y2), ...]
    for ann in annotations["annotations"]:
        if ann["category_id"] != courtship_cat_id:
            continue
        if ann["image_id"] not in fn_by_image_id:
            continue
        fn = fn_by_image_id[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        track_id = ann["attributes"].get("track_id", -1)
        courtship_by_frame.setdefault(fn, []).append(
            (track_id, x, y, x + w, y + h)
        )

    # track_id -> interval_id, for attributing markers to bouts.
    courtship_track_to_interval = {
        int(row["track_id"]): int(row["interval_id"])
        for _, row in intervals[["interval_id", "track_id"]].drop_duplicates().iterrows()
    }

    # For early-fail validation we also need: which (interval_id, chunk) pairs
    # are *expected* to have markers? Any chunk that contains at least one
    # raw-annotated COURTSHIP frame for the bout's track.
    keyframes_by_key = {}  # (interval_id, chunk) -> {frame_number, ...}
    for fn, courtships in courtship_by_frame.items():
        chunk = fn // chunksize
        for (track_id, *_) in courtships:
            iid = courtship_track_to_interval.get(int(track_id))
            if iid is not None:
                keyframes_by_key.setdefault((iid, chunk), set()).add(fn)
    expected_keys = set(keyframes_by_key)

    # Walk all annotations, identify markers by containment.
    # Per (interval_id, chunk), accumulate marker labels AND track the set
    # of raw-annotated frames at which markers were found (for the "real
    # frame, not interpolated" requirement).
    by_chunk = {}  # (interval_id, chunk) -> set of local_ids
    marker_frames = {}  # (interval_id, chunk) -> set of frame_numbers

    for ann in annotations["annotations"]:
        if ann["category_id"] == courtship_cat_id:
            continue
        if ann["image_id"] not in fn_by_image_id:
            continue
        cat = categories_index[ann["category_id"]]
        try:
            local_id = int(cat)
        except ValueError:
            continue

        fn = fn_by_image_id[ann["image_id"]]
        if fn not in courtship_by_frame:
            continue  # not at a raw COURTSHIP frame; cannot be a marker

        x, y, w, h = ann["bbox"]
        ax1, ay1, ax2, ay2 = x, y, x + w, y + h

        parent_iid = None
        for (track_id, cx1, cy1, cx2, cy2) in courtship_by_frame[fn]:
            if (cx1 - tolerance <= ax1
                    and cy1 - tolerance <= ay1
                    and ax2 <= cx2 + tolerance
                    and ay2 <= cy2 + tolerance):
                parent_iid = courtship_track_to_interval.get(int(track_id))
                break
        if parent_iid is None:
            continue  # checkpoint annotation outside any COURTSHIP

        chunk = fn // chunksize
        key = (parent_iid, chunk)
        by_chunk.setdefault(key, set()).add(local_id)
        marker_frames.setdefault(key, set()).add(fn)

    # --- Validate: which chunks does each bout actually span? --------------
    # `intervals` is dense (one row per integer frame), so its (interval_id,
    # frame_number // chunksize) pairs give the full spanned chunks per bout.
    spanned = (
        intervals
        .assign(chunk=intervals["frame_number"] // chunksize)
        .groupby("interval_id")["chunk"]
        .apply(lambda s: set(s.unique().tolist()))
        .to_dict()
    )  # {interval_id: {chunk, ...}}

    errors = []

    # Frame range of each (interval_id, chunk), so errors can name a concrete
    # window to annotate rather than just a chunk index.
    span_frames = (
        intervals
        .assign(chunk=intervals["frame_number"] // chunksize)
        .groupby(["interval_id", "chunk"])["frame_number"]
        .agg(["min", "max"])
        .to_dict("index")
    )  # {(interval_id, chunk): {"min": f0, "max": f1}}

    def _range_str(iid, chunk):
        rng = span_frames.get((iid, chunk))
        if rng is None:
            return f"chunk {chunk}"
        return f"frames {rng['min']}-{rng['max']} (chunk {chunk})"
    
    # Chunks the bout spans that have NO raw COURTSHIP keyframe at all.
    # These can't be in expected_keys (which is keyed off raw keyframes), so
    # they'd silently skip validation under the previous logic.
    chunks_seen_per_interval = {}  # {iid: {chunks with raw keyframes}}
    for (iid, chunk) in expected_keys:
        chunks_seen_per_interval.setdefault(iid, set()).add(chunk)

    for iid, spanned_chunks in spanned.items():
            seen = chunks_seen_per_interval.get(iid, set())
            missing_keyframes = spanned_chunks - seen
            for chunk in sorted(missing_keyframes):
                errors.append(
                    f"  bout {iid}: NO COURTSHIP KEYFRAME in {_range_str(iid, chunk)}.\n"
                    f"      The bout is visible there only because CVAT interpolates "
                    f"between keyframes in chunks {sorted(seen) or 'n/a'}.\n"
                    f"      Fix: open any frame in that range, make the COURTSHIP box "
                    f"a keyframe, then draw {expected_per_chunk} markers inside it."
                )
    # Per (interval_id, chunk) marker-count validation (unchanged).
    for key in sorted(expected_keys):
        iid, chunk = key
        markers = by_chunk.get(key, set())
        if not markers:
            errors.append(
                f"  bout {iid}, {_range_str(iid, chunk)}: COURTSHIP keyframe(s) "
                f"exist at frame(s) {sorted(keyframes_by_key[key])}, but NO "
                f"engagement markers were found inside the box there.\n"
                f"      Fix: on one of those exact frames, draw {expected_per_chunk} "
                f"integer-labelled rectangles fully inside the COURTSHIP box.\n"
                f"      (Markers drawn on a non-keyframe, or sticking out past the "
                f"COURTSHIP edge, are ignored.)"
            )
            continue
        if len(markers) != expected_per_chunk:
            errors.append(
                f"  bout {iid}, {_range_str(iid, chunk)}: found {len(markers)} "
                f"marker(s) {sorted(markers)} at frame(s) "
                f"{sorted(marker_frames[key])}, expected {expected_per_chunk}.\n"
                f"      Fix: add or delete markers on those frames so exactly "
                f"{expected_per_chunk} are inside the COURTSHIP box. If you drew "
                f"more, check whether one is flush with / outside the box edge "
                f"(tolerance={tolerance} px)."
            )

    # (Whole-track-missing check can now go away — the spanned-vs-seen
    # check above subsumes it, with a more useful error message.)
    if errors:
        raise EngagementMarkerError(
            f"Engagement marker validation failed ({len(errors)} problem(s)):\n"
            + "\n".join(errors)
            + "\n" + _KEYFRAME_HELP
        )


    # --- Pack output (unchanged) ------------------------------------------
    out = {}
    for (iid, chunk), labels in by_chunk.items():
        out.setdefault(iid, {"engaged_per_chunk": {}})["engaged_per_chunk"][chunk] = labels
    return out


def prepare_data_for_identity_annnotation_with_courtship(experiment, data):
    chunksize = get_chunksize(experiment)

    tasks = sorted(get_tasks_for_project(
        get_project_id_from_name(experiment, errors="raise")
    ))
    annotations = get_annotations(tasks=tasks)
    intervals = load_intervals(experiment, annotations=annotations)

    all_intervals_engaged_labels = parse_engagement_markers(
        annotations, intervals, chunksize,
    )

    data = mark_courtship(data, intervals)
    all_intervals_ok_labels = mark_ok_labels(annotations, intervals, chunksize)

    data = replace_courtship_identities(
        data,
        all_intervals_ok_labels=all_intervals_ok_labels,
        all_intervals_engaged_labels=all_intervals_engaged_labels,
        intervals=intervals,
        chunksize=chunksize,
    )

    data = annotate_validated_fragments(data)
    assert (
        data.loc[data["validated_fragment"] == True, "local_identity"]
        == data.loc[data["validated_fragment"] == True, "fragment_identity"]
    ).all()
    data = remove_blobs_associated_to_courtship(data)

    return data, all_intervals_ok_labels, all_intervals_engaged_labels