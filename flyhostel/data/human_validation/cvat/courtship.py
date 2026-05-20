import os.path
import subprocess
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

from flyhostel.data.human_validation.cvat.cvat_integration import (
    download_annotations_from_cvat
)
INTERVAL_BETWEEN_CHECKPOINTS_IN_SECONDS=1

def parse_frame_number(x):
    return int(x.split("_")[0])


# def detect_continuous_bouts(annotations, label, isi, framerate):
#     label_id=list(filter(lambda x: x["name"]==label, annotations["categories"]))[0]["id"]
#     label_events=list(filter(lambda x: x["category_id"]==label_id, annotations["annotations"]))
#     label_events_images=[x["image_id"] for x in label_events]
#     label_events_fn=[parse_frame_number(x["file_name"]) for x in annotations["images"] if x["id"] in label_events_images]
    
#     t_diff = np.diff(label_events_fn)
    
#     max_t_between=isi*framerate
    
#     new_bouts = t_diff > max_t_between
#     new_bouts = np.concatenate([[True], new_bouts])
#     new_bouts_idx = np.where(new_bouts)[0]
    
#     frame_number_start = np.array(label_events_fn)[new_bouts_idx]
    
#     end_bouts_idx=new_bouts_idx[1:]
#     end_bouts_idx-=1
#     end_bouts_idx=np.concatenate([end_bouts_idx, [len(new_bouts)-1]])
#     frame_number_end = np.array(label_events_fn)[end_bouts_idx]

#     assert len(frame_number_start) == len(frame_number_end)
#     intervals = list(zip(frame_number_start, frame_number_end))
#     return intervals


# def detect_continuous_bouts(annotations, label, isi, framerate, interpolate=True):
#     """
#     Group annotated events of a given category into temporally continuous bouts,
#     optionally interpolating bbox coordinates at every integer frame within each bout.

#     Two events belong to the same bout if their frame_number distance is at most
#     `isi * framerate` frames.

#     Parameters
#     ----------
#     annotations : dict
#         COCO-style annotations dict with `categories`, `annotations`, `images`.
#     label : str
#         Category name to filter on.
#     isi : float
#         Maximum inter-event interval in seconds.
#     framerate : float
#         Frames per second.
#     interpolate : bool
#         If True (default), fill in every integer frame between the first and
#         last annotated frame of each bout, with bbox coords linearly
#         interpolated. If False, return only the original annotated frames.

#     Returns
#     -------
#     pd.DataFrame
#         Columns: interval_id, frame_number, x1, y1, x2, y2, is_annotated.
#         `is_annotated` is True for rows from the original annotations,
#         False for interpolated rows. Sorted by (interval_id, frame_number).
#     """
#     cols = ["interval_id", "frame_number", "x1", "y1", "x2", "y2", "is_annotated"]

#     # Resolve label name -> category id
#     matching = [c for c in annotations["categories"] if c["name"] == label]
#     if not matching:
#         raise ValueError(f"Label {label!r} not found in annotations[categories].")
#     label_id = matching[0]["id"]

#     # image_id -> frame_number lookup
#     fn_by_image_id = {
#         img["id"]: parse_frame_number(img["file_name"])
#         for img in annotations["images"]
#     }

#     # Collect (frame_number, x1, y1, x2, y2) per matching annotation
#     records = []
#     for ann in annotations["annotations"]:
#         if ann["category_id"] != label_id:
#             continue
#         if ann["image_id"] not in fn_by_image_id:
#             continue
#         x, y, w, h = ann["bbox"]
#         records.append({
#             "frame_number": fn_by_image_id[ann["image_id"]],
#             "x1": x,
#             "y1": y,
#             "x2": x + w,
#             "y2": y + h,
#         })

#     if not records:
#         return pd.DataFrame(columns=cols)

#     df = pd.DataFrame(records).sort_values("frame_number").reset_index(drop=True)

#     # Bout segmentation
#     max_gap = isi * framerate
#     gaps = df["frame_number"].diff()
#     new_bout = (gaps > max_gap) | gaps.isna()
#     df["interval_id"] = new_bout.cumsum().astype(int)
#     df["is_annotated"] = True

#     if not interpolate:
#         return df[cols]

#     # Per-bout interpolation: for each bout, build a dense frame_number range
#     # spanning its first to last annotated frame, then linearly interpolate.
#     pieces = []
#     for interval_id, bout in df.groupby("interval_id", sort=True):
#         pieces.append(_interpolate_bout(bout, interval_id))
#     out = pd.concat(pieces, ignore_index=True)
#     return out[cols]

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

    Parameters
    ----------
    annotations : dict
        COCO-style annotations dict with `categories`, `annotations`, `images`.
        Each annotation must carry a top-level `track_id`.
    label : str
        Category name to filter on.
    isi : float
        Maximum inter-event interval in seconds.
    framerate : float
        Frames per second.
    interpolate : bool
        If True (default), fill in every integer frame between the first and
        last annotated frame of each bout, with bbox coords linearly
        interpolated. If False, return only the original annotated frames.

    Returns
    -------
    pd.DataFrame
        Columns: interval_id, track_id, frame_number, x1, y1, x2, y2, is_annotated.
        `is_annotated` is True for rows from the original annotations,
        False for interpolated rows. Sorted by (interval_id, frame_number).
    """
    cols = ["interval_id", "track_id", "frame_number",
            "x1", "y1", "x2", "y2", "is_annotated"]

    # Resolve label name -> category id
    matching = [c for c in annotations["categories"] if c["name"] == label]
    if not matching:
        raise ValueError(f"Label {label!r} not found in annotations[categories].")
    label_id = matching[0]["id"]

    # image_id -> frame_number lookup
    fn_by_image_id = {
        img["id"]: parse_frame_number(img["file_name"])
        for img in annotations["images"]
    }

    # Collect one record per matching annotation, including its track_id.
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

    # Sort by (track_id, frame_number) so the per-track gap diff below is
    # meaningful. We use a stable sort so equal frame_numbers within a track
    # keep their original order.
    df = (
        pd.DataFrame(records)
        .sort_values(["track_id", "frame_number"], kind="mergesort")
        .reset_index(drop=True)
    )

    # Bout segmentation, applied within each track:
    #   - a gap > isi*framerate frames from the previous annotation in the
    #     same track starts a new bout
    #   - the first annotation in each track also starts a new bout (its gap
    #     is computed against the previous track and so is meaningless; we
    #     mark it explicitly)
    max_gap = isi * framerate
    gaps_within_track = df.groupby("track_id")["frame_number"].diff()
    track_changed = df["track_id"] != df["track_id"].shift()
    new_bout = track_changed | (gaps_within_track > max_gap)
    df["interval_id"] = new_bout.cumsum().astype(int)
    df["is_annotated"] = True

    if not interpolate:
        return df[cols]

    # Per-bout interpolation: dense frame_number range from first to last
    # annotated frame of the bout, with bbox coords linearly interpolated.
    # track_id is constant within a bout (by construction above), so
    # _interpolate_bout can carry it through from the input rows.
    pieces = []
    for interval_id, bout in df.groupby("interval_id", sort=True):
        pieces.append(_interpolate_bout(bout, interval_id))
    out = pd.concat(pieces, ignore_index=True)

    return out[cols]

def _interpolate_bout(bout: pd.DataFrame, interval_id: int) -> pd.DataFrame:
    """Linearly interpolate bbox coords at every integer frame in [first, last].

    track_id is constant within a bout (bouts are segmented per-track upstream),
    so we read it once from the input and stamp it onto every output row,
    including the interpolated ones.
    """
    first = int(bout["frame_number"].iloc[0])
    last = int(bout["frame_number"].iloc[-1])
    track_id = bout["track_id"].iloc[0]

    if first == last:
        # Single-frame bout: nothing to interpolate
        return bout.copy()

    # Index the annotated rows by frame_number
    annotated = bout.set_index("frame_number")[["x1", "y1", "x2", "y2"]]

    # Reindex to every integer frame in the span, introducing NaNs for new rows
    full_range = pd.RangeIndex(first, last + 1, name="frame_number")
    dense = annotated.reindex(full_range)

    # Linear interpolation: integer frame_number index is evenly spaced
    # (one unit per frame), which is exactly what we want.
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
    Add a boolean `courtship` column to `new_data`.

    A row is marked True iff there exists a bbox in `intervals` at the same
    frame_number such that (x, y) falls inside [x1, x2] x [y1, y2] (inclusive).

    Parameters
    ----------
    new_data : DataFrame with at least `frame_number`, x_col, y_col columns.
    intervals : DataFrame with `frame_number`, x1, y1, x2, y2 columns
        (the per-frame output of detect_continuous_bouts with interpolate=True).
    x_col, y_col, frame_col : optional column-name overrides.

    Returns
    -------
    pd.DataFrame
        A copy of new_data with one new boolean column `courtship`.
    """
    out = new_data.copy()

    # Stable row id so we can collapse multiple bbox matches back to per-row.
    out = out.reset_index(drop=False).rename(columns={"index": "_row_id"})

    boxes = intervals[[frame_col, "x1", "y1", "x2", "y2"]]

    # Inner-merge on frame_number: only frames that exist in `intervals`
    # produce candidate rows. Rows in new_data with no matching frame
    # naturally drop out and will be marked False.
    candidates = out.merge(boxes, on=frame_col, how="inner")

    inside = (
        (candidates[x_col] >= candidates["x1"])
        & (candidates[x_col] <= candidates["x2"])
        & (candidates[y_col] >= candidates["y1"])
        & (candidates[y_col] <= candidates["y2"])
    )

    # Row ids that had at least one bbox containing the point
    matched_ids = set(candidates.loc[inside, "_row_id"])

    out["courtship"] = out["_row_id"].isin(matched_ids)

    out=out.merge(intervals[["frame_number", "is_annotated", "interval_id"]], on="frame_number", how="left")
    out.loc[out["is_annotated"].isna(), "is_annotated"]=True

    courtship_index=out.groupby("frame_number").agg({"courtship": np.any}).reset_index().rename({"courtship": "has_courtship"}, axis=1)
    out=out.merge(courtship_index, on="frame_number", how="outer")

    return out.drop(columns="_row_id")

def mark_ok_labels(annotations, intervals, chunksize):
    """
    For every courtship bout (frames where the male is mounting the female)
    and chunk, stores a set of the local identities that are not involved in the courtship

    Returns x (dict): x[interval_id][chunk]
    """
    intervals_index=intervals.groupby(["interval_id", "track_id"]).agg({"frame_number": [np.min, np.max]}).reset_index()
    intervals_index.columns=["interval_id", "track_id", "min", "max"]   
    intervals_index["chunk"]=intervals_index["min"]//chunksize
    categories_index={x["id"]: x["name"] for x in annotations["categories"]}
    id_fn_index={x["id"]: int(x["file_name"].split("_")[0]) for x in annotations["images"]}

    all_intervals_ok_labels={}
    for interval_id in range(intervals_index.shape[0]):
        labels={}
        interval_start=intervals_index.iloc[interval_id]["min"]
        interval_end=intervals_index.iloc[interval_id]["max"]
        

        for ann in annotations["annotations"]:
            fn=id_fn_index[ann["image_id"]]

            if fn >= interval_start and fn <= interval_end:
                cat_id=ann["category_id"]
                cat=categories_index[cat_id]
                try:
                    cat=int(cat)
                except ValueError:
                    continue
                
                # if cat_id == 3: print(fn, fn//chunksize, fn%chunksize)
                chunk=fn//chunksize
                if chunk not in labels:
                    labels[chunk]=set([cat])
                else:
                    labels[chunk].add(cat)
        
        all_intervals_ok_labels[interval_id]={
            "labels_per_chunk": labels,
            "interval": (interval_start, interval_end),
        }
    return all_intervals_ok_labels


def get_annotations(experiment, tasks=None, download=True):

    if download:
        if os.path.exists("annotations"):
            shutil.rmtree("annotations")
    
        zip_files=download_annotations_from_cvat(experiment, ".", tasks=tasks)
        for zip_file in zip_files:
            process=subprocess.Popen(["unzip",  zip_file])
            process.communicate()

    with open("annotations/instances_default.json", "r") as handle:
        annotations=json.load(handle)

    return annotations
        
def load_intervals(experiment, annotations=None, tasks=None):
    assert annotations is not None or tasks is not None

    if annotations is None:
        annotations=get_annotations(experiment, tasks=tasks)


    original_width, original_height=get_resolution(experiment)
    framerate=get_framerate(experiment)

    intervals = detect_continuous_bouts(annotations, "COURTSHIP", 1, framerate)
    mult_x, mult_y = original_width / annotations["images"][0]["width"], original_height / annotations["images"][0]["height"]
    intervals["x1"]*=mult_x
    intervals["y1"]*=mult_y
    intervals["x2"]*=mult_x
    intervals["y2"]*=mult_y
    return intervals


# Discard integer identities that spuriously appear during the courtship bout
# I dont want to keep brief restorations of identities that belong to flies engaged in courtship mounting
# until the flies finally separate

def discard_courtship_identities(data, all_intervals_ok_labels, intervals, chunksize, local_identities=None):
    
    rows=[]

    tracks_index=intervals[["interval_id", "track_id"]].drop_duplicates()


    for interval_id in all_intervals_ok_labels:
        interval_start, interval_end = all_intervals_ok_labels[interval_id]["interval"]
        track_id = tracks_index.query("interval_id == @interval_id")["track_id"].item()



        for chunk in all_intervals_ok_labels[interval_id]["labels_per_chunk"]:
            ok_labels =all_intervals_ok_labels[interval_id]["labels_per_chunk"][chunk]

            data=data.loc[
                ~(
                    (data["interval_id"]==interval_id) & \
                    (data["frame_number"]>=interval_start) & \
                    (data["frame_number"]<=interval_end) & \
                    (~data["local_identity"].isin(ok_labels))
                )
            ]
            if local_identities is not None:
                courtship_identities = [identity for identity in local_identities if identity not in ok_labels]
                for frame_number in tqdm(range(interval_start, interval_end)):
                    if frame_number//chunksize == chunk:
                        for local_identity in courtship_identities:
                            row = pd.Series({
                                "interval_id": interval_id,
                                "frame_number": frame_number,
                                "local_identity": local_identity,
                            })
                            rows.append(row)
                    else:
                        continue

    # TODO Update this so rows contains the new data points
    # corresponding to flies engaged in courtship at the corresponding frames
    # You may want to just have interval_id, frame_number, local_identity, x and y
    # and have the rest of columns be added later outside of discard_courtship_identities,
    # in the call to prepare_data_for_identity_annnotation_with_courtship

    if rows:
        data=pd.concat([
            data,
            rows
        ], axis=0).sort_values("frame_number")

    return data



def annotate_validated_fragments(data):
    """
    Blobs occuring in frames with courtship that belong to normal flies will be protected even if not manually annotated
    as long as they belong to a fragment which is annotated in some other frame
    """
    fragment_index=data.loc[data["validated"]>0].groupby(["chunk", "fragment", "local_identity"]).size().reset_index(name="count")
    assert not fragment_index.duplicated(["chunk", "fragment", "local_identity"]).any()
    fragment_index["validated_fragment"]=True
    fragment_index.rename({"local_identity": "fragment_identity"}, axis=1, inplace=True)
    data=data.merge(fragment_index, on=["chunk", "fragment"], how="left")
    data.loc[data["validated_fragment"].isna(), "validated_fragment"]=False
    return data



def remove_blobs_associated_to_courtship(data):
    """
    Removes blobs that are either
    1) a crossing blob produced by a courtship event
    2) not crossing blobs that belong to flies engaged in an ongoing courtship heavy contact
        which happen to be distinguishable (for a few frames at a time only) 
    """

    fragment_ok = (
        # if the fragment has at least 1 blob I have annotated
        (data["validated_fragment"]==True) |
        # if the blob has no fragment i.e. it is a singleton
        # = a blob that cannot be placed in a fragment and is a length 1 fragment
        data["fragment"].isna()
    )

    selector= ~(data["courtship"]) & (fragment_ok | (~data["has_courtship"]))
    
    data.loc[np.bitwise_not(selector)].to_csv("courtship_discarded.csv")

    data=data.loc[selector]

    return data


def remove_courtship_identities_from_local_identity_table(lid_table, all_intervals_ok_labels, chunksize):
    """
    If a good blob has a local_identity associated with courtship in the same chunk,
    ignore it regarding the propagation of identities between chunks, since by definition,
    the fly is not properly segmented in at least the beginning or the end of the chunk
    (possibly both if the courtship takes the whole chunk, which can happen because 1 chunk = 5 minutes)
    """

    for interval_id in all_intervals_ok_labels:

        interval_start, interval_end=all_intervals_ok_labels[interval_id]["interval"]


        if interval_start//chunksize==interval_end//chunksize:
            continue

        for i, chunk in enumerate(all_intervals_ok_labels[interval_id]["labels_per_chunk"]):
            ok_labels=all_intervals_ok_labels[interval_id]["labels_per_chunk"][chunk]
    
            if i==0:    
                lid_table=lid_table.loc[
                    ~((lid_table["chunk"]==chunk) & (lid_table["position"]=="last") & (~lid_table["local_identity"].isin(ok_labels)))
                ]
            elif i==len(all_intervals_ok_labels[interval_id]["labels_per_chunk"])-1:
                lid_table=lid_table.loc[
                    ~((lid_table["chunk"]==chunk) & (lid_table["position"]=="first") & (~lid_table["local_identity"].isin(ok_labels)))
                ]
            else:
                lid_table=lid_table.loc[
                    ~((lid_table["chunk"]==chunk)  & (~lid_table["local_identity"].isin(ok_labels)))
                ]

    return lid_table

# Public

def prepare_data_for_identity_annnotation_with_courtship(experiment, data, download=True):
    chunksize=get_chunksize(experiment)
    number_of_animals=get_number_of_animals(experiment)
    local_identities=list(range(1, number_of_animals+1))

    annotations=get_annotations(experiment, tasks=None, download=download)
    
    intervals=load_intervals(experiment, annotations=annotations)

    data=mark_courtship(data, intervals)
    all_intervals_ok_labels=mark_ok_labels(annotations, intervals, chunksize)
    data=discard_courtship_identities(
        data, all_intervals_ok_labels, intervals=intervals,
        chunksize=chunksize,
        local_identities=local_identities
    )
    
    data=annotate_validated_fragments(data)
    assert (data.loc[data["validated_fragment"]==True, "local_identity"]==data.loc[data["validated_fragment"]==True, "fragment_identity"]).all()
       
    courtship_data=data.loc[(data["courtship"] & data["is_annotated"])]
    data=remove_blobs_associated_to_courtship(data)

    return data, all_intervals_ok_labels