import logging
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
logger=logging.getLogger(__name__)


"""Apply manual validation corrections from a CSV to a tracking dataset."""
import logging

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# Defaults applied to every synthesized row, keeping the per-branch code focused
# only on the fields that actually differ.
_DEFAULT_ROW_FIELDS = {
    "is_a_crossing": False,
    "validated": 1,
    "frame_validated": False,
    "class_name": "undefined",
}


def _load_validation_table(validation_csv, replace):
    """Read the validation CSV and optionally filter by the `replace` flag."""
    table = pd.read_csv(validation_csv, comment="#")
    if replace is not None:
        table = table.loc[table["replace"] == replace]
    return table


def _frame_range(row):
    """Return the inclusive frame range described by a validation row.

    Falls back to the single `frame_number` when no explicit range is given.
    """
    first = row.get("first_frame_number", np.nan)
    if pd.isna(first):
        return np.array([row["frame_number"]], dtype=np.int64)
    last = row["last_frame_number"]
    return np.arange(first, last + 1, dtype=np.int64)


def _drop_rows_by_index(new_data, to_drop):
    """Return `new_data` with rows whose index appears in `to_drop` removed."""
    if to_drop.empty:
        return new_data
    keep_mask = ~new_data.index.isin(to_drop.index)
    return new_data.loc[keep_mask]


def _synthesize_rows_for_range(frame_numbers, chunk, row, local_identity):
    """Build a fresh block of rows when no source rows are available to copy."""
    block = pd.DataFrame({"frame_number": frame_numbers})
    block["fragment"] = np.nan
    block["chunk"] = chunk
    block["x"] = row["x"]
    block["y"] = row["y"]
    block["local_identity"] = local_identity
    block["in_frame_index"] = np.nan
    block["modified"] = 1
    for key, value in _DEFAULT_ROW_FIELDS.items():
        block[key] = value
    return block


def _apply_replace(new_data, row, chunksize, extra_rows):
    """Handle a validation row with replace == True.

    Returns the (possibly modified) `new_data` frame; appends synthesized rows
    to `extra_rows` in place.
    """
    frame_number = row["frame_number"]
    chunk = frame_number // chunksize
    fragment = row["fragment"]
    local_identity = row["local_identity"]

    if row.get("by_identity", True):
        mask = (
            (new_data["frame_number"] == frame_number)
            & (new_data["local_identity"] == local_identity)
        )
        extra_data = new_data.loc[mask].copy()
        extra_data["fragment"] = np.nan

    elif pd.isna(row.get("first_frame_number", np.nan)):
        mask = (
            (new_data["frame_number"] == frame_number)
            & (new_data["fragment"] == fragment)
        )
        extra_data = new_data.loc[mask].copy()
        if extra_data.empty:
            logger.warning("Ignoring line in validation.csv:\n%s", row)
            return new_data

    else:
        frame_numbers = _frame_range(row)

        if pd.isna(fragment):
            # No source rows to copy from -> synthesize a fresh block and
            # remove any pre-existing rows for this identity over the range.
            synthesized = _synthesize_rows_for_range(
                frame_numbers, chunk, row, local_identity
            )
            stale_mask = (
                new_data["frame_number"].isin(frame_numbers)
                & (new_data["local_identity"] == local_identity)
            )
            n_removed = stale_mask.sum()
            if n_removed:
                logger.info("Removing %d rows", n_removed)
            new_data = new_data.loc[~stale_mask]
            extra_rows.append(synthesized)
            return new_data

        mask = (
            new_data["frame_number"].isin(frame_numbers)
            & (new_data["fragment"] == fragment)
        )
        extra_data = new_data.loc[mask].copy()

    # Common tail for all replace==True branches that produced `extra_data`
    # via copying. Drop the source rows by index, then queue the modified
    # copy with the standard fields applied.
    new_data = _drop_rows_by_index(new_data, extra_data)
    logger.info("Modified %d rows of dataset", extra_data.shape[0])

    extra_data["class_name"] = "undefined"
    extra_data["in_frame_index"] = np.nan
    extra_data["local_identity"] = local_identity
    for key, value in _DEFAULT_ROW_FIELDS.items():
        extra_data[key] = value

    extra_rows.append(extra_data)
    return new_data


def _next_ids_for_new_row(frame_number, chunksize, machine_data, extra_rows):
    """Compute the next fragment id and in_frame_index for a brand-new row.

    Looks across both the original machine data and any rows already queued
    in `extra_rows`, so that successive new rows in the same call don't
    collide on ids.
    """
    chunk = frame_number // chunksize
    chunk_ref = machine_data.loc[machine_data["chunk"] == chunk]
    frame_ref = chunk_ref.loc[chunk_ref["frame_number"] == frame_number]

    max_fragment = chunk_ref["fragment"].max()
    max_in_frame_index = frame_ref["in_frame_index"].max()

    if extra_rows:
        queued = pd.concat(extra_rows, axis=0, ignore_index=True)
        queued_chunk = queued.loc[queued["frame_number"] // chunksize == chunk]
        if not queued_chunk.empty:
            max_fragment = np.nanmax([max_fragment, queued_chunk["fragment"].max()])
            queued_frame = queued_chunk.loc[
                queued_chunk["frame_number"] == frame_number
            ]
            if not queued_frame.empty:
                max_in_frame_index = np.nanmax(
                    [max_in_frame_index, queued_frame["in_frame_index"].max()]
                )

    return max_fragment + 1, max_in_frame_index + 1


def _apply_insert(machine_data, row, chunksize, extra_rows):
    """Handle a validation row with replace == False (i.e. insert new rows)."""
    fragment = row["fragment"]
    local_identity = row["local_identity"]
    frame_numbers = _frame_range(row)
    chunk_ref = machine_data.loc[
        machine_data["chunk"] == row["frame_number"] // chunksize
    ]

    for frame_number in frame_numbers:
        chunk = frame_number // chunksize

        if pd.isna(fragment):
            next_fragment, next_in_frame_index = _next_ids_for_new_row(
                frame_number, chunksize, machine_data, extra_rows
            )
            extra_data = pd.DataFrame({
                "frame_number": [frame_number],
                "in_frame_index": [next_in_frame_index],
                "fragment": [next_fragment],
                "modified": [1],
                "class_name": ["undefined"],
                "chunk": [chunk],
            })
        else:
            extra_data = chunk_ref.query(
                "fragment == @fragment and frame_number == @frame_number"
            ).copy()

        extra_data["x"] = row["x"]
        extra_data["y"] = row["y"]
        extra_data["local_identity"] = local_identity
        for key, value in _DEFAULT_ROW_FIELDS.items():
            extra_data[key] = value

        extra_rows.append(extra_data)


def apply_validation_csv_file(
    new_data, machine_data, validation_csv, chunksize, replace=None
):
    """Apply manual validation corrections to `new_data`.

    Parameters
    ----------
    new_data : pd.DataFrame
        The current (already-validated or in-progress) tracking dataframe.
        Expected columns include: frame_number, in_frame_index, local_identity,
        validated, fragment, x, y, modified, class_name, chunk.
    machine_data : pd.DataFrame
        The original machine-produced tracking data, used as a reference when
        inserting brand-new rows (replace==False).
    validation_csv : str | Path
        CSV of manual corrections. Lines starting with '#' are treated as
        comments.
    chunksize : int
        Number of frames per chunk; used to map frame_number -> chunk.
    replace : bool | None
        If given, only rows whose `replace` column matches are applied.

    Returns
    -------
    pd.DataFrame
        The updated tracking dataframe.
    """
    validation = _load_validation_table(validation_csv, replace)
    extra_rows = []

    for _, row in tqdm(
        validation.iterrows(),
        desc="Applying manual validation",
        total=validation.shape[0],
    ):
        if row["replace"]:
            new_data = _apply_replace(new_data, row, chunksize, extra_rows)
        else:
            _apply_insert(machine_data, row, chunksize, extra_rows)

    new_data = new_data.reset_index(drop=True)

    if extra_rows:
        extra_data = pd.concat(extra_rows, axis=0, ignore_index=True)
        new_data = (
            pd.concat([extra_data[new_data.columns], new_data], axis=0)
            .reset_index(drop=True)
            .sort_values(
                ["frame_number", "validated", "local_identity"],
                ascending=[True, False, True],
            )
        )

    # Drop validated rows with no assigned identity — these are leftovers
    # from prior tracking that got superseded by a manual correction.
    drop_mask = new_data["local_identity"].isna() & (new_data["validated"] > 0)
    new_data = new_data.loc[~drop_mask]

    return new_data



# def apply_validation_csv_file(new_data, machine_data, validation_csv, chunksize, replace=None):
#     extra_rows=[]
#     #columns
#     # frame_number  in_frame_index  local_identity  validated  fragment           x           y  modified class_name  chunk

#     manual_validation=pd.read_csv(validation_csv, comment="#")
#     if replace is not None:
#         manual_validation=manual_validation.loc[manual_validation["replace"]==replace]

#     for _, manual_validation in tqdm(manual_validation.iterrows(), desc="Applying manual validation", total=manual_validation.shape[0]):

#         frame_number=manual_validation["frame_number"]
#         chunk=frame_number//chunksize
#         fragment=manual_validation["fragment"]
#         replace_row=manual_validation["replace"]
#         local_identity=manual_validation["local_identity"]

#         if replace_row:
#             if manual_validation.get("by_identity", True):
#                 extra_data=new_data.loc[((new_data["frame_number"]==frame_number)&(new_data["local_identity"]==local_identity))]
#                 extra_data["fragment"]=np.nan

#             else:
#                 if np.isnan(manual_validation.get("first_frame_number", np.nan)):
#                     extra_data=new_data.loc[((new_data["frame_number"]==frame_number)&(new_data["fragment"]==fragment))].copy()
#                     if extra_data.shape[0]==0:
#                         logger.warning("Ignoring line in validation.csv")
#                         logger.warning(manual_validation)
                       
#                 else:
#                     frame_numbers=np.arange(manual_validation["first_frame_number"], manual_validation["last_frame_number"]+1)
#                     if np.isnan(fragment):
#                         extra_data=pd.DataFrame({"frame_number": frame_numbers})
#                         extra_data["fragment"]=np.nan
#                         extra_data["chunk"]=chunk
#                         extra_data["x"]=manual_validation["x"]
#                         extra_data["y"]=manual_validation["y"]
#                         extra_data["local_identity"]=local_identity
#                         extra_data["is_a_crossing"]=False
#                         extra_data["validated"]=1
#                         extra_data["in_frame_index"]=np.nan
#                         extra_data["modified"]=1
#                         extra_data["class_name"]="undefined"
#                         extra_data["frame_validated"]=False
#                         index=((new_data["frame_number"].isin(frame_numbers))&(new_data["local_identity"]==local_identity))
#                         nrows_removed=index.sum()
#                         print(f"Removing {nrows_removed} rows")
#                         new_data=new_data.loc[~index]
#                         extra_rows.append(extra_data)
#                         continue

#                     else:
#                         # Build a clear boolean mask for the rows of interest
#                         is_same_fragment_and_frame = (
#                             new_data["frame_number"].isin(frame_numbers)
#                             & (new_data["fragment"] == fragment)
#                         )

#                         # Rows matching the condition
#                         extra_data = new_data.loc[is_same_fragment_and_frame].copy()
#                         # All the other rows
#                         # new_data = new_data.loc[~is_same_fragment_and_frame].copy()


#             extra_data["class_name"]="undefined"
#             extra_data["in_frame_index"]=np.nan
#             nrows=new_data.shape[0]
 
#             foo=new_data.merge(extra_data[[]], left_index=True, right_index=True, how="outer", indicator=True)
#             new_data=foo.loc[foo["_merge"]=="left_only"].drop("_merge", axis=1)
#             new_nrows=new_data.shape[0]
#             # assert nrows-new_nrows==extra_data.shape[0]
#             logger.info("Modified %s rows of dataset", extra_data.shape[0])
#             del foo

#             extra_data["local_identity"]=local_identity
#             extra_data["is_a_crossing"]=False
#             extra_data["validated"]=1
#             extra_data["frame_validated"]=False
#             extra_rows.append(extra_data)

#         # dont replace_row
#         else:
#             extra_data_all=machine_data.loc[(machine_data["chunk"]==chunk)]
#             if np.isnan(manual_validation.get("first_frame_number", np.nan)):
#                 frame_numbers=[frame_number]
#             else:
#                 frame_numbers=np.arange(
#                     manual_validation["first_frame_number"],
#                     manual_validation["last_frame_number"]+1
#                 )

#             for frame_number in frame_numbers:

#                 if extra_rows:
#                     extra_data_temp=pd.concat(extra_rows, axis=0)\
#                         .reset_index(drop=True)
#                 else:
#                     extra_data_temp=None

#                 if np.isnan(fragment):
#                     frame_ref_data=extra_data_all.loc[
#                         extra_data_all["frame_number"]==frame_number
#                     ]
#                     chunk_ref_data=extra_data_all.loc[
#                         extra_data_all["frame_number"]//chunksize==frame_number//chunksize
#                     ]
#                     if extra_data_temp is not None:
#                         extra_data_temp_ref=extra_data_temp.loc[
#                             extra_data_temp["frame_number"]//chunksize==frame_number//chunksize
#                         ]
#                     else:
#                         extra_data_temp_ref=None

#                     max_fragment=chunk_ref_data["fragment"].max()
#                     max_in_frame_index=frame_ref_data["in_frame_index"].max()
#                     if extra_data_temp_ref is not None and extra_data_temp_ref.shape[0]>0:
#                         max_fragment=max(max_fragment, extra_data_temp_ref["fragment"].max())
#                         max_in_frame_index=max(max_in_frame_index, extra_data_temp_ref.loc[
#                             extra_data_temp_ref["frame_number"]==frame_number
#                         ]["in_frame_index"].max())

#                     extra_data=pd.DataFrame({
#                         "frame_number": [frame_number],
#                         "in_frame_index": [max_in_frame_index+1],
#                         "fragment": [max_fragment+1],
#                         "modified": [1],
#                         "class_name": ["undefined"],
#                         "chunk": [chunk]
#                     })
#                 else:
#                     # extra_data=extra_data_all.loc[(extra_data_all["fragment"]==fragment)&(extra_data_all["frame_number"]==frame_number)].copy()
#                     extra_data=extra_data_all.query("(fragment == @fragment) and (frame_number == @frame_number)").copy()

#                 extra_data["x"]=manual_validation["x"]
#                 extra_data["y"]=manual_validation["y"]
#                 extra_data["local_identity"]=local_identity
#                 extra_data["is_a_crossing"]=False
#                 extra_data["validated"]=1
#                 extra_data["frame_validated"]=False
#                 extra_rows.append(extra_data)
#                 print(len(extra_rows))

#     new_data.reset_index(drop=True, inplace=True)
    
#     if extra_rows:
#         extra_data=pd.concat(extra_rows, axis=0).reset_index(drop=True)
#         new_data=pd.concat([
#             extra_data[new_data.columns],
#             new_data,
#         ], axis=0).reset_index(drop=True).sort_values(
#             ["frame_number", "validated", "local_identity"],
#             ascending=[True, False, True]
#         )

#     new_data=new_data.loc[~((new_data["local_identity"].isna()) & (new_data["validated"]>0))]

#     return new_data
