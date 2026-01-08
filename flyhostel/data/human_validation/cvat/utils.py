import logging
import os.path
import sqlite3
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from flyhostel.utils.utils import get_dbfile, get_chunksize

logger=logging.getLogger(__name__)

def assign_in_frame_indices_old(data, number_of_animals, experiment=None):
    """
    Populate in_frame_index column so that no nan values are left
    """
    if experiment is None:
        chunksize=None
    else:
        chunksize=get_chunksize(experiment)

    data.reset_index(drop=True, inplace=True)
    if "in_frame_index" not in data.columns:
        data["in_frame_index"]=np.nan
    fn_index=data.loc[data["in_frame_index"].isna(), "frame_number"].unique().tolist()
    rows_to_annotate=data.loc[(data["frame_number"].isin(fn_index))]
    data_ok=data.drop(index=rows_to_annotate.index)

    new_rows=[]
    for frame_number in tqdm(fn_index, desc="Assign in frame index"):
        one_frame_data=rows_to_annotate.loc[rows_to_annotate["frame_number"]==frame_number]
        try:
            last_in_frame_index=np.nanmax(one_frame_data["in_frame_index"])
        except AttributeError as error:
            if one_frame_data.shape[0]!=number_of_animals:
                raise error
            else:
                last_in_frame_index=-1

        if np.isnan(last_in_frame_index):
            last_in_frame_index=-1
        for i, row in one_frame_data.iterrows():
            if np.isnan(row["in_frame_index"]):
                one_frame_data.loc[i, "in_frame_index"]=last_in_frame_index+1
                last_in_frame_index+=1
        new_rows.append(one_frame_data)
        
    data=pd.concat([data_ok] + new_rows, axis=0)

    sorting_columns=["frame_number"]
    if "local_identity" in data.columns:
        sorting_columns+=["local_identity"]

    data["in_frame_index"]=data["in_frame_index"].astype(int)
    data.sort_values(sorting_columns, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data

def assign_in_frame_indices(data, number_of_animals, experiment=None):
    """
    Populate in_frame_index column so that no nan values are left
    """
    if experiment is None:
        chunksize=None
    else:
        chunksize=get_chunksize(experiment)

    data_orig = data.copy()

    data.reset_index(drop=True, inplace=True)
    if "in_frame_index" not in data.columns:
        data["in_frame_index"]=np.nan

    fn_missing_index=data.loc[data["in_frame_index"].isna(), "frame_number"].drop_duplicates().values.tolist()
    counts=data.value_counts(["frame_number", "in_frame_index"])
    fn_duplicated_index=counts[counts>1].index.get_level_values("frame_number").values.tolist()
    fn_index=list(set(fn_missing_index + fn_duplicated_index))

    rows_to_annotate=data.loc[(data["frame_number"].isin(fn_index))]
    data_ok=data.drop(index=rows_to_annotate.index)

    new_rows=[]
    for frame_number in tqdm(fn_index, desc="Assign in frame index"):
        one_frame_data=rows_to_annotate.loc[rows_to_annotate["frame_number"]==frame_number]
        one_frame_data["in_frame_index"]=np.arange(one_frame_data.shape[0])
        new_rows.append(one_frame_data)

    data=pd.concat([data_ok] + new_rows, axis=0)

    sorting_columns=["frame_number"]
    if "local_identity" in data.columns:
        sorting_columns+=["local_identity"]
    data.sort_values(sorting_columns, inplace=True)
    data.reset_index(drop=True, inplace=True)

    counts=data.groupby(["frame_number", "in_frame_index"]).size().reset_index(name="count")
    mistakes=counts.loc[counts["count"]>1].groupby("frame_number").first().reset_index()
    if mistakes.shape[0]>0:
        assert chunksize is not None
        mistakes["chunk"]=mistakes["frame_number"]//chunksize
        print(mistakes)
        raise ValueError(f"""
        Mistakes: {mistakes}
        """)

    return data

def load_original_resolution(basedir):

    dbfile=get_dbfile(basedir)
    with sqlite3.connect(dbfile) as conn:
        original_resolution=pd.read_sql_query(sql="SELECT value FROM METADATA WHERE field IN ('frame_width', 'frame_height');", con=conn).values.flatten().tolist()
        original_resolution=[int(e) for e in original_resolution]
    return original_resolution

def load_tracking_data(basedir, frame_numbers):
    dbfile=get_dbfile(basedir)
    frame_numbers_str="(" + ",".join([str(fn) for fn in frame_numbers]) + ")"
    with sqlite3.connect(dbfile) as conn:
        roi0=pd.read_sql_query(sql=f"SELECT * FROM ROI_0 WHERE frame_number IN {frame_numbers_str};", con=conn)
        ident=pd.read_sql_query(sql=f"SELECT * FROM IDENTITY WHERE frame_number IN {frame_numbers_str};", con=conn)
    tracking_data=roi0.drop("id", axis=1).merge(ident.drop("id", axis=1), on=["frame_number", "in_frame_index"])
    return tracking_data

def load_machine_data(basedir, where=""):
    dbfile=get_dbfile(basedir)
    with sqlite3.connect(dbfile) as conn:
        cmd=f"SELECT * FROM ROI_0 {where};"
        logger.debug(cmd)
        roi_0_table=pd.read_sql_query(sql=cmd, con=conn)
        logger.debug("Done")

        cmd=f"SELECT * FROM IDENTITY {where};"
        logger.debug(cmd)
        identity_table=pd.read_sql_query(sql=cmd, con=conn)
        logger.debug("Done")

    return identity_table, roi_0_table

def annotate_crossings(ident, roi0, crossings):
    ident_crossings=ident.groupby(["frame_number", "in_frame_index"]).size().reset_index()
    ident_crossings.columns=["frame_number", "in_frame_index", "size"]
    ident_crossings["is_a_crossing"]=False
    ident_crossings.loc[
        ident_crossings["size"]>1,
        "is_a_crossing"
    ]=True
    ident=ident.merge(ident_crossings[["frame_number", "in_frame_index", "is_a_crossing"]], on=["frame_number", "in_frame_index"])
    return ident


def select_zt(data, chunksize, min_t, max_t):
    chunk_time=pd.read_csv("2024-02-10_validation-videos_0.1/chunk_time.csv", index_col=0)
    data["chunk"]=data["frame_number"]//chunksize
    data=data.drop("zt", axis=1, errors="ignore").merge(chunk_time[["chunk", "zt"]], on="chunk")
    data=data.loc[(data["zt"]>=min_t/3600)&(data["zt"]<max_t/3600)]
    return data

