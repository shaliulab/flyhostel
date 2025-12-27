import os.path
import logging
import sqlite3
import pandas as pd
import numpy as np
from flyhostel.utils import get_chunksize
from flyhostel.data.human_validation.update_identity import update_identity
from flyhostel.utils.utils import get_first_frame, get_last_frame 

logger=logging.getLogger()
TRACKING_FIELDS=["modified", "fragment", "x", "y"]
FIELD="local_identity"


def check_if_validated(dbfile):

    with sqlite3.connect(dbfile) as conn:
        cur=conn.cursor()
        cur.execute(
            """SELECT name FROM sqlite_master  
            WHERE type='table';"""
        )
        tables=[value[0] for value in cur.fetchall()]

        if "IDENTITY_VAL" in tables:
            return "_VAL"
        else:
            return ""

def get_identity(number_of_animals, dbfile, local_identity, chunk):
    if number_of_animals==1:
        identity="0"
        validated=True
    else:
        if "_VAL" == check_if_validated(dbfile):
            validated=True
            concatenation_table="CONCATENATION_VAL"
        else:
            logger.warning("%s not validated", dbfile)
            validated=False
            concatenation_table="CONCATENATION"
        with sqlite3.connect(dbfile) as conn:
            sql=f"SELECT identity FROM {concatenation_table} WHERE chunk = {chunk} AND local_identity = {local_identity};"
            identity=str(pd.read_sql(con=conn, sql=sql).iloc[0].item())
    return identity, validated


def load_store_index(conn, min_frame_number=None, max_frame_number=None):
    where_statement=write_where_statement(min_frame_number, max_frame_number)
    store_index=pd.read_sql_query(con=conn, sql=f"SELECT * FROM STORE_INDEX {where_statement}")
    return store_index

def write_where_statement(min_frame_number, max_frame_number):
    min_statement=None
    max_statement=None

    if min_frame_number is not None:
        min_statement=f"frame_number >= {min_frame_number}"
    if max_frame_number is not None:
        max_statement=f"frame_number < {max_frame_number}"
    
    if min_statement is None and max_statement is None:
        return ""
    else:
        parts=[part for part in [min_statement, max_statement] if part is not None]
        statement="WHERE " + " AND ".join(parts)
        return statement


def load_identity(conn, min_frame_number=None, max_frame_number=None):
    where_statement=write_where_statement(min_frame_number, max_frame_number)
    identity=pd.read_sql_query(
        con=conn, sql=f"SELECT * FROM IDENTITY {where_statement};"
    )
    return identity

def load_roi0(conn, tracking_fields, min_frame_number=None, max_frame_number=None):
    where_statement=write_where_statement(min_frame_number, max_frame_number)
    roi0=pd.read_sql_query(
        con=conn, sql=f"""
        SELECT frame_number, in_frame_index, {','.join(tracking_fields)} FROM ROI_0 {where_statement};
    """)
    return roi0


def load_data(dbfile, tracking_fields, min_frame_number=None, max_frame_number=None):

    with sqlite3.connect(dbfile) as conn:
        logger.debug("Loading from %s - IDENTITY", dbfile)
        identity=load_identity(conn,
            min_frame_number=min_frame_number,
            max_frame_number=max_frame_number
        )
        logger.debug("Loading from %s - ROI_0", dbfile)
        roi0=load_roi0(
            conn, tracking_fields=tracking_fields,
            min_frame_number=min_frame_number,
            max_frame_number=max_frame_number
        )

        logger.debug("Loading from %s - STORE_INDEX", dbfile)
        store_index=load_store_index(conn,
            min_frame_number=min_frame_number,
            max_frame_number=max_frame_number
        )
    
    chunksize=get_chunksize(dbfile=dbfile)

    store_index["t"]=store_index["frame_time"]/1000
    data=identity.merge(roi0, on=["frame_number", "in_frame_index"]).sort_values(["frame_number", "fragment"])
    data["chunk"]=data["frame_number"]//chunksize
    data["frame_idx"]=data["frame_number"]%chunksize

    # Annotate time of frame number
    data=data.merge(store_index[["frame_number", "t"]], on=["frame_number"])
    return data


def annotate_nan_frames(df,  min_frame_number, max_frame_number, chunksize):
    index=pd.DataFrame({"frame_number": np.arange(min_frame_number, max_frame_number)})
    df["missing_flies"]=False
    df=index.merge(df, on="frame_number", how="left")
    df.loc[df["missing_flies"].isna(), "missing_flies"]=True
    df.loc[df["missing_flies"], "x"]=0
    df.loc[df["missing_flies"], "y"]=0
    df.loc[df["missing_flies"], "local_identity"]=0
    df.loc[df["missing_flies"], "in_frame_index"]=0
    df.loc[df["missing_flies"], "frame_idx"]=df.loc[df["missing_flies"], "frame_number"]%chunksize
    df.loc[df["missing_flies"], "chunk"]=df.loc[df["missing_flies"], "frame_number"]//chunksize
    df["t"].interpolate(inplace=True, method="linear", limit_direction="both")
    return df

def generate_label(df):
    # Generate label for identogram

    df["label"]=df["local_identity"].astype("string")
    df_0=df.loc[df["local_identity"]==0]
    df_0["label"]=[f"fragment_{row['fragment']}" for _, row in df_0.iterrows()]

    df=pd.concat([
        df.loc[df["local_identity"]!=0],
        df_0
    ], axis=0).sort_values(["frame_number", "local_identity"])

    return df

def load_tracking_data(
        dbfile, folder, experiment,
        min_frame_number=None,
        max_frame_number=None,
        n_jobs=1, cache=True
    ):


    output_path_feather_df=os.path.join(folder, experiment + f"_tracking_data.feather")

    chunksize=get_chunksize(experiment)
    if min_frame_number is None:
        min_frame_number=get_first_frame(dbfile)

    if max_frame_number is None:
        max_frame_number=get_last_frame(dbfile)
        

    if os.path.exists(output_path_feather_df) and cache:
        logger.debug("Loading %s", output_path_feather_df)
        df=pd.read_feather(output_path_feather_df)
        if min_frame_number is not None:
            df=df.loc[(df["frame_number"] >= min_frame_number)]
        if max_frame_number is not None:
            df=df.loc[(df["frame_number"] < max_frame_number)]
    else:
        os.makedirs(folder, exist_ok=True)
        df=load_data(
            dbfile, TRACKING_FIELDS,
            min_frame_number=min_frame_number,
            max_frame_number=max_frame_number
        )
        df = update_identity(df.copy(), field=FIELD, n_jobs=n_jobs)
        logger.debug("Generating identogram label")
        df=generate_label(df)
        if cache:
            df.reset_index(drop=True).to_feather(output_path_feather_df)

    # handle frames where no fly is detected
    # such frames may be "invisible" in the dataset
    # because there is no animal found.
    # To make them visible, I need to create one row per such frame
    # where identity lcoal identity x a nd y ares set to NaN
    # Then, the all_id_expected_qc can flag it as problematic
    df=annotate_nan_frames(df, min_frame_number, max_frame_number, chunksize)
    return df