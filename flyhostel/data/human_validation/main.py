"""
Generate short videos whenever the tracking analysis captured in the dbfile
has issues
"""
import sqlite3
import os.path
import logging

import yaml
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import joblib

from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.human_validation.qc import analyze_video
from flyhostel.data.human_validation.video import generate_validation_video
from zeitgeber.rle import encode

logger=logging.getLogger(__name__)


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
        statement="WHERE" + " AND ".join(parts)
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

def load_store_index(conn, min_frame_number=None, max_frame_number=None):
    where_statement=write_where_statement(min_frame_number, max_frame_number)
    store_index=pd.read_sql_query(con=conn, sql=f"SELECT * FROM STORE_INDEX {where_statement}")
    return store_index

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
        # last_chunk=roi0["frame_number"].iloc[-1]//CHUNKSIZE
        store_index=load_store_index(conn,
            min_frame_number=min_frame_number,
            max_frame_number=max_frame_number
        )
    
    store_index["t"]=store_index["frame_time"]/1000
    data=identity.merge(roi0, on=["frame_number", "in_frame_index"]).sort_values(["frame_number", "fragment"])
    data["chunk"]=data["frame_number"]//CHUNKSIZE
    data["frame_idx"]=data["frame_number"]%CHUNKSIZE

    # Annotate time of frame number
    data=data.merge(store_index[["frame_number", "t"]], on=["frame_number"])
    return data

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


def annotate_for_validation(
        experiment, output_folder,
        time_window_length=1,
        format=".png", n_jobs=20,
        min_frame_number=None,
        max_frame_number=None,
        cache=False,
    ):
    """
    Entrypoint

    Generate movies capturing each scene where validation may be needed based on heuristics and QC
    """

    tokens=experiment.split("_")
    suffix="/".join([tokens[0], tokens[1], "_".join(tokens[2:4])])

    number_of_animals=int(tokens[1].replace("X",""))

    basedir=os.path.join(os.environ["FLYHOSTEL_VIDEOS"], suffix)
    store_path=os.path.join(basedir, "metadata.yaml")
    dbfile=os.path.join(basedir, experiment + ".db")
    assert os.path.exists(dbfile), f"{dbfile} not found"

    tracking_fields=["modified", "fragment", "x", "y"]
    field="local_identity"
    output_path_feather_bin=os.path.join(output_folder, experiment + f"_machine-validation-index-{time_window_length}-s.feather")
    output_path_feather_df=os.path.join(output_folder, experiment + f"_tracking_data.feather")
    output_path_csv=os.path.join(output_folder, experiment + f"_machine-qc-index-{time_window_length}-s.csv")

    df_bin=None
    df=None
    qc_fail=None


    if os.path.exists(output_path_feather_df) and cache:
        logger.debug("Loading %s", output_path_feather_df)
        df=pd.read_feather(output_path_feather_df)
        if min_frame_number is not None:
            df=df.loc[(df["frame_number"] >= min_frame_number)]
        if max_frame_number is not None:
            df=df.loc[(df["frame_number"] < max_frame_number)]
    else:
        os.makedirs(output_folder, exist_ok=True)
        df=load_data(
            dbfile, tracking_fields,
            min_frame_number=min_frame_number,
            max_frame_number=max_frame_number
        )
        df = update_identity(df.copy(), field=field, n_jobs=n_jobs)
        logger.debug("Generating identogram label")
        df=generate_label(df)
        if cache:
            df.reset_index(drop=True).to_feather(output_path_feather_df)

    if os.path.exists(output_path_feather_bin) and cache:
        df_bin=pd.read_feather(output_path_feather_bin)

        if min_frame_number is not None:
            df_bin=df_bin.loc[
                (df_bin["frame_number"] >= min_frame_number)
            ]

        if max_frame_number is not None:
            df_bin=df_bin.loc[
                (df_bin["frame_number"] < max_frame_number)
            ]
    else:
        
        logger.debug("Binning into %s second windows", time_window_length)
        df_bin=bin_windows(df, time_window_length=time_window_length)

    logger.debug("Running QC of experiment %s", experiment)
    
    qc, tests=analyze_video(df_bin.copy(), number_of_animals, n_jobs=n_jobs)
    logger.info("%s %% of %s passes QC", round(100*qc["qc"].mean(), 2), experiment)

    df_bin=df_bin.drop(tests, axis=1, errors="ignore").merge(qc[["frame_number", "chunk"] + tests], on=["chunk", "frame_number"], how="left")

    qc_rle=pd.DataFrame.from_records(encode([str(e)[0] for e in qc["qc"].values]), columns=["status", "length"])
    qc_rle["duration"]=qc_rle["length"]*time_window_length
    qc_rle["experiment"]=experiment


    qc_rle["index"]=[0] + qc_rle["length"].cumsum().tolist()[:-1]
    qc_rle["frame_number"]=qc["frame_number"].iloc[qc_rle["index"]].values
    qc_rle["chunk"]=qc["chunk"].iloc[qc_rle["index"]].values

    for qc_col in tests:
        qc_rle[qc_col]=qc[qc_col].iloc[qc_rle["index"]].values
        
    qc_fail=qc_rle.loc[qc_rle["status"]=="F"]
    
    if cache and not os.path.exists(output_path_feather_bin):
        df_bin.to_feather(output_path_feather_bin)
    
    qc_fail.to_csv(output_path_csv)

    kwargs=[]

    for i, row in qc_fail.iterrows():
        frame_number_0=row["frame_number"]
        duration=row["duration"]
        frame_number_last=frame_number_0+duration*FRAMERATE

        tracking_data=df.loc[
            (df["frame_number"]>=frame_number_0) &
            (df["frame_number"]<=frame_number_last)
        ]
        if df.shape[0]>0:
            kwargs.append({"row": row, "tracking_data": tracking_data})

    logger.debug("Will generate %s videos", len(kwargs))

    movies_folder=os.path.join(output_folder, "movies")
    os.makedirs(movies_folder, exist_ok=True)

    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            generate_validation_video
        )(
            store_path, row=kwargs[i]["row"], df=kwargs[i]["tracking_data"],
            number_of_animals=number_of_animals, output_folder=movies_folder,
            framerate=FRAMERATE, format=format, field=field
        )
        for i in range(len(kwargs))
    )
    return df, df_bin, qc_fail


def bin_windows(df, time_window_length=1):
    """
    Bin the tracking results into brief bins. Every unique entry of local identity identity chunk fragment or modified
    gets a separate bin
    """
    df["t_round"]=time_window_length*(df["t"]//time_window_length)

    df_bin=df.drop_duplicates(["t_round","local_identity", "identity", "chunk", "fragment", "modified"])[[
        "t_round","local_identity", "identity", "chunk", "fragment", "modified", "x", "y"
    ]]

    df_bin=df_bin.merge(
        df[["t_round", "frame_number", "chunk", "frame_idx"]].groupby(["t_round", "chunk"]).first().reset_index(),
        on=["t_round", "chunk"],
        how="left"
    )
    df_bin.sort_values(["chunk", "frame_number", "local_identity"], inplace=True)
    return df_bin


# Update local_identity
def update_identity(df, field="identity", n_jobs=1):
    """
    Replaces all instances of local identity = 0 by 
    negative local identities so that no two flies have the same local identity=0
    """

    frame_number_with_missing_flies=df.loc[df["local_identity"]==0, "frame_number"].unique()
    shape=df.shape

    df_ok=df.loc[~df["frame_number"].isin(frame_number_with_missing_flies)].copy()
    df_not_ok=df.loc[df["frame_number"].isin(frame_number_with_missing_flies)].copy()
    del df

    if df_not_ok.shape[0] > 0:
        df_not_ok.sort_values(["frame_number", "fragment"], inplace=True)
        diff=np.diff(df_not_ok["frame_number"])
        new_scene=np.concatenate([
            [True],
            diff>1
        ])
        df_not_ok["scene"]=np.cumsum(new_scene)

        dfs=update_identity_in_all_scenes(df_not_ok, n_jobs=n_jobs, field=field)

        df_not_ok=pd.concat(dfs, axis=0)
        df=pd.concat([
            df_ok,
            df_not_ok
        ], axis=0)
    else:
        df=df_ok

    df.sort_values(["frame_number", "fragment"], inplace=True)

    assert shape[0]==df.shape[0], f"{shape[0]}!={df.shape[0]}"

    return df


def update_identity_in_all_scenes(df, n_jobs, field="identity"):
    dfs=joblib.Parallel(
        n_jobs=n_jobs
    )(
        joblib.delayed(
            update_identity_in_scene
        )(
            df_scene.copy(), field=field, scene_id=scene_id
        )
        for scene_id, df_scene in df.groupby("scene")
    )
    return dfs


def update_identity_in_scene(df, field="identity", scene_id=None):

    # Track the last used identity for animals with identity == 0
    counter = 0
    # Dictionary to store the fragment and its updated identity
    fragment_identity = {}

    out_df=df.copy()

    for index, row in tqdm(df.iterrows()):

        fragment_identifier=row['fragment'].item()
        if row[field] == 0:
            if fragment_identifier in fragment_identity:
                # If fragment already encountered, use the stored identity
                out_df.at[index, field] = fragment_identity[fragment_identifier]
            else:
                # Assign new identity and update the counter and dictionary
                counter-=1
                out_df.at[index, field] = counter
                fragment_identity[fragment_identifier] = counter

    if scene_id is not None:
        logfile=os.path.join(
            "logs",
            str(scene_id).zfill(6) + "_" + str(df["frame_number"].iloc[0]) + "_status.txt"
        )

        metadata={
            "scene_size": df.shape[0],
            "min_counter": counter,
            "fragment_identity": fragment_identity
        }
        logger.debug(metadata)
        with open(logfile, "w") as handle:
            yaml.dump(metadata, handle, yaml.SafeDumper)

    return out_df
