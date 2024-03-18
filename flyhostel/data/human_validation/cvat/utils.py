import logging
import os.path
import sqlite3

import pandas as pd

logger=logging.getLogger(__name__)

def get_dbfile(basedir):
    dbfile=os.path.join(
        basedir,
        "_".join(basedir.rstrip(os.path.sep).split(os.path.sep)[-3:]) + ".db"
    )
    assert os.path.exists(dbfile), f"{dbfile} not found"
    return dbfile

def get_basedir(experiment):
    tokens = experiment.split("_")
    basedir=f"/flyhostel_data/videos/{tokens[0]}/{tokens[1]}/{'_'.join(tokens[2:4])}"
    return basedir

def get_number_of_animals(experiment):
    tokens = experiment.split("_")
    number_of_animals=int(tokens[1].rstrip("X"))
    return number_of_animals

def load_original_resolution(basedir):

    dbfile=get_dbfile(basedir)
    # store_path=os.path.join(basedir, "metadata.yaml")
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

