import glob
import sqlite3

import numpy as np
import pandas as pd

from flyhostel.utils import (
    get_chunksize,
    get_basedir,
    get_dbfile
)

def get_chunk_duration_from_dbfile(experiment):
    basedir=get_basedir(experiment)
    dbfile=get_dbfile(basedir)
    chunksize=get_chunksize(experiment)

    with sqlite3.connect(dbfile) as conn:
        cursor=conn.cursor()
        cursor.execute("SELECT frame_number, frame_time FROM STORE_INDEX;")
        df=pd.DataFrame.from_records(cursor.fetchall(), columns=["frame_number", "frame_time"])
    diff=df.loc[(df["frame_number"]+1)%chunksize<=1]["frame_time"].diff().iloc[1::2]/1000
    diff=sorted(diff.values.tolist())
    return diff


def get_chunk_duration_from_npz_files(experiment):
    basedir=get_basedir(experiment)
    chunksize=get_chunksize(experiment)
    
    npz_files=sorted(glob.glob(f"{basedir}/*.npz"))[:-2]
    records=[]
    for file in npz_files:
        arr=np.load(file)
        fts=arr["frame_time"][[0, -1]]
        fns=arr["frame_number"][[0, -1]]
        records.extend(list(zip(fns, fts)))
    
    df=pd.DataFrame.from_records(records, columns=["frame_number", "frame_time"])
    diff=df.loc[(df["frame_number"]+1)%chunksize<=1]["frame_time"].diff().iloc[1::2]/1000
    diff=sorted(diff.values.tolist())
    return diff
