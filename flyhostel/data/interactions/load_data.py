import os
import sqlite3
import pandas as pd
import numpy as np
from flyhostel.utils import get_sqlite_file

def read_animal_position(sqlite3_file, identity, chunks):

    placeholders = ', '.join('?' for _ in chunks)
    
    with sqlite3.connect(sqlite3_file) as conn:
        cur=conn.cursor()
        cmd=f"""
            SELECT R0.x, R0.y, R0.frame_number
            FROM ROI_0 AS R0
                INNER JOIN STORE_INDEX AS IDX ON R0.frame_number = IDX.frame_number AND IDX.chunk IN ({placeholders}) AND IDX.half_second = 1
                INNER JOIN IDENTITY AS ID ON R0.frame_number = ID.frame_number AND R0.in_frame_index = ID.in_frame_index AND ID.identity = {identity};
        """
        # print(cmd)
        
        cur.execute(cmd, tuple([*chunks]))
        records=cur.fetchall()
    data=pd.DataFrame.from_records(records, columns=["x", "y", "frame_number"])

    return data


def read_animal_speed(*args, first_chunk=50, **kwargs):
    raise NotImplementedError()
    
    data = read_animal_position(*args, **kwargs)
        
    diff=np.diff(data[["x","y"]].values, axis=0)
    speed=np.sqrt((diff**2).sum(axis=1))
    data["speed"]=0

    assert len(speed) == data.shape[0]-1
    data.loc[1:, "speed"]=speed
    data["pose_fn"]=data["frame_number"] - chunksize*first_chunk    
    return data


def add_multiindex(df):
    index = df.index.copy()
    df=df.T
    df.rename_axis(index="coords", inplace=True)
    df["scorer"]="SLEAP"
    df["bodyparts"]="centroid"
    df.set_index(["scorer", "bodyparts"], append=True, inplace=True)
    df=df.T.reorder_levels([1, 2, 0], axis=1)
    df.set_index(index, inplace=True)
    return df


def load_animal_dataset(animal):
    raise NotImplementedError()
    #DATA_PATH=os.environ["MOTIONMAPPER_DATA"]

    identity = int(animal.split("__")[1])
    datasetnames = [
        animal
    ]
    positions = [f"{DATA_PATH}/{file}_positions.h5" for file in datasetnames]

    df=pd.read_hdf(positions[datasetnames.index(animal)], key="pose")
    index=pd.read_hdf(positions[datasetnames.index(animal)], key="index")
    sqlite3_file = get_sqlite_file(animal)
    chunks=np.array(list(set(index["chunk"].tolist())))
    chunks=[int(e) for e in chunks]
    
    centroid_data=read_animal_position(
        sqlite3_file=sqlite3_file, identity=identity, chunks=chunks
    )
    centroid_data.set_index("frame_number", inplace=True)
    centroid_data=add_multiindex(centroid_data)
    df=pd.merge(df, centroid_data, left_index=True, right_index=True, how="outer")
    df.loc[:, pd.IndexSlice[:, "centroid", "x"]]=df.loc[:, pd.IndexSlice[:, "centroid", "x"]].ffill()
    df.loc[:, pd.IndexSlice[:, "centroid", "y"]]=df.loc[:, pd.IndexSlice[:, "centroid", "y"]].ffill()
    return df, index