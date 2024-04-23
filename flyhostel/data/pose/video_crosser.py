import os.path
from abc import ABC
import logging

from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from flyhostel.utils import get_sqlite_file, get_local_identities, get_single_animal_video, get_chunksize

logger=logging.getLogger(__name__)


def cross_with_video_data(dt):
    dt["identity"] = [int(e.split("|")[1]) for e in dt["id"]]
    animal=dt["animal"].iloc[0]
    dbfile=get_sqlite_file(animal.split("|")[0] + "*")

    chunksize=get_chunksize(dbfile)

    dt["chunk"]=dt["frame_number"]//chunksize

    dt_chunk=dt[["chunk", "identity", "id"]].drop_duplicates()
    dt_chunk["video"]=None
    dt_chunk["dbfile"]=None
    dt_chunk["local_identity"]=None

 
    for i, row in tqdm(dt_chunk.iterrows()):
        dt_chunk["dbfile"]=get_sqlite_file(row["id"].split("|")[0] + "*")

    frame_numbers=[int(e) for e in sorted(dt["frame_number"].unique())]
    frame_numbers=np.unique(np.array(frame_numbers)//chunksize)*chunksize
    frame_numbers=frame_numbers.tolist()


    table = get_local_identities(dt_chunk.iloc[0]["dbfile"], frame_numbers=frame_numbers)
    dt_chunk["frame_number"]=dt_chunk["chunk"]*chunksize
 
    for i, row in tqdm(dt_chunk.iterrows()):
        video=get_single_animal_video(row["dbfile"], row["frame_number"], table=table, identity=row["identity"], chunksize=chunksize)
        print(i, row["identity"], row["chunk"], video)
        dt_chunk["video"].loc[i]=video
        dt_chunk["local_identity"].loc[i]=int(os.path.basename(os.path.dirname(video)))

    dt=dt.merge(dt_chunk[["dbfile", "id", "video", "chunk", "local_identity"]], on=["id", "chunk"], how="left")

    dt["frame_idx"]=dt["frame_number"]%chunksize
    assert dt.duplicated(["id", "frame_idx", "video"]).sum() == 0
    dt=dt.loc[~dt["video"].isna()]
    dt["video"]=[video.replace("//", "/") for video in dt["video"]]
    return dt

class CrossVideo(ABC):
    """
    Link or cross pose data with flyhostel database to incorporate path to .mp4 videos used to generate the pose 
    TODO Rewrite so it works on a single id at a time (so that it does not assume we have an interactions data frame)
    """

    @staticmethod
    def get_dbfile(dt):
        animal=dt["id1"].iloc[0]
        dbfile=get_sqlite_file(animal.split("|")[0] + "*")
        return dbfile
    
    
    def cross_with_video_data(self, dt):
        dt["identity1"] = [int(e.split("|")[1]) for e in dt["id1"]]
        dt["identity2"] = [int(e.split("|")[1]) for e in dt["id2"]]
        dt["id1"].iloc[0].split("|")[0]
        dt["dbfile"]=[get_sqlite_file(animal.split("|")[0] + "*") for animal in dt["id1"]]
        dt.sort_values("bp_distance_mm", inplace=True)
        dbfile=dt.iloc[0]["dbfile"]

        chunksize=get_chunksize(dbfile)

        frame_numbers=[int(e) for e in sorted(dt["frame_number"].unique())]    
        table = get_local_identities(dt.iloc[0]["dbfile"], frame_numbers=frame_numbers)
        dt["frame_idx"]=dt["frame_number"] % chunksize
        dt["video_time"]=dt["frame_idx"] / 150

        dt["video1"]=None
        dt["video2"]=None

        for i, row in dt.iterrows():
            dt["video1"].loc[i]=get_single_animal_video(row["dbfile"], row["frame_number"], table=table, identity=row["identity1"], chunksize=chunksize)
            dt["video2"].loc[i]=get_single_animal_video(row["dbfile"], row["frame_number"], table=table, identity=row["identity2"], chunksize=chunksize)

        dt=dt.loc[~pd.isna(dt["video1"])]
        return dt
