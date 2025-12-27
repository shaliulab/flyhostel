import pandas as pd
import os.path
import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict
import sleap.io
from flyhostel.data.pose.sleap import make_slp_with_frames_for_labeling
from flyhostel.utils import get_chunksize, get_single_animal_video, get_basedir, get_dbfile, get_local_identities

def validation(experiment):
    chunksize=get_chunksize(experiment)
    basedir=get_basedir(experiment)
    dbfile=get_dbfile(get_basedir(experiment))
    database=pd.read_feather("pose_database.feather")
    index=pd.read_csv("frame_index.csv", index_col=0)

    database["frame_number"]=database["frame_number"].astype(np.int64)
    database["chunk"]=database["frame_number"]//chunksize
    database["frame_idx"]=database["frame_number"]%chunksize
    database["identity"]=database["animal"].str.slice(-2).astype(int)

    # these are the frames where an interactin occurs
    frame_numbers=database["frame_number"].unique()

    # instead of generating a table using the chunks where these frames are
    # use all chunks between the min and max frame (even if there is no interaction)
    # this defends against edge cases wehre t+1 might be in the next chunk and that chunk is not
    # included in the table
    frame_numbers=np.arange(
        max(frame_numbers.min()//chunksize-1, 1),
        frame_numbers.max()//chunksize+1
    )*chunksize

    table = get_local_identities(dbfile, frame_numbers=frame_numbers)

    dfs=[]
    for t in tqdm(database["t"].unique(), desc="Regenerating frame index from database"):
        start=index.query(f"t == {t}")
        stop=index.query(f"t == {t+1}")
        if start.shape[0]>0 and stop.shape[0]>0:
            start_frame=start["frame_number"].item()
            stop_frame=stop["frame_number"].item()
            assert start_frame<stop_frame
            df=pd.DataFrame({"frame_number": np.arange(start_frame, stop_frame)})
            df["t"]=t
        dfs.append(df)
    df=pd.concat(dfs, axis=0).reset_index(drop=True)
    df2=[]
    for animal in database["animal"].unique():
        for bodypart in database["bodypart"].unique():
            dff=df.copy()
            dff["animal"]=animal
            dff["bodypart"]=bodypart
            df2.append(dff)


    df=pd.concat(df2, axis=0).reset_index(drop=True)
    database2=df.merge(database, on=["t", "frame_number", "animal", "bodypart"], how="left")
    database2["frame_number"]=database2["frame_number"].astype(np.int64)

    database2["frame_idx"]=database2["frame_number"]%chunksize
    database2["chunk"]=database2["frame_number"]//chunksize
    database2=database2.groupby(["animal", "bodypart"]).apply(lambda df: df.ffill().bfill()).reset_index(drop=True)

    video_path_index=database2[["frame_number", "identity"]]
    video_path_index["chunk"]=video_path_index["frame_number"]//chunksize
    video_path_index=video_path_index[["chunk", "identity"]].drop_duplicates()

    video_path_index["video_path"]=[
        get_single_animal_video(basedir, int(row["chunk"]*chunksize), table, row["identity"], chunksize)
        for _, row in tqdm(video_path_index.iterrows(), total=video_path_index.shape[0], desc="Linking to cached .slp files")
    ]
    video_not_found=video_path_index.loc[video_path_index["video_path"].isna()]

    if video_not_found.shape[0]>0:
        raise ValueError(f"""
        Videos not found: {video_not_found}
        """)


    database2=database2.merge(video_path_index, on=["chunk", "identity"], how="left")
    index=database2[["video_path", "frame_idx", "identity", "chunk"]].drop_duplicates()

    PARTITION_SIZE=100
    index["partition"]=index["chunk"]//PARTITION_SIZE
    for identity in index["identity"].unique():
        for partition in index["partition"].unique():
            video_to_frames=OrderedDict()
            index_loop=index.query("identity == @identity & partition == @partition")
            videos=sorted(index_loop["video_path"].unique())
            
            for video in tqdm(videos, desc="Building video_to_frames map"):
                video_to_frames[os.path.realpath(video)]=index_loop.loc[index_loop["video_path"]==video, "frame_idx"]

            sleap_file=f"predictions_{str(partition*PARTITION_SIZE).zfill(6)}s__{str(int(identity)).zfill(2)}.slp"
            
            ref_labels={
                path: sleap.io.dataset.Labels.load_file(filename=path + ".predictions.slp")
                for path in tqdm(video_to_frames, desc="Loading from cached .slp files")
            }
            
            make_slp_with_frames_for_labeling(video_to_frames, sleap_file, ref_labels=ref_labels, save_suggestions=True)
