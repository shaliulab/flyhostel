import itertools
import os.path
import logging

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from flyhostel.utils import (
    get_chunksize,
    get_framerate,
    get_basedir,
    get_local_identities,
    get_dbfile,
    get_single_animal_video,
    get_identities,
    annotate_local_identity,
    build_interaction_video_key
)
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.interactions.sociability.image import ImageWriter
from .identify_partner import draw_partner_fly_translucent

logger=logging.getLogger(__name__)
EXTRA_COLS=["t", "pre_longImmobile_duration", "post_longImmobile_duration"]


# def write_separate_videos(experiment, index, videoWriterClass, overwrite=False, root="interactions/videos"):
#     video_writer=None
#     last_video=None
#     last_frame_idx=None
#     last_frame_number=None
#     cap=None

#     for _, row in tqdm(index.iterrows()):
#         key=build_interaction_video_key(experiment, row)

#         video_name=os.path.join(
#             root,
#             key, f"{key}.mp4"
#         )
#         os.makedirs(os.path.dirname(video_name), exist_ok=True)
        
#         if cap is None:
#             cap=cv2.VideoCapture(row["video"])
            
#         elif last_video!=row["video"]:
#             cap.release()
#             cap=cv2.VideoCapture(row["video"])
#         last_video=row["video"]

#         if last_frame_idx is not None and last_frame_idx+1==row["frame_idx"]:
#             ret, frame=cap.read()
#         else:
#             cap.set(1, row["frame_idx"])
#             ret, frame=cap.read()
    
#         if len(frame.shape)==3:
#             frame=frame[:,:,0]
#         last_frame_idx=row["frame_idx"]
#         frame_number=row["frame_number"]
        

#         if last_frame_number is None or last_frame_number+1==frame_number:
#             pass
#         else:
#             video_writer.release()
#             video_writer=None
#         last_frame_number=row["frame_number"]
       

#         if video_writer is None:
#             kwargs={}
#             if videoWriterClass is ImageWriter:
#                 kwargs.update({
#                     "counter": row["first_frame"],
#                     "overwrite": overwrite
#                 })
#             video_writer=videoWriterClass(
#                 video_name,
#                 cv2.VideoWriter_fourcc(*"MPEG"),
#                 FRAMERATE,
#                 frame.shape[:2][::-1],
#                 isColor=False,
#                 **kwargs
#             )
#         video_writer.write(frame)

#     if cap is not None:
#         cap.release()
#     if video_writer is not None:
#         video_writer.release()


def process__make_sure_gray(frame, row):
    if len(frame.shape)==3:
        frame=frame[:,:,0]
    return frame


def process__draw_square_around_partner(frame, row):
    frame=draw_partner_fly_translucent(frame, row, square_size=100, thickness=1, alpha=0.5, blend_to=0)
    return frame


def process__all(frame, row):
    frame=process__make_sure_gray(frame, row)
    frame=process__draw_square_around_partner(frame, row)
    return frame



# def process__crop(frame, row, half_width):

#     x0=row["center_x"]-half_width
#     y0=row["center_y"]-half_width
#     x1=row["center_x"]+half_width
#     y1=row["center_y"]+half_width

#     frame=frame[y0:y1, x0:x1]
#     return frame


def process__crop(frame, row, half_width=100):

    assert len(frame.shape)==2, "Please pass grayscale images"

    h, w = frame.shape  # Get original dimensions

    cx, cy = row["center_x"], row["center_y"]
    x0, y0 = cx - half_width, cy - half_width
    x1, y1 = cx + half_width, cy + half_width

    crop_w, crop_h = x1 - x0, y1 - y0

    # Initialize white canvas
    white_crop = np.full((crop_h, crop_w), 255, dtype=frame.dtype)

    # Compute intersection with frame
    x0_src = max(x0, 0)
    y0_src = max(y0, 0)
    x1_src = min(x1, w)
    y1_src = min(y1, h)

    # Destination coordinates in the padded crop
    x0_dst = x0_src - x0
    y0_dst = y0_src - y0
    x1_dst = x0_dst + (x1_src - x0_src)
    y1_dst = y0_dst + (y1_src - y0_src)

    # Copy valid region into white canvas
    white_crop[y0_dst:y1_dst, x0_dst:x1_dst] = frame[y0_src:y1_src, x0_src:x1_src]

    return white_crop


# def process__all(frame, row, half_width=100):
#     frame=process__make_sure_gray(frame, row)
#     frame=process__crop(frame, row, half_width=half_width)
#     return frame

def write_separate_videos(experiment, index, videoWriterClass, overwrite=False, root="interactions/videos", process_frame=process__all, cache=True, **kwargs):
    """
    index needs to contain:
        1) video (str) path to single_animal video which should be used as INPUT
        2) output_video (str) path to video file where the OUTPUT will be saved
        3) key (str): If output_video is not provided, alternatively the interaction can be provided and the video will be called key.mp4
        4) frame_idx (int): frame in the input video which the information in the row corresponds to
        5) frame_number (int): frame number relative to the beginning of the recording (from chunk 0)

    There needs to be one row per frame to be written in any video
    """
    video_writer=None
    last_video=None
    last_output_video=None
    last_frame_idx=None
    last_frame_number=None
    cap=None

    framerate=get_framerate(experiment)

    for _, row in tqdm(index.iterrows()):

        if "output_video" in row:
            video_name=row["output_video"]
        else:
            key=row["key"]
            video_name=os.path.join(
                root,
                key, f"{key}.mp4"
            )

        os.makedirs(os.path.dirname(video_name), exist_ok=True)

        if cap is None:
            cap=cv2.VideoCapture(row["video"])

        elif last_video!=row["video"]:
            cap.release()
            cap=cv2.VideoCapture(row["video"])
        last_video=row["video"]

        if last_frame_idx is not None and last_frame_idx+1==row["frame_idx"]:
            ret, frame=cap.read()
        else:
            cap.set(1, row["frame_idx"])
            ret, frame=cap.read()

        if process_frame is not None:
            frame=process_frame(frame, row, **kwargs)
        
        last_frame_idx=row["frame_idx"]
        frame_number=row["frame_number"]


        if last_frame_number is None or last_frame_number+1==frame_number:
            pass
        else:
            video_writer.release()
            video_writer=None
        last_frame_number=row["frame_number"]


        if video_writer is None:
            video_writer_kwargs={}
            if videoWriterClass is ImageWriter:
                video_writer_kwargs.update({
                    "counter": row["first_frame"],
                    "overwrite": overwrite
                })

            last_output_video=video_name
            video_writer=videoWriterClass(
                video_name,
                cv2.VideoWriter_fourcc(*"MP4V"),
                framerate,
                frame.shape[:2][::-1],
                isColor=False,
                **video_writer_kwargs
            )
        video_writer.write(frame)

    if cap is not None:
        cap.release()
    if video_writer is not None:
        video_writer.release()


def extend_database(database):

    index=pd.DataFrame.from_records(
        list(itertools.chain(*[
            zip(
                np.arange(row["first_frame"], row["last_frame_number"]),                           # frame_number
                [int(row["id"][-2:]),]*(row["last_frame_number"]-row["first_frame"]),              # identity
                [i,]*(row["last_frame_number"]-row["first_frame"]),                                # index of database
                [row["id"],]*(row["last_frame_number"]-row["first_frame"]),                        # id
                [row["nn"],]*(row["last_frame_number"]-row["first_frame"]),                        # nn
                [row["frame_number"],]*(row["last_frame_number"]-row["first_frame"]),              # frame_number_t
                [row["t"],]*(row["last_frame_number"]-row["first_frame"]),                         # t 
                [row["pre_longImmobile_duration"]]*(row["last_frame_number"]-row["first_frame"]),  # pre_longImmobile_duration
                [row["post_longImmobile_duration"]]*(row["last_frame_number"]-row["first_frame"]), # post_longImmobile_duration
            ) for i, row in database.iterrows()
        ])),
        columns=["frame_number", "identity", "index", "id", "nn", "frame_number_t"] + EXTRA_COLS
    )
    
    index=index.merge(database[["first_frame", "last_frame_number"]].reset_index(), on=["index"])

    return index


def test(index):
    key="FlyHostel3_3X_2025-01-08_17-00-00_rejections_11377257_11399073_003_02"
    index_this_video=index.loc[(index["id"]=="FlyHostel3_3X_2025-01-08_1|02") & (index["first_frame"]==int(key.split("_")[5]))]
    print(index_this_video.shape[0])


def annotate_interaction_database(experiment, database):


    basedir=get_basedir(experiment)
    # from one row per interaction to one row per frame in each interaction
    index=extend_database(database)  
    chunksize=get_chunksize(experiment)
    index["chunk"]=index["frame_number"]//chunksize
    index["frame_idx"]=index["frame_number"]%chunksize

    n_rows=index.shape[0]
    # add local_identity
    index=annotate_local_identity(index, experiment)

    assert index.shape[0]==n_rows, f"{index.shape[0]}!={n_rows}"
    
    missing_lid_count=index["local_identity"].isna().sum()
    assert missing_lid_count==0, f"{missing_lid_count}!=0"

    # add path to input video
    chunk_identity_local_identity_index=index.groupby(["chunk", "identity", "local_identity"]).first().reset_index()
    chunk_identity_local_identity_index["video"]=[
        get_single_animal_video(basedir, row["frame_number"], chunk_identity_local_identity_index.loc[[i]], row["identity"], chunksize)
        for i, row in chunk_identity_local_identity_index[["frame_number", "identity"]].iterrows()
    ]
    index=index.merge(chunk_identity_local_identity_index[["chunk", "identity", "local_identity", "video"]], on=["chunk", "identity", "local_identity"], how="left")
    test(index)
    assert index.shape[0]==n_rows, f"{index.shape[0]}!={n_rows}"

    # add spatial coordinates
    identities=get_identities(experiment)
    loaders=[
        FlyHostelLoader(experiment=experiment, identity=identity) for identity in identities
    ]
    for loader in tqdm(loaders, desc="Loading centroid data"):
        loader.load_centroid_data(cache="/flyhostel_data/cache")

    dt=pd.concat([loader.dt for loader in loaders], axis=0).reset_index(drop=True)
    # get coordinates of centroid
    index=index.merge(dt[["identity", "frame_number", "center_x", "center_y"]], on=["identity", "frame_number"], how="left")
    assert index.shape[0]==n_rows, f"{index.shape[0]}!={n_rows}"
    # get coordinates of partner
    index=index.merge(
        dt[["id", "frame_number", "center_x", "center_y"]].rename({"id": "nn", "center_x": "center_x_nn", "center_y": "center_y_nn"}, axis=1),
        on=["nn", "frame_number"],
        how="left"
    )
    assert index.shape[0]==n_rows, f"{index.shape[0]}!={n_rows}"

    id_missing_coords=index["center_x"].isna().sum()
    nn_missing_coords=index["center_x_nn"].isna().sum()

    if id_missing_coords!=0 or nn_missing_coords!=0:
        logger.warning("%s!=0 OR %s!=0", id_missing_coords, nn_missing_coords)
        logger.warning("Consider remaking the validation and especiallly inspecting these frames:")
        logger.warning(index.loc[(index["center_x"].isna()) | (index["center_x_nn"].isna()), ["id", "frame_number", "chunk"]].drop_duplicates())

    # index.rename({"local_identity": "local_identity_live"}, axis=1, inplace=True)
    index["local_identity_live"]=index["local_identity"].copy()
    index=index.drop("local_identity", axis=1).merge(
        index.groupby("index").first().reset_index()[["index", "local_identity"]],
        on="index",
        how="left"
    )

    index["key"]=[build_interaction_video_key(experiment, row) for _, row in index.iterrows()]
    index.sort_values(["key", "id", "nn", "frame_number"], inplace=True)
    index.reset_index(drop=True, inplace=True)
    test(index)
    return index


def make_rejections_video(experiment, video_writer_f, overwrite=False):
    """
    main entrypoint
    """

    database=pd.read_feather("database.feather")
    index=annotate_interaction_database(experiment, database)


    if video_writer_f is not None:
        write_separate_videos(experiment, index, video_writer_f, overwrite=overwrite)
    return index


def export_videos(experiment):
    index=make_rejections_video(experiment, cv2.VideoWriter, overwrite=True)
    return index


import glob
import joblib
from flyhostel.data.pose.main import init_loaders
from flyhostel.utils.deepethogram import load_predictions_to_df

def load_prediction_file(prediction_file):
    # TODO This should be merged with read_label_file_rejections
    key=os.path.basename(os.path.dirname(prediction_file))
    try:
        df=load_predictions_to_df(prediction_file, "tgmj")
    except AssertionError as error:
        logger.error(error)
        df=None
    
    if df is None:
        return None

    df["key"]=key
    df["touch"]=np.bitwise_or(df["touch_focal"]==1, df["touch_side"]==1)
    return df

def load_deg_data(experiment, metadata={}):

    loaders=init_loaders(experiment, metadata)
    
    if not loaders:
        print(f"Skip {experiment}")
        return [], None

    entries=glob.glob(f"/flyhostel_data/fiftyone/FlyBehaviors/DEG-REJECTIONS/rejections_deepethogram/DATA/{experiment}*")
    prediction_files=[entry + "/" + os.path.basename(entry) + "_outputs.h5" for entry in entries]
    prediction_files=[file for file in prediction_files if os.path.exists(file)]
    all_df=joblib.Parallel(n_jobs=10)(
        joblib.delayed(
            load_prediction_file
        )(prediction_file)
        for prediction_file in tqdm(prediction_files, desc=f"Loading DEG predictions of {experiment}")
    )

    all_df=[df for df in all_df if df is not None]
    if all_df:
        deg_predictions=pd.concat(all_df, axis=0)
        deg_predictions.sort_values(["frame_number", "identity"], inplace=True)
        return loaders, deg_predictions
    else:
        return loaders, None


def find_other_fly(all_flies, partners):
    if len(all_flies)==3:
        other=list(all_flies.difference(partners))
        return other[0]
    elif len(all_flies)<3:
        return np.nan
    else:
        return np.nan

def load_data(experiment, min_time_immobile=300, **kwargs):

    loaders, deg_predictions=load_deg_data(experiment, **kwargs)

    basedir=get_basedir(experiment)
    dbfile=get_dbfile(basedir)
    chunksize=get_chunksize(experiment)
    
    # rejections=pd.read_csv(f"{basedir}/interactions/{experiment}_rejections.csv")
    try:
        database=pd.read_feather(f"{basedir}/interactions/{experiment}_database.feather")
    except FileNotFoundError as error:
        if "1X" in experiment:
            database=None
        else:
            raise error

    for loader in loaders:
        loader.load_behavior_data()
        loader.sleep=loader.compute_sleep_from_behavior(min_time_immobile=min_time_immobile, bin_size=None)

        if database is not None:
            loader.interaction=database[database["id"]==loader.ids[0]]
            loader.interaction["identity"]=loader.interaction["id"].str.slice(start=-2).astype(int)
            loader.interaction["interaction"]=np.arange(loader.interaction.shape[0])
            interaction_index=loader.interaction.groupby("interaction").first().reset_index().sort_values("t")
            interaction_index["t_next"]=np.concatenate([interaction_index["t"].iloc[1:], [np.nan]])
            interaction_index["frame_number_next"]=np.concatenate([interaction_index["frame_number"].iloc[1:], [np.nan]])
        
            loader.interaction=loader.interaction.merge(interaction_index[["interaction", "t_next", "frame_number_next"]], on="interaction")
            loader.interaction["chunk"]=loader.interaction["frame_number"]//chunksize
            
            local_identity_index=get_local_identities(dbfile, loader.interaction["frame_number"])[["identity", "chunk", "local_identity"]]
            loader.interaction=loader.interaction.drop(["local_identity"], errors="ignore", axis=1).merge(local_identity_index, on=["chunk", "identity"], how="left")
            loader.interaction["name"]=np.where(loader.interaction["pre_longImmobile"], "rejections", "interaction")

            loader.interaction["key"]=[build_interaction_video_key(experiment, row) for _, row in loader.interaction.iterrows()]


            if deg_predictions is None:
                loader.interaction["touch_focal"]=np.nan
                loader.interaction["touch_side"]=np.nan
                loader.interaction["touch"]=np.nan
            else:
                    
                deg_predictions_summ=deg_predictions[["touch_focal", "touch_side", "touch", "key"]].groupby("key").agg(np.sum).reset_index()

                loader.interaction=loader.interaction.merge(
                    deg_predictions_summ,
                    on="key",
                    how="left"
                )

            if loader.interaction["touch"].isna().any():
                logger.warning(
                    "Fly %s: Not all interactions are analyzed: %s %% are missing",
                    loader, 100*loader.interaction["touch"].isna().mean()
                )

            loader.interaction=loader.interaction.merge(
                loader.interaction.groupby("interaction").agg({"touch_focal": lambda x: x.sum()==0}).reset_index().rename({"touch_focal": "ignore"}, axis=1),
                on="interaction"
            )
        
            distances=[]
            loader.interaction["discard"]=(loader.interaction["frame_number_next"]-loader.interaction["last_frame_number"]<150*60)
            for i, row in loader.interaction.iterrows():
                # loader.behavior[(row["first_frame"]>=loader.behavior["frame_number"])&(row["last_frame_number"]<loader.behavior["frame_number"])]
                distance=loader.behavior[(loader.behavior["frame_number"]>=row["last_frame_number"])&(loader.behavior["frame_number"]<(row["last_frame_number"]+150*60))]["centroid_speed"].sum()
                distances.append(distance)
            loader.interaction["distance_traveled_after_interaction"]=distances


    if database is None:
        return loaders, None

    interactions=[]
    for loader in loaders:
        interactions.append(loader.interaction)
    interactions=pd.concat(interactions, axis=0).reset_index()
    bin_size=1
    interactions["t_round"]=bin_size*interactions["t"]//bin_size
    
    all_flies=set(interactions["id"].unique().tolist())

    interactions["other"]=[find_other_fly(all_flies.copy(), set([row["id"], row["nn"]])) for _, row in interactions.iterrows()]
    interactions["experiment"]=experiment
    return loaders, interactions



