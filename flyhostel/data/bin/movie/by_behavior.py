"""
Generate behavior-annotated flyhostel single fly videos
"""

import argparse
import logging
import os.path
logging.getLogger("flyhostel.data.pose.movie.opencv").setLevel(logging.WARNING)
logger=logging.getLogger(__name__)

import pandas as pd
from flyhostel.data.pose.video_crosser import cross_with_video_data
from flyhostel.data.pose.movie.movie import annotate_behavior_in_video
from flyhostel.data.pose.ethogram.utils import most_common_behavior_vectorized
from flyhostel.data.pose.ethogram.utils import annotate_bout_duration, annotate_bouts
from flyhostel.utils import (
    get_framerate,
    get_chunksize,
    get_wavelet_downsample
)


def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--identity", type=int, required=True)
    ap.add_argument("--behavior", type=str, required=True)
    ap.add_argument("--n-videos", type=int, default=None)
    return ap


# DO NOT REMOVE THESE COMMENTS!!
# Run in environment with flyhostel fully installed
# from flyhostel.data.pose.main import FlyHostelLoader
# metadata=metadata.head(4).tail(1)

# loader=FlyHostelLoader.from_metadata(
#     '3', 6, '2023-08-24', '13-00-00', 2
# )
# dt=loader.load_analysis_data()
# dt.to_feather(loader.datasetnames[0]+".feather")

def make_illustrations(dt, behavior, animal, framerate, min_duration=None, n_seconds_before=1, n_seconds_after=1, n_videos=None):
    # predict every 1 second
    time_window_length=1
    logger.debug("Setting time resolution to %s second(s)", time_window_length)
    dt_no_noise=most_common_behavior_vectorized(dt.copy(), time_window_length, other_cols=["score", "chunk", "frame_idx"])
    dt_postprocessed=dt.drop("behavior", axis=1).merge(dt_no_noise[["id", "frame_number", "behavior"]], on=["frame_number","id"], how="left")


    experiment=animal.split("__")[0]
    chunksize=get_chunksize(experiment)
    wavelet_downsample=get_wavelet_downsample(experiment)
    dt_postprocessed["behavior"]=dt_postprocessed["behavior"].ffill(limit=time_window_length * (framerate//wavelet_downsample))
    
    # annotate bout duration and id
    dt_postprocessed=annotate_bouts(dt_postprocessed, "behavior")
    dt_postprocessed=annotate_bout_duration(dt_postprocessed, fps=framerate//wavelet_downsample)

    for _, df in dt_postprocessed.groupby("chunk"):
        
        index_input=df[["frame_number", "identity", "id"]]
        index_input["animal"]=animal
        index=cross_with_video_data(index_input)
        video_input=index["video"].unique()
        assert len(video_input)==1
        video_input=video_input[0]
       
        bouts=df.loc[(df["bout_in"]==1) & (df["behavior"]==behavior)]
        if min_duration is not None:
            bouts=bouts.loc[bouts["duration"]>=min_duration]

        if n_videos is not None and bouts.shape[0]>0:
            bouts=bouts.sample(n=min(bouts.shape[0], n_videos)).sort_values("frame_number")
       
        for i, frame_number in enumerate(bouts["frame_number"]):
            print(frame_number)
            # frame_numbers=np.arange(frame_number-framerate*n_seconds_before, frame_number+framerate*n_seconds_after+bouts["duration"].iloc[i]*framerate, wavelet_downsample)
            interval=(frame_number-framerate*n_seconds_before, frame_number+framerate*n_seconds_after+bouts["duration"].iloc[i]*framerate)
            frame_numbers=df.loc[(df["frame_number"]>=interval[0]) & (df["frame_number"]<interval[1]), "frame_number"]
            
            bout=pd.DataFrame({
                "frame_number": frame_numbers,
                "frame_idx": frame_numbers%chunksize,
                "behavior": df["behavior"].loc[df["frame_number"].isin(frame_numbers)].values
            })
            
            
            video_output=os.path.join("illustrations", animal + f"__{behavior}__{frame_number}.mp4")
            annotate_behavior_in_video(video_input, bout["frame_idx"].values, bout["behavior"].values, video_output, gui_progress=False, fps=framerate//wavelet_downsample)


def annotate_by_behavior():

    ap=get_parser()
    args=ap.parse_args()
        
    # Load dataset
    # TODO Change this so we dont assume the dataset is present in the working directory
    animal=f"{args.experiment}__{str(args.identity).zfill(2)}"
    dt=pd.read_feather(f"{animal}.feather")

    chunksize=get_chunksize(args.experiment)
    framerate=get_framerate(args.experiment)


    dt["frame_idx"]=dt["frame_number"]%chunksize
    make_illustrations(dt.copy(), args.behavior, animal, framerate=framerate, n_videos=args.n_videos)