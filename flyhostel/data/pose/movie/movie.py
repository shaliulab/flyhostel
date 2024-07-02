import logging
import os.path

import numpy as np

from flyhostel.data.pose.video_crosser import cross_with_video_data
from flyhostel.data.pose.constants import bodyparts_xy, chunksize, framerate
from flyhostel.data.pose.sleap import draw_video
from .opencv import annotate_behavior_in_video_cv2
from .ffmpeg import annotate_behavior_in_video_ffmpeg
stride=1


logger=logging.getLogger(__name__)
logging.getLogger("flyhostel.data.pose.sleap").setLevel(logging.INFO)


def annotate_behavior_in_video(*args, **kwargs):
    return annotate_behavior_in_video_cv2(*args, **kwargs)
    
    
def make_video(pose, id, filename, frame_numbers, output_folder="."):
    pose=pose.loc[pose["id"]==id]
    pose["animal"]=pose["id"]
    identity=int(id.split("|")[1])
    logger.info("id =  %s, identity = %s", id, identity)

    
    # make index: annotate in which .mp4 video file 
    # can the source video be found for each fly in each frame
    index=cross_with_video_data(pose[["frame_number", "identity", "id", "animal"]])
    draw_video(
        pose, index,
        identity=identity,
        frame_numbers=frame_numbers,
        chunksize=chunksize,
        fps=framerate//stride,
        output_filename=os.path.join(output_folder, filename),
        gui_progress=False,
    )


def annotate_chunk(experiment, behavior_df, chunk, identity, input_video, pose=None, output_video="./output.mp4", with_pose=False, frame_numbers=None, **kwargs):

    output_folder=os.path.dirname(output_video)
   
    if frame_numbers is None:
        frame_numbers=np.arange(chunk*chunksize, (chunk+1)*chunksize, stride)
    
    key=f"{experiment}__{str(identity).zfill(2)}_{str(chunk).zfill(6)}"
    
    if output_video is None:
        output_video=os.path.join(output_folder, f"{key}_scored.mp4")

    dt_scored=behavior_df.loc[behavior_df["frame_number"].isin(frame_numbers), ["id", "identity", "frame_number", "frame_idx", "behavior"]]
    
    dt_scored=dt_scored.loc[dt_scored["identity"]==int(identity)].drop("identity", axis=1)
    assert dt_scored.shape[0] > 0, f"No data found for {experiment}__{str(identity).zfill(2)}"

    
    if with_pose:
        pose=pose.loc[pose["identity"]==identity].drop("identity", axis=1)
        filename, ext = os.path.splitext(os.path.basename(output_video))
        filename = filename +"_with_pose" + ext
        id = dt_scored["id"].iloc[0]
        make_video(pose, id, filename, frame_numbers=dt_scored["frame_number"].values, output_folder=output_folder)
        input_video=os.path.join(output_folder, filename)

    annotate_behavior_in_video(
        input_video,
        dt_scored["frame_idx"].values,
        dt_scored["behavior"].values,
        output_video,
        **kwargs
    )
    return output_video

