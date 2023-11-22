import itertools
import queue
import os.path
from typing import Union, Dict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from sleap.io.dataset import Labels
from sleap.io.video import Video
from sleap.instance import LabeledFrame, Instance
from sleap.io.visuals import resize_images, VideoMarkerThread
from sleap.io.visuals import save_labeled_video

from .filters import filter_pose_median

cwd=os.getcwd()
ref_labels_file="/Users/FlySleepLab_Dropbox/Data/flyhostel_data/fiftyone/FlyBehaviors/FlyBehaviors_6cm.v003.slp"
os.chdir(os.path.dirname(ref_labels_file))
skeleton=Labels.load_file(ref_labels_file).skeleton
os.chdir(cwd)

def numpy(instance, bodyparts):
    
    """
    Arguments:

        instance (pd.Series): Coordinates of the nodes (bodyparts) of an instance
            in format node_x node_y (as columns of a pd.Series)
        bodyparts (list):
    Returns:
        np.array of shape (n_nodes, 2) of dtype float32
        containing the coordinates of the instance’s nodes.
        Missing/not visible nodes will be replaced with NaN.
    """    
    data=[]
    for bp in bodyparts:
        data.append(instance[[bp + "_x", bp + "_y"]].values.flatten())
    data=np.stack(data, axis=0)
    return data

def make_labeled_frames(pose, identity, frame_numbers, chunksize, video):
    labeled_frames=[]

    for frame_number in frame_numbers:
        frame_idx=frame_number%chunksize

        instance_series=pose.loc[(pose["identity"]==identity) & (pose["frame_number"]==frame_number)]
        base_instance_numpy=numpy(instance_series, bodyparts=[node.name for node in skeleton.nodes])


        instance=Instance.from_numpy( 
            base_instance_numpy, skeleton=skeleton
        )
        lf = LabeledFrame(video=video, frame_idx=frame_idx, instances=[instance])
        labeled_frames.append(lf)
    return labeled_frames


def draw_frame(pose, index, identity, frame_number, chunksize=45000):
    frame_idx=frame_number % chunksize

    video=Video.from_filename(index.loc[index["frame_number"]==frame_number]["video"].item())
    labeled_frames=make_labeled_frames(
        pose, identity,
        frame_numbers=[frame_number], chunksize=chunksize, video=video
    )
    labels=Labels(labeled_frames=labeled_frames, videos=[video], skeletons=[skeleton])

    q1=queue.Queue()
    q2=queue.Queue()
    vmt=VideoMarkerThread(
        in_q=q1, out_q=q2, labels=labels, video_idx=0, scale=5,
        show_edges=True, edge_is_wedge=False, marker_size=1,
        crop_size_xy=False, color_manager=None, palette="standard", distinctly_color="instances"
    )

    loaded_chunk_idxs, video_frame_images=labels.videos[0].get_frames_safely([frame_idx])
    assert video_frame_images is not None
    assert len(loaded_chunk_idxs) > 0
    video_frame_images = resize_images(video_frame_images, 5)
    imgs=vmt._mark_images(loaded_chunk_idxs, video_frame_images)
    return imgs[0]


def draw_video(pose, index, identity, frame_numbers, chunksize=45000, fps=15, output_filename=None):
    
    chunks=[frame_number // chunksize for frame_number in frame_numbers]
    assert len(set(chunks)) == 1, f"Please pass frames from within the same chunk"

    video=Video.from_filename(index.loc[index["frame_number"]==frame_numbers[0]]["video"].item())

    labeled_frames=make_labeled_frames(
        pose, identity,
        frame_numbers=frame_numbers, chunksize=chunksize, video=video
    )
    labels=Labels(labeled_frames=labeled_frames, videos=[video], skeletons=[skeleton])

    fn, extension = os.path.splitext(os.path.basename(video.backend.filename))   

    if output_filename is None:
        output_filename=os.path.join(os.path.dirname(video.backend.filename), fn + "_render" + extension)

    save_labeled_video(
        output_filename,
        labels=labels,
        video=labels.video,
        frames=frame_numbers%chunksize,
        fps=fps,
        scale=5.0,
        crop_size_xy= None,
        show_edges= True,
        edge_is_wedge=False,
        marker_size=1,
        color_manager=None,
        palette="standard",
        distinctly_color="instances",
        gui_progress=False
    )


def filter_pose_far_from_median(pose, bodyparts, px_per_cm=175, min_score=0.5, window_size_seconds=0.5, max_jump_mm=1):
    for bp in bodyparts:
        bp_cols=[bp + "_x", bp + "_y"]
        bp_cols_ids=[pose.columns.tolist().index(c) for c in bp_cols]
        pose.iloc[np.where(pose[bp + "_likelihood"] < min_score)[0], bp_cols_ids] = np.nan

    median_pose, values = filter_pose_median(pose=pose, bodyparts=bodyparts, window_size=window_size_seconds)
    jump_px=np.sqrt(np.sum((median_pose-values)**2, axis=1))
    jump_mm=(10*jump_px/px_per_cm)

    rows, cols = np.where((jump_mm>max_jump_mm).T)

    for i, (row, col) in enumerate(zip(rows, cols)):
        bp=bodyparts[col]
        columns=[bp + "_x", bp + "_y"]
        col_ids=[pose.columns.tolist().index(c) for c in columns]
        pose.iloc[row, col_ids]=np.nan
        pose.loc[
            np.isnan(pose[columns]).any(axis=1),
            bp + "_is_interpolated"
        ]=True
        
    return pose

def arr2df(pose, arr, bodyparts):
    data={}
    for i, bp in enumerate(bodyparts):
        data[bp + "_x"]=arr[i, 0, :]
        data[bp + "_y"]=arr[i, 1, :]
    
    data=pd.DataFrame(data)
    new_pose=pose.drop(data.columns.tolist(), axis=1)
    new_pose=pd.concat([new_pose, data], axis=1)

    return new_pose



def interpolate_pose(pose, bodyparts, seconds: Union[Dict, float, int]=0.5, pose_framerate=15):
    bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))

    for c in bodyparts_xy:
        bp = c.replace("_x", "").replace("_y", "")
        if isinstance(seconds, float) or isinstance(seconds, int):
            seconds_bp=seconds
        else:
            seconds_bp=seconds[bp]
        interpolation_limit=int(seconds_bp*pose_framerate)
        pose[c].interpolate(method="linear", limit_direction="both", inplace=True, limit=interpolation_limit)
    return pose
