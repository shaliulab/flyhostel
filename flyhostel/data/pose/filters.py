import logging
import time
from functools import partial
import itertools
from typing import Union, Dict
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from flyhostel.data.pose.constants import bodyparts as BODYPARTS
from flyhostel.data.pose.constants import bodyparts_xy as BODYPARTS_XY
from flyhostel.data.pose.constants import framerate as FRAMERATE

logger=logging.getLogger(__name__)

CHUNK_SECONDS=30*60
CHUNK_FRAMES=CHUNK_SECONDS*FRAMERATE


try:
    import cupy as cp
except:
    logger.warning("cupy not installed")

def filter_pose(
        filter_f, pose, bodyparts,
        window_size=0.5, min_window_size=100,
        min_supporting_points=3, features=["x", "y"],
        useGPU=-1
    ):
    """
    Arguments:
        pose (pd.DataFrame): contains columns t (seconds), frame_number and bp_x, bp_y
        bodyparts (list): Must contain all and only all the bp in pose
        window_size (float): Size of the time window used to compute the filter, in seconds
            The window is centered around each point
        min_window_size (int): This number of points around each point is preselected around each point
           to check whether these points are within window_size of the centered point.
           If your framerate is huge, you should increase it

        min_supporting_points (int): If the window has less than this amount of points,
            the available points are ignored and the body part is treated as not seen
           
    Returns:
        filtered_pose (np.array) bodyparts x 2 x n_points filtered estimates
        values (np.array) bodyparts x 2 x n_points raw estimates

    Details:
        the for loop will populate a dimension in a numpy array containing the input pose values for a window of time
        for example the nth value of that dimension will contain the input values for the nth window (the context of the nth point)
        Once this array is built, the data is actually filtered in a single call (outside of the for loop) using nice np functionality
    """
    inputs=[]
    paddings=[]
    values_arr=[]
    if bodyparts is None:
        bodyparts=BODYPARTS

    bodyparts_feats=list(itertools.chain(*[list(itertools.chain(*[[bp + "_" + coord] for coord in features])) for bp in bodyparts]))
    pose_values=pose[bodyparts_feats].values.reshape((-1, len(bodyparts), len(features)))

    max_end=pose.shape[0]
    for i, t in enumerate(tqdm(pose["t"], desc="Frames processed")):
        start=max(0, i - min_window_size//2)
        end=min(start+min_window_size, max_end)

        # end=min(i+min_window_size//2, max_end)
        context = pose["t"].values[start:end]
        this_values=pose_values[start:end, ...]
        # if this_values.shape[0]!=min_window_size:
        #     print(start, end, this_values.shape)
        
        if start == 0:
            padding_size=min_window_size-context.shape[0]
            padding=[np.nan for _ in range(padding_size)]

            this_values = np.concatenate([
                np.array([np.nan for _ in range(padding_size*len(features)*len(bodyparts))]).reshape((padding_size, len(bodyparts), len(features))),
                this_values,
            ], axis=0)

            context = np.concatenate([
                padding,
                context,
            ], axis=0)
        elif end==max_end:
            padding_size=min_window_size-context.shape[0]
            padding=[np.nan for _ in range(padding_size)]
            this_values = np.concatenate([
                this_values,
                np.array([np.nan for _ in range(padding_size*len(features)*len(bodyparts))]).reshape((padding_size, len(bodyparts), len(features))),
            ], axis=0)

            context = np.concatenate([
                context,
                padding,
            ], axis=0)
        else:
            padding_size=0

        values_arr.append(this_values)
        paddings.append(padding_size)
        inputs.append(
            np.array([
               [t for _ in range(min_window_size)],
               context
            ])
        )


    # the stack steps take a while when framerate=150 fps
    # window_size x bodyparts x 2 x frames
    # before=time.time()
    # values_arr=np.stack(values_arr, axis=3)
    # after=time.time()
    # logger.debug("Took %s seconds to stack values_arr", after-before)

    
    # 2 x window_size x frames
    # [0,...] contains the t0 of the window
    # [1,...] contains the t of each frame in the window
    before=time.time()
    inputs=np.stack(inputs, axis=2)
    after=time.time()
    logger.debug("Took %s seconds to stack inputs", after-before)
    

    in_window=np.abs(np.diff(inputs, axis=0))[0, ...] < window_size/2

    assert in_window.shape[0] == values_arr[0].shape[0]
    assert in_window.shape[1] == len(values_arr)
    # # for each window, set to nan the values outside of the window_size limit (0.5 seconds)
    # for i, frame in enumerate(tqdm(window_id)):
    #     values_arr[frame][window_pos[i], :, :]=np.nan

    # values_arr has shape min_window_size, _, _, n_windows
    if useGPU >= 0:

        try:
            block_starts=np.arange(0, len(values_arr), CHUNK_FRAMES)
            block_ends=block_starts+CHUNK_FRAMES
            filtered_pose_list=[]
            for i, block_start in enumerate(tqdm(block_starts, desc="GPU Processing")):
                block_end=block_ends[i]
                logger.debug("Uploading to GPU %d", i)
                values_arr_np=np.stack(values_arr[block_start:block_end], axis=3)

                window_pos, window_id=np.where(~in_window[:, block_start:block_end])
                values_arr_np[window_pos, :, :, window_id]=np.nan
                
                supporting_values=min_window_size-np.isnan(values_arr_np).sum(axis=0)
        
                if min_supporting_points > 1:
                    for x, y, z in zip(*np.where(supporting_values<min_supporting_points)):
                        values_arr_np[:, x, y, z]=np.nan


                values_arr_cp=cp.array(values_arr_np)
                logger.debug("Applying %s filter to data of length %s using %s  %d", filter_f, len(values_arr), "cupy", i)
                filtered_pose_cp=getattr(cp, filter_f)(values_arr_cp, axis=0)
                logger.debug("Downloading from GPU  %d", i)
                filtered_pose_list.append(filtered_pose_cp.get())

            filtered_pose=np.concatenate(filtered_pose_list, axis=2)

        except Exception as error:
            raise error
            
    else:
        values_arr=np.stack(values_arr, axis=3)
        logger.debug("Applying %s filter to data of shape %s using %s", filter_f, values_arr.shape, "numpy")
        filtered_pose=getattr(np, filter_f)(values_arr, axis=0)
        logger.debug("Done")
    
    pose_values=np.moveaxis(pose_values, 0, -1)
    return filtered_pose, pose_values


filter_pose_median=partial(filter_pose, filter_f="nanmedian")


def filter_pose_far_from_median(pose, bodyparts, px_per_cm=175, window_size_seconds=0.5, max_jump_mm=1, useGPU=-1):


    median_pose, values = filter_pose_median(pose=pose, bodyparts=bodyparts, window_size=window_size_seconds, useGPU=useGPU)

    logger.debug("Computing distance from median")
    jump_px=np.sqrt(np.sum((median_pose-values)**2, axis=1))
    jump_mm=(10*jump_px/px_per_cm)

    logger.debug("Detecting jumps")
    jump_matrix=(jump_mm>max_jump_mm).T

    mask = np.zeros(pose.shape) == 1

    for bp_i, bp in enumerate(bodyparts):
        columns=[bp + "_x", bp + "_y"]
        col_ids=[pose.columns.tolist().index(c) for c in columns]
        mask[:, col_ids[0]]=jump_matrix[:, bp_i]
        mask[:, col_ids[1]]=jump_matrix[:, bp_i]

    pose[mask]=np.nan

    for bp in tqdm(bodyparts):
        pose.loc[
            np.isnan(pose[[bp + "_x", bp + "_y"]]).any(axis=1),
            bp + "_is_interpolated"
        ]=True        
    return pose



def arr2df(pose, arr, bodyparts, features=["x", "y"]):
    data={}
    for i, bp in enumerate(bodyparts):
        for j, feature in enumerate(features):
            data[bp + "_" + feature]=arr[i, j, :]
    
    data=pd.DataFrame(data, index=pose.index)
    new_pose=pose.drop(data.columns.tolist(), axis=1)
    new_pose=pd.concat([new_pose, data], axis=1)

    return new_pose


def interpolate_pose(pose, columns=None, seconds: Union[None, Dict, float, int]=0.5, pose_framerate=FRAMERATE, cache=None):
    if columns is None:
        columns=BODYPARTS_XY


    if isinstance(seconds, float) or isinstance(seconds, int):
        interpolation_limit=max(1, int(seconds*pose_framerate))
        pose[columns].interpolate(method="linear", limit_direction="both", inplace=True, limit=interpolation_limit)

    elif seconds is None:
        before=time.time()
        pose[columns]=pose[columns].interpolate(method="linear", limit_direction="both", limit=None)
        after=time.time()
        logger.debug("interpolate took %s seconds for colums %s with limit None", after-before, columns)

    elif isinstance(seconds, dict):
        values=sorted(list(set(list(seconds.values()))))
        reverse_dict={v: [k for k in seconds if seconds[k]==v] for v in values}

        for seconds in values:
            interpolation_limit=max(1, int(seconds*pose_framerate))
            bodyparts=reverse_dict[seconds]
            columns=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
            before=time.time()
            # pose[columns]=pose[columns].interpolate(method="linear", limit_direction="both", limit=interpolation_limit)
            pose[columns]=pose[columns].ffill(limit=interpolation_limit)
            pose[columns]=pose[columns].bfill(limit=interpolation_limit)
            after=time.time()
            logger.debug("interpolate took %s seconds for colums %s with limit %s", after-before, columns, interpolation_limit)

    return pose

def impute_proboscis_to_head(pose, selection=None):
    if selection is None:
        selection=np.bitwise_and(pose["proboscis_likelihood"].isna(), ~pose["head_likelihood"].isna())
    for coord in ["x", "y"]:
        pose.loc[selection, f"proboscis_{coord}"]=pose.loc[selection, f"head_{coord}"]
    return pose