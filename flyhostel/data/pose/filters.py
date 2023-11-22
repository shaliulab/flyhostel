from functools import partial
import itertools
import numpy as np
from flyhostel.data.interactions.bodyparts import bodyparts as BODYPARTS


def filter_pose(filter_f, pose, bodyparts, window_size=0.5, min_window_size=40):
    """
    Arguments:
        pose (pd.DataFrame): contains columns t (seconds), frame_number and bp_x, bp_y
        bodyparts (list): Must contain all and only all the bp in pose
        window_size (float): Size of the time window used to compute the filter, in seconds
            The window is centered around each point
        min_window_size (int): This number of points around each point is preselected around each point
           to check whether these points are within window_size of the centered point.
           If your framerate is huge, you should increase it
           
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

    bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
    pose_values=pose[bodyparts_xy].values.reshape((-1, len(bodyparts), 2))


    max_end=pose.shape[0]
    for i, t in enumerate(pose["t"]):
        start=max(0, i - min_window_size//2)

        end=min(i+min_window_size//2, max_end)
        context = pose["t"].values[start:end]
        this_values=pose_values[start:end, ...]
        if start == 0:
            padding_size=min_window_size-context.shape[0]
            padding=[np.nan for _ in range(padding_size)]

            this_values = np.concatenate([
                np.array([np.nan for _ in range(padding_size*2*len(bodyparts))]).reshape((padding_size, len(bodyparts), 2)),
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
                np.array([np.nan for _ in range(padding_size*2*len(bodyparts))]).reshape((padding_size, len(bodyparts), 2)),
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

    values_arr=np.stack(values_arr, axis=3)
    inputs=np.stack(inputs, axis=2)

    in_window=np.abs(np.diff(inputs, axis=0))[0, ...] < window_size/2

    assert in_window.shape[0] == values_arr.shape[0]
    assert in_window.shape[1] == values_arr.shape[3]

    window_pos, window_id=np.where(~in_window)
    # for each window, set to nan the values outside of the window_size limit (0.5 seconds)
    values_arr[window_pos, :, :, window_id]=np.nan
    filtered_pose=getattr(np, filter_f)(values_arr, axis=0)
    pose_values=np.moveaxis(pose_values, 0, -1)
    return filtered_pose, pose_values


filter_pose_median=partial(filter_pose, filter_f="nanmedian")