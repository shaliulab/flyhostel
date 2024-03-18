import logging
import time
import itertools
from tqdm import tqdm

import numpy as np
import pandas as pd
logger=logging.getLogger(__name__)


try:
    import cudf
    import cupy as cp

except:
    cudf=None
    cp=None
    logger.warning("Cannot load cudf or cupy")

from .filters import (
    impute_proboscis_to_head,
    interpolate_pose,
)


# upload to GPU batches of 1 hour long data

from flyhostel.data.pose.constants import MAX_JUMP_MM, JUMP_WINDOW_SIZE_SECONDS, PARTITION_SIZE, min_score, PX_PER_CM, APPLY_MEDIAN_FILTER
from flyhostel.data.pose.constants import framerate as FRAMERATE


def filter_pose_df(data, *args, columns=None, download=False, nx=cp, **kwargs):
    original_columns=data.columns
    if isinstance(data, pd.DataFrame):
        assert columns is not None
        cudf_df=cudf.DataFrame(cp.array(data[columns].values), columns=columns)
        other_columns=data.drop(columns, axis=1).reset_index(drop=True)
    else:
        cudf_df=data

    # cudf_df.interpolate(method="linear", axis=0, limit_direction="both", inplace=True)
    # cudf_df=cudf_df.fillna(method="bfill", axis=0)
    if nx is np:
        data=cudf_df.to_pandas()
        arr=data.values

        data[data.columns]=filter_pose_partitioned(arr, *args, **kwargs, nx=nx)
    elif nx is cp:
        arr=cp.from_dlpack(cudf_df.interpolate(method="linear", axis=0, limit_direction="both").fillna(method="bfill", axis=0).to_dlpack())
        cudf_df[cudf_df.columns]=filter_pose_partitioned(arr, *args, **kwargs, nx=nx)
            

        if download:
            data=cudf_df.to_pandas()
            data=pd.concat([data, other_columns], axis=1)[original_columns]

        else:
            data=cudf_df
    
    return data
        

def filter_pose_partitioned(data, f, window_size, partition_size, pad=False, nx=np):

    def process_partition(partition):
        shape = (partition.shape[0] - window_size + 1, partition.shape[1], window_size)
        strides = (partition.strides[0], partition.strides[1], partition.strides[0])
        strided_partition=nx.lib.stride_tricks.as_strided(partition, shape=shape, strides=strides)
        return f(strided_partition, axis=-1)

    n_rows = data.shape[0]
    results = []

    for start in tqdm(range(0, n_rows, partition_size), desc="Filtering pose"):
        end = start + partition_size + window_size - 1
        end = min(end, n_rows)  # Ensure we don't go beyond the array
        partition = data[start:end]
        results.append(process_partition(partition))
    concatenated = nx.concatenate(results, axis=0)

    # Pad the end of the array with the last value if pad_end is True
    if pad:
        padding_rows = n_rows - concatenated.shape[0]
        if padding_rows > 0:
            number_of_rows_padded_start=(padding_rows//2)            
            first_values = data[:number_of_rows_padded_start, :]
            padding=data[-(padding_rows-number_of_rows_padded_start):]
            concatenated = nx.concatenate([first_values, concatenated, padding], axis=0)

    return concatenated


def split_xy(arr, nx=cp):
    """
    Given an array of shape Nxm
    where the columns are arranged as feat1_x feat1_y, feat2_x, feat2_y,
    return a new array of shape Nxm/2x2 so that feat1_x and feat1_y are stored on arr[:,1]
    """
    return nx.stack([
        arr[:,range(0, arr.shape[1], 2)],
        arr[:,range(1, arr.shape[1], 2)],
    ], axis=2)


def filter_pose_far_from_median_gpu(pose, bodyparts, px_per_cm=PX_PER_CM, window_size_seconds=JUMP_WINDOW_SIZE_SECONDS, max_jump_mm=MAX_JUMP_MM, framerate=FRAMERATE, nx=np):
    """
    Ignore points that deviate from the median

    Points deviating from the median are those farther than `max_jump_mm` mm of the median computed on a window around them, of `window_size_seconds` seconds
    framerate tells the program how many points make one second and px_per_cm how many pixels are 1 cm
    """
    window_size=int(window_size_seconds*framerate)


    arr=pose.to_pandas().values
    # arr=cp.from_dlpack(pose.interpolate(method="linear", axis=0, limit_direction="both").fillna(method="bfill", axis=0).to_dlpack())

    median_arr=filter_pose_partitioned(arr, nx.median, window_size, PARTITION_SIZE, pad=True, nx=nx)

    # we make the split after computing the median
    # because only 2D arrays are supported (time x features is supported, but not time x features x XY_dimensions)
    median_arr=split_xy(median_arr, nx=nx)
    arr=split_xy(arr, nx=nx)
    jump_px=nx.sqrt(nx.sum((median_arr-arr)**2, axis=2))
    del arr
    del median_arr
    jump_mm=(10*jump_px/px_per_cm)
    del jump_px
    jump_matrix=(jump_mm>max_jump_mm)
    del jump_mm

    for i, bp in enumerate(bodyparts):
        # even though working on GPU
        # (so cudf.NA or None may be more appropriate https://docs.rapids.ai/api/cudf/stable/user_guide/missing-data/#inserting-missing-data)
        # if we use either of those, pose.values breaks
        pose.loc[jump_matrix[:,i], bp + "_x"]=nx.nan
        pose.loc[jump_matrix[:,i], bp + "_y"]=nx.nan
    
    del jump_matrix
    return pose


def filter_and_interpolate_pose_single_animal_gpu_(pose, bodyparts, filters, window_size_seconds=0.5, max_jump_mm=1, interpolate_seconds=0.5, download=True, framerate=150):

    bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
    pose=pose.sort_values("t")

    # use cpu
    nx=np
    # use gpu
    # nx=cp

    before=time.time()
    # NOTE for some reason
    # pose_cudf=cudf.from_pandas(pose[bodyparts_xy])
    # does not work well, because
    # pose_cudf.values
    # throws an error
    pose_cudf=cudf.DataFrame(cp.array(pose[bodyparts_xy].values), columns=bodyparts_xy)
    other_columns=pose.drop(bodyparts_xy, axis=1).reset_index(drop=True)

    after=time.time()
    logger.debug("Upload data to GPU in %s seconds", round(after-before, 1))
    logger.debug("Filtering jumps deviating from median")
    missing_data_mask=pose[bodyparts_xy].isna()
    pose_cudf=filter_pose_far_from_median_gpu(
        pose_cudf, bodyparts,
        window_size_seconds=window_size_seconds,
        max_jump_mm=max_jump_mm,
        framerate=framerate,
        nx=nx
    )

    logger.debug("Interpolating pose")
    pose_cudf=interpolate_pose(pose_cudf, bodyparts_xy, seconds=interpolate_seconds, pose_framerate=FRAMERATE)
    # NOTE be aware this interpolation is not necessarily complete
    # only up to a given amount of seconds are interpolated!
    logger.debug("Imputing proboscis to head")
    pose_cudf=impute_proboscis_to_head(
        pose=pose_cudf,
        selection=np.bitwise_and(~pose["head_x"].isna(), pose["proboscis_x"].isna())
    )

    if APPLY_MEDIAN_FILTER:
        before=time.time()
        pose_cudf=filter_pose_df(pose_cudf, f=nx.median, window_size=int(0.2*FRAMERATE), partition_size=PARTITION_SIZE, pad=True, nx=nx)
        after=time.time()
        logger.debug("Apply median filter on pose in %s seconds (%s)", round(after-before, 1), nx)

    before=time.time()
    pose_cudf=filter_pose_df(pose_cudf, f=nx.mean, window_size=int(0.2*FRAMERATE), partition_size=PARTITION_SIZE, pad=True, nx=nx)
    after=time.time()
    logger.debug("Apply mean filter on pose in %s seconds (%s)", round(after-before, 1), nx)

    # reset missing data but still impute proboscis
    for bp in bodyparts:
        for feat in ["x", "y"]:
            bp_feat=bp+"_"+feat
            pose_cudf.loc[missing_data_mask[bp_feat], bp_feat]=nx.nan
        # missing_data_mask[bp_x] is the same as missing_data_mask[bp_y]
        other_columns.loc[missing_data_mask[bp_feat].values, bp + "_is_interpolated"]=True
    logger.debug("Imputing proboscis to head")

    selection=np.bitwise_and(~other_columns["head_is_interpolated"], other_columns["proboscis_is_interpolated"])
    pose_cudf=impute_proboscis_to_head(pose_cudf, selection=selection)
    assert np.bitwise_and(
        pose_cudf["proboscis_x"].isna(),
        ~pose_cudf["head_x"].isna()
    ).sum() == 0
    
    out=pose_cudf
    if download:
        before=time.time()
        try:
            out=pose_cudf.to_pandas()
            del pose_cudf
        except:
            out=pose_cudf.copy()            
        after=time.time()
        out=pd.concat([out, other_columns], axis=1)[pose.columns]

        logger.debug("Download pose from GPU in %s seconds", round(after-before, 1))
    else:

        logger.warning("Only columns in bodyparts_xy are available in output. Please set download=True if you need other columns present in the input")

    return {"jumps": None, "filters": {"nanmean": out}}
