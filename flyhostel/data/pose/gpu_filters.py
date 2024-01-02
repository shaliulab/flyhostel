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
    from cupy.lib.stride_tricks import as_strided

except:
    cudf=None
    cp=None
    as_strided=None
    logger.warning("Cannot load cudf or cupy")

from .filters import (
    impute_proboscis_to_head,
    interpolate_pose,
)


# upload to GPU batches of 1 hour long data
POSE_FRAMERATE=150
PARTITION_SIZE=POSE_FRAMERATE*3600 

def filter_pose_df(data, *args, columns=None, download=False, **kwargs):
    original_columns=data.columns
    if isinstance(data, pd.DataFrame):
        assert columns is not None
        cudf_df=cudf.DataFrame(cp.array(data[columns].values), columns=columns)
        other_columns=data.drop(columns, axis=1)
    else:
        cudf_df=data

    if columns is None:
        cudf_df[:]=cp.from_dlpack(cudf_df.interpolate(method="linear", axis=0, limit_direction="both").fillna(method="bfill", axis=0).to_dlpack())
        
    else:
        cudf_df[columns]=cp.from_dlpack(cudf_df[columns].interpolate(method="linear", axis=0, limit_direction="both").fillna(method="bfill", axis=0).to_dlpack())
    
    if download:
        data=cudf_df.to_pandas()
        data=pd.concat([data, other_columns], axis=1)[original_columns]

    else:
        data=cudf_df
    return data
        

def filter_pose_partitioned(data, f, window_size, partition_size, pad=False):

    def process_partition(partition):
        shape = (partition.shape[0] - window_size + 1, partition.shape[1], window_size)
        strides = (partition.strides[0], partition.strides[1], partition.strides[0])
        strided_partition = as_strided(partition, shape=shape, strides=strides)
        return f(strided_partition, axis=-1)

    n_rows = data.shape[0]
    results = []

    for start in tqdm(range(0, n_rows, partition_size)):
        end = start + partition_size + window_size - 1
        end = min(end, n_rows)  # Ensure we don't go beyond the array
        partition = data[start:end]
        results.append(process_partition(partition))
    concatenated = cp.concatenate(results, axis=0)

    # Pad the end of the array with the last value if pad_end is True
    if pad:
        padding_rows = n_rows - concatenated.shape[0]
        if padding_rows > 0:
            number_of_rows_padded_start=(padding_rows//2)            
            first_values = data[:number_of_rows_padded_start, :]
            padding=data[-(padding_rows-number_of_rows_padded_start):]
            concatenated = cp.concatenate([first_values, concatenated, padding], axis=0)

    return concatenated


def split_xy(arr):
    """
    Given an array of shape Nxm
    where the columns are arranged as feat1_x feat1_y, feat2_x, feat2_y,
    return a new array of shape Nxm/2x2 so that feat1_x and feat1_y are stored on arr[:,1]
    """
    return cp.stack([
        arr[:,range(0, arr.shape[1], 2)],
        arr[:,range(1, arr.shape[1], 2)],
    ], axis=2)


def filter_pose_far_from_median_gpu(pose, bodyparts, px_per_cm=175, min_score=0.5, window_size_seconds=0.5, max_jump_mm=1, framerate=150):
    
    window_size=int(window_size_seconds*framerate)
    min_periods=3
    arr=cp.from_dlpack(pose.interpolate(method="linear", axis=0, limit_direction="both").fillna(method="bfill", axis=0).to_dlpack())
    median_arr=filter_pose_partitioned(arr, cp.median, window_size, PARTITION_SIZE, pad=True)

    # we make the split after computing the median
    # because only 2D arrays are supported (time x features is supported, but not time x features x XY_dimensions)
    median_arr=split_xy(median_arr)
    arr=split_xy(arr)
    jump_px=cp.sqrt(cp.sum((median_arr-arr)**2, axis=2))
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
        pose.loc[jump_matrix[:,i], bp + "_x"]=np.nan
        pose.loc[jump_matrix[:,i], bp + "_y"]=np.nan
    
    del jump_matrix
    return pose


def filter_and_interpolate_pose_single_animal_gpu_(pose, bodyparts, filters, min_score=0.5, window_size_seconds=0.5, max_jump_mm=1, interpolate_seconds=0.5, download=True):

    bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
    pose=pose.sort_values("t")


    before=time.time()
    # NOTE for some reason
    # pose_cudf=cudf.from_pandas(pose[bodyparts_xy])
    # does not work well, because
    # pose_cudf.values
    # throws an error 
    pose_cudf=cudf.DataFrame(cp.array(pose[bodyparts_xy].values), columns=bodyparts_xy)
    other_columns=pose.drop(bodyparts_xy, axis=1)

    after=time.time()
    logger.debug("Upload data to GPU in %s seconds", round(after-before, 1))
    logger.debug("Filtering jumps deviating from median")
    missing_data_mask=pose[bodyparts_xy].isna()
    pose_cudf=filter_pose_far_from_median_gpu(
        pose_cudf, bodyparts, min_score=min_score,
        window_size_seconds=window_size_seconds,
        max_jump_mm=max_jump_mm,
    )


    logger.debug("Interpolating pose")
    pose_cudf=interpolate_pose(pose_cudf, bodyparts, seconds=interpolate_seconds, pose_framerate=POSE_FRAMERATE)
    # NOTE be aware this interpolation is not necessarily complete
    # only up to a given amount of seconds are interpolated!
    logger.debug("Imputing proboscis to head")
    pose_cudf=impute_proboscis_to_head(
        pose=pose_cudf,
        selection=pose["proboscis_x"].isna()
    )

    before=time.time()   
    pose_cudf=filter_pose_df(pose_cudf, columns=bodyparts_xy, f=cp.median, window_size=int(0.2*POSE_FRAMERATE), partition_size=PARTITION_SIZE, pad=True)
    after=time.time()
    logger.debug("Apply median filter on pose in %s seconds (GPU)", round(after-before, 1))

    before=time.time()
    pose_cudf=filter_pose_df(pose_cudf, columns=bodyparts_xy, f=cp.mean, window_size=int(0.2*POSE_FRAMERATE), partition_size=PARTITION_SIZE, pad=True)
    after=time.time()
    logger.debug("Apply mean filter on pose in %s seconds (GPU)", round(after-before, 1))

    # reset missing data but still impute proboscis
    for bp in bodyparts:
        for feat in ["x", "y"]:
            bp_feat=bp+"_"+feat
            pose_cudf.loc[missing_data_mask[bp_feat], bp_feat]=np.nan
        # missing_data_mask[bp_x] is the same as missing_data_mask[bp_y]
        other_columns.loc[missing_data_mask[bp_feat], bp + "_is_interpolated"]=True
    logger.debug("Imputing proboscis to head")
    pose_cudf=impute_proboscis_to_head(pose_cudf, selection=other_columns["proboscis_is_interpolated"])
    assert np.bitwise_and(
        pose_cudf["proboscis_x"].isna(),
        ~pose_cudf["head_x"].isna()
    ).sum() == 0
    


    out=pose_cudf
    if download:
        before=time.time()
        out=pose_cudf.to_pandas()
        del pose_cudf
        after=time.time()
        out=pd.concat([out, other_columns], axis=1)[pose.columns]

        logger.debug("Download pose from GPU in %s seconds", round(after-before, 1))
    else:
        logger.warning("Only columns in bodyparts_xy are available in output. Please set download=True if you need other columns present in the input")

    return {"jumps": None, "filters": {"nanmean": out}}
