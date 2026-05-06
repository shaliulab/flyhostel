import logging
import time
import itertools
from tqdm import tqdm
import joblib
import numpy as np
import pandas as pd
logger=logging.getLogger(__name__)
from flyhostel.utils.filters import one_pass_filter_all
from flyhostel.data.pose.constants import DEFAULT_FILTERS
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

from flyhostel.data.pose.constants import MAX_JUMP_MM, JUMP_WINDOW_SIZE_SECONDS, min_score


def filter_pose_df(data, *args, columns=None, download=False, nx=cp, n_jobs=1, **kwargs):
    original_columns=data.columns
    
    if isinstance(data, pd.DataFrame) and nx is cp:
        assert columns is not None
        cudf_df=cudf.DataFrame(cp.array(data[columns].values), columns=columns)
    else:
        cudf_df=data

    other_columns=data.drop(columns, axis=1).reset_index(drop=True)
    if nx is np:
        if isinstance(cudf_df, cudf.DataFrame):
            cudf_df=cudf_df.to_pandas()
    
        arr=cudf_df.values

        data[data.columns]=filter_pose_partitioned(arr, *args, **kwargs, nx=nx, n_jobs=n_jobs)

    elif nx is cp:
        arr=cp.from_dlpack(cudf_df.interpolate(method="linear", axis=0, limit_direction="both").fillna(method="bfill", axis=0).to_dlpack())
        cudf_df[cudf_df.columns]=filter_pose_partitioned(arr, *args, **kwargs, nx=nx, n_jobs=n_jobs)
            

        if download:
            data=cudf_df.to_pandas()
            data=pd.concat([data, other_columns], axis=1)[original_columns]

        else:
            data=cudf_df
    
    return data
        

def filter_pose_partitioned(data, f, window_size, partition_size, pad=False, nx=np, n_jobs=-2):

    def process_partition(i, partition):
        with open("partition_index.txt", "w") as handle:
            handle.write(f"Working on partition {i}\n")

        shape = (partition.shape[0] - window_size + 1, partition.shape[1], window_size)
        strides = (partition.strides[0], partition.strides[1], partition.strides[0])
        strided_partition=nx.lib.stride_tricks.as_strided(partition, shape=shape, strides=strides)
        return f(strided_partition, axis=-1)

    n_rows = data.shape[0]
    results = []

    if nx is cp:
        for i, start in tqdm(enumerate(range(0, n_rows, partition_size)), desc="Filtering pose"):
            end = start + partition_size + window_size - 1
            end = min(end, n_rows)  # Ensure we don't go beyond the array
            partition = data[start:end]
            results.append(process_partition(i, partition))
    elif nx is np:
        results=joblib.Parallel(
            n_jobs=n_jobs
        )(
            joblib.delayed(process_partition)(
                i,
                data[start:min(n_rows, start+partition_size+window_size-1)]
            )
            for i, start in enumerate(range(0, n_rows, partition_size))
        )
    
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


def filter_pose_far_from_median_gpu(pose, bodyparts, framerate, partition_size, pixels_per_mm, window_size_seconds=JUMP_WINDOW_SIZE_SECONDS, max_jump_mm=MAX_JUMP_MM, n_jobs=1):
    """
    Ignore points that deviate from the median

    Points deviating from the median are those farther than `max_jump_mm` mm of the median computed on a window around them, of `window_size_seconds` seconds
    framerate tells the program how many points make one second and px_per_mm how many pixels are 1 mm
    """

    window_size=int(window_size_seconds*framerate)


    if isinstance(pose, cudf.DataFrame):
        arr=pose.to_pandas().values
    else:
        arr=pose.values

    nx=np
    # arr=cp.from_dlpack(pose.interpolate(method="linear", axis=0, limit_direction="both").fillna(method="bfill", axis=0).to_dlpack())

    median_arr=filter_pose_partitioned(arr, nx.median, window_size, partition_size, pad=True, nx=nx, n_jobs=n_jobs)

    # we make the split after computing the median
    # because only 2D arrays are supported (time x features is supported, but not time x features x XY_dimensions)
    median_arr=split_xy(median_arr, nx=nx)
    arr=split_xy(arr, nx=nx)
    jump_px=nx.sqrt(nx.sum((median_arr-arr)**2, axis=2))
    del arr
    del median_arr
    jump_mm=(jump_px/pixels_per_mm)
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


def filter_and_interpolate_pose_single_animal_gpu_(
        pose, bodyparts, framerate, partition_size, filters, pixels_per_mm, window_size_seconds=0.5, max_jump_mm=1, interpolate_seconds=0.5,
        download=True, n_jobs=1, filters_order=None,
    ):

    bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
    pose=pose.sort_values("t")
    if filters_order is None:
        filters_order=DEFAULT_FILTERS
 
    filters=[]
    nx=np

    missing_data_mask=pose[bodyparts_xy].isna()
    pose_cudf=pd.DataFrame(np.array(pose[bodyparts_xy].values), columns=bodyparts_xy)
    other_columns=pose.drop(bodyparts_xy, axis=1).reset_index(drop=True)


    # RLE filter
    # Overwrites bouts of one value surrounded by bouts of the same other value so that a single value is present
    # https://github.com/talmolab/sleap/discussions/1739
    if "rle" in filters_order:
        logger.debug("Applying rle filter")
        pose_cudf_arr=one_pass_filter_all(pose_cudf.values, n_jobs=n_jobs)
        before=time.time()
        pose_cudf=pd.DataFrame(pose_cudf_arr, columns=pose_cudf.columns)
        after=time.time()
        logger.debug("Upload data to GPU in %s seconds", round(after-before, 1))
        filters.append("rle")
   
    # Jump from median filter
    # Sets data to nan if it deviates more than a given distance from the median of a local temporal window
    # because it is likely to be a spurious detection (an impossible jump of a body part)
    if "jump" in filters_order:
        logger.debug("Filtering jumps deviating from median")
        pose_cudf=filter_pose_far_from_median_gpu(
            pose_cudf, bodyparts,
            framerate=framerate,
            partition_size=partition_size,
            pixels_per_mm=pixels_per_mm,
            window_size_seconds=window_size_seconds,
            max_jump_mm=max_jump_mm,
            n_jobs=n_jobs,
        )
        filters.append(f"jump(max_jump_mm={max_jump_mm})")
    

    # Apply linear filters: mean and median, if contained in filters_order
    pose_cudf, filters=apply_linear_filters(pose_cudf, framerate, partition_size, filters, interpolate_seconds, filters_order=filters_order, bodyparts_xy=bodyparts_xy, n_jobs=n_jobs)
    # Set the data that was originally missing to missing
    # The previous interpolation were only done because the filters needed to have no missing data
    # however, we want to reflect the missing data in the output
    pose_cudf, other_columns=reset_interpolation(pose_cudf, other_columns, missing_data_mask, bodyparts, nx)
    logger.debug("Imputing proboscis to head")
    selection=np.bitwise_and(~pose_cudf["head_x"].isna(), pose_cudf["proboscis_x"].isna())
    pose_cudf=impute_proboscis_to_head(pose_cudf, selection=selection)

    if download:
        out = download_from_gpu(pose_cudf)
    else:
        out=pose_cudf
        logger.warning("Only columns in bodyparts_xy are available in output. Please set download=True if you need other columns present in the input")

    out=pd.concat([out, other_columns], axis=1)[pose.columns]
    return out, filters


def download_from_gpu(pose_cudf):
    before=time.time()
    try:
        out=pose_cudf.to_pandas()
        del pose_cudf
    except:
        out=pose_cudf.copy()            
    after=time.time()
    logger.debug("Download pose from GPU in %s seconds", round(after-before, 1))
    return out
    

def apply_linear_filters(pose_cudf, framerate, partition_size, filters, interpolate_seconds, filters_order, bodyparts_xy, n_jobs):
    logger.debug("Interpolating pose")
    pose_cudf=interpolate_pose(pose_cudf, framerate, bodyparts_xy, seconds=interpolate_seconds)
    # NOTE be aware this interpolation is not necessarily complete
    # only up to a given amount of seconds are interpolated!
    logger.debug("Imputing proboscis to head")
    pose_cudf=impute_proboscis_to_head(pose=pose_cudf, selection=np.bitwise_and(~pose_cudf["head_x"].isna(), pose_cudf["proboscis_x"].isna()))

    extra_filters=[filter_FUN for filter_FUN in filters_order if filter_FUN in ["mean", "median"]]
    nx=np
    for filter_FUN in extra_filters:
        window_size=int(0.2*framerate)
        before=time.time()
        logger.debug("Applying %s filter on pose (%s)", filter_FUN, nx)
        pose_cudf=filter_pose_df(
            pose_cudf, columns=pose_cudf.columns,
            f=getattr(nx, filter_FUN),
            window_size=window_size,
            partition_size=partition_size,
            pad=True, nx=nx, n_jobs=n_jobs
        )
        after=time.time()
        logger.debug("Apply %s filter on pose in %s seconds (%s)", filter_FUN, round(after-before, 1), nx)
        filters.append(f"{filter_FUN}(window_size={window_size})")
    
    return pose_cudf, filters


def reset_interpolation(pose_cudf, other_columns, missing_data_mask, bodyparts, nx):
    """
    pose_cudf: Pose dataset (bp_x, bp_y, ...)
    other_columns: Interpolation annotation (bp_is_interpolated, ...)
    missing_data_mask: Boolean array with same shape as pose_cudf and True in the i,j cell if the i,j cell of the pose dataset
    contained missing data originally
    """
    missing_data_mask.index=pose_cudf.index

    # reset missing data but still impute proboscis
    for bp in bodyparts:
        for feat in ["x", "y"]:
            bp_feat=bp+"_"+feat
            rows=missing_data_mask[bp_feat]
            if rows.sum()==0:
                continue
            # position=np.where(rows)[0]
            
            try:
                pose_cudf.loc[rows, bp_feat]=nx.nan
            except Exception:
                import ipdb; ipdb.set_trace()
                raise error
        
        if rows.sum()!=0:
            other_columns.loc[rows.values, bp + "_is_interpolated"]=True
    
    return pose_cudf, other_columns