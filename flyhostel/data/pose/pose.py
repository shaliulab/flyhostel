import pickle
import logging
import os.path
import itertools

from abc import ABC
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from flyhostel.data.pose.filters import (
    interpolate_pose,
    filter_pose_far_from_median,
    impute_proboscis_to_head,
    filter_pose,
    arr2df,
)
from flyhostel.data.pose.constants import framerate as POSE_FRAMERATE
from flyhostel.data.pose.constants import MIN_TIME, MAX_TIME
from flyhostel.data.pose.constants import bodyparts as BODYPARTS
from flyhostel.data.pose.constants import min_score as MIN_SCORE

from flyhostel.utils import restore_cache, save_cache
from .gpu_filters import filter_and_interpolate_pose_single_animal_gpu_



class FilterPose(ABC):
    """
    Refine pose estimate predictions using filters inspired in https://www.biorxiv.org/content/10.1101/2023.10.30.564733v1.full.pdf

    Pose estimation software that works on a frame-by-frame basis (predicitons are performed independently for each frame)
    are very efficient and parallelizable, but their output does not benefit from the added information gained by looking at the immediate
    context of a frame i.e. its neighboring frames

    This module allows us to refine the pose estimation by leveraging the context to more reliably detect spurious detections of movement
    and to smoothen the trajectory through pose space over time

    Usage:

    foo.filter_and_interpolate_pose(
        # list of bodyparts to process
        bodyparts,
        # dictionary of numpy functions used to smoothen the data, example:
        # {"nanmedian": {"window_size": 0.2, "min_window_size": 40}, "nanmean": {"window_size": 0.2, "min_window_size":10}}
        filters,
        # minimum pose estimate score to consider the prediction as valid
        min_score=0.5,
        # size of window to detect jumps
        window_size_seconds=0.5,
        # maximum allowed jump between frames
        max_jump_mm=1,
        # how much to interpolate the data in case of missing coordinates,
        # either a float or a bodypart indexed dictionary of floats
        interpolate_seconds=0.5
    )
    """


    def __init__(self, *args, **kwargs):

        self.pose=None
        self.pose_boxcar=None
        self.pose_interpolated=None
        self.experiment=None
        super(FilterPose, self).__init__(*args, **kwargs)


    def filter_and_interpolate_pose_single_animal(self, pose, min_time, max_time, stride, identifier, *args, cache=None, **kwargs):
        if cache is not None:
            cache_file=f"{cache}/{identifier}_{min_time}_{max_time}_{stride}_pose_filtered.pkl"
            if not os.path.exists(cache_file):
                cache_file=f"{cache}/{identifier}_-inf_+inf_1_pose_filtered.pkl"
                if os.path.exists(cache_file):
                    ret, pose=restore_cache(cache_file)
                    self.filter_pose_by_time(pose=pose, min_time=min_time, max_time=max_time)
                    if stride!=1:
                        pose=pose.iloc[::stride]
                    return pose

            else:
                ret, pose=restore_cache(cache_file)
                if ret:
                    return pose
                else:
                    logger.debug("Cannot find %s", cache_file)
                
        pose=self.filter_and_interpolate_pose_single_animal_all_filters(pose, *args, **kwargs)["filters"]["nanmean"]
        
        if cache is not None:
            save_cache(cache_file, pose)
        
        return pose


    def filter_and_interpolate_pose_single_animal_all_filters(self, pose, *args, bodyparts=BODYPARTS, filters=None, min_score=MIN_SCORE, useGPU=-1, cache=None, **kwargs):
        
        logger.debug("Removing low quality points")
        logger.debug(min_score)
        pose=self.ignore_low_q_points(pose, bodyparts, min_score=min_score)
            
        if useGPU >= 0:
            out=self.filter_and_interpolate_pose_single_animal_gpu(pose, bodyparts, filters, *args, **kwargs)
        else:
            out=self.filter_and_interpolate_pose_single_animal_cpu(pose, bodyparts, filters, *args, **kwargs)



        return out


    @staticmethod
    def filter_and_interpolate_pose_single_animal_gpu(*args, **kwargs):
        return filter_and_interpolate_pose_single_animal_gpu_(*args, **kwargs)


    @staticmethod
    def ignore_low_q_points(pose, bodyparts, min_score):
        """
        Points not passing the min_score criteria are ignored (set to nan in pose)
        pose (pd.DataFrame): Raw pose estimate containing, for every bodypart foo, columns foo_x, foo_y, foo_likelihood, foo_is_interpolated as well as columns id, frame_number, t
        bodyparts (list)
        min_score (float): Either a float, with the minimum score all body part pose predictions should have, or a dictionary of body part - float pairs
        """

        for bp in bodyparts:
            bp_cols=[bp + "_x", bp + "_y"]

            if isinstance(min_score, float):
                score=min_score
            elif isinstance(min_score, dict):
                score=min_score[bp]
            else:
                raise ValueError("min_score must be a float or a dictionary of bodypart - float pairs")
        
            bp_cols_ids=[pose.columns.tolist().index(c) for c in bp_cols]
            lk_cols_ids=[pose.columns.tolist().index(c) for c in [bp + "_likelihood"]]
            pose.iloc[pose[bp + "_likelihood"] < score, bp_cols_ids] = np.nan
            pose.iloc[pose[bp + "_likelihood"] < score, lk_cols_ids] = np.nan
    
        return pose
    

    @staticmethod
    def filter_and_interpolate_pose_single_animal_cpu(pose, bodyparts, filters, window_size_seconds=0.5, max_jump_mm=1, interpolate_seconds=0.5, framerate=POSE_FRAMERATE):
        """
        Process the raw pose data of a single animal

        Information in the neighboring frames can be leveraged to improve the estimate of each frame
        Example -> a body part movement which moves to point B in only one frame, and is in point A one frame before and after
        is unlikely to really have moved and instead is probably a spurious pose estimate error.
        This function processes the passed pose data with multiple steps and returns all these steps

        NOTE: This function assumes there are at least two bodyparts called head and proboscis.
        If the proboscis is missing, it is imputed to be where the head is

        Arguments:

            pose (pd.DataFrame): Raw pose estimate containing, for every bodypart foo, columns foo_x, foo_y, foo_likelihood, foo_is_interpolated as well as columns id, frame_number, t
            bodyparts (list)
            filters (dict): Dictionary of filters to apply. The keys must be numpy functions, and the values must be another dictionary with keys window_size, min_window_size, and order
            window_size_seconds (float): Window size used to compute the median to remove spurious "jumps"
            max_jump_mm (float): Number of mm away from the window median that a prediction must be for it to be considered a jump
            interpolate_seconds (float): Body parts missing are imputed forward and backwards (pd.Series.interpolate limit_direction both) up to this many seconds

        Returns: Dictionary with keys:
            jumps (pd.DataFrame): detections of a body part more than max_jump_mm mm away from the median of a rolling window of window_size_seconds seconds are ignored
            filters (dict): Contains one entry per filter in filters. Each of them is a dataframe based on the pose of the previous filter (given by its order attribute, lowest first).
               The filter consists of applying the numpy function to windows of up to min_window_size points but actually only the points within window_size seconds
               So each new processed point will be the result of applying the numpy function to points within window_size seconds of it
               The filter with order 0 is run on the pose_jumps dataset

        """
        raise NotImplementedError()
        useGPU=-1
        logger.debug("Filtering jumps deviating from median")
        pose=filter_pose_far_from_median(
            pose, bodyparts, window_size_seconds=window_size_seconds, max_jump_mm=max_jump_mm,
            useGPU=useGPU
        )
        logger.debug("Interpolating pose")

        bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
        
        pose_jumps=interpolate_pose(pose, bodyparts_xy, seconds=interpolate_seconds, pose_framerate=POSE_FRAMERATE)
        logger.debug("Imputing proboscis to head")
        pose_jumps=impute_proboscis_to_head(pose_jumps)


        pose_filters={}
        filters_sorted=sorted([(filters[f]["order"], f) for f in filters], key=lambda x: x[0])
        filters_sorted=[e[1] for e in filters_sorted]
        pose_filtered=pose_jumps

        for filt in filters_sorted:
            filter_f=filt
            window_size=filters[filt]["window_size"]
            min_window_size=filters[filt]["min_window_size"]
            
            logger.debug("Applying %s filter to pose", filt)
            pose_filtered_arr, _ = filter_pose(
                filter_f=filter_f, pose=pose_filtered, bodyparts=bodyparts,
                window_size=window_size, min_window_size=min_window_size,
                useGPU=useGPU
            )
            assert filter_f not in pose_filters
            logger.debug("Imputing proboscis to head")
            pose_filtered=impute_proboscis_to_head(arr2df(pose, pose_filtered_arr, bodyparts))
            del pose_filtered_arr
            pose_filters[filter_f]=pose_filtered
            
        return {"jumps": pose_jumps, "filters": pose_filters}
    
    def filter_pose_by_time(self, min_time, max_time, pose):
        raise NotImplementedError()


    def filter_and_interpolate_pose(self, *args, min_time=MIN_TIME, max_time=MAX_TIME, **kwargs):
        """
        Call filter_and_interpolate_pose_single_animal once for every animal in the experiment
        """
        
        pose_datasets=self.pose.groupby("id")
        all_poses=[]

        for id, pose in pose_datasets:

            # NOTE
            # this check makes sense because id is categorical
            # and if the user is loading a single fly
            # then the pose of all other flies will be empty
            # (but its id still will be in the categories)
            if pose.shape[0]==0:
                continue
            identity=pose["identity"].iloc[0].item()
            identifier=self.experiment + "__" + str(identity).zfill(2)
            pose = self.filter_and_interpolate_pose_single_animal(pose.copy(), *args, min_time=min_time, max_time=max_time, identifier=identifier, **kwargs)
            pose=self.filter_pose_by_time(min_time=min_time, max_time=max_time, pose=pose)

            all_poses.append(pose)
        
 
        logger.debug("Concatenating dataset")
        self.pose_boxcar=pd.concat(all_poses, axis=0)
        logger.debug("Done")

    @staticmethod
    def full_interpolation(pose, columns):
        logger.debug("Running interpolation on dataset of shape %s on columns %s", pose.shape, columns)
        out=[]
        for id, pose_dataset in pose.groupby("id"):
            out.append(interpolate_pose(pose_dataset.copy(), columns, seconds=None))
        pose=pd.concat(out, axis=0)
        logger.debug("Done")
        return pose

