import pandas as pd
import numpy as np
import functools

from flyhostel.data.interactions.touch_from_pose.loaders import load_experiment_features
from flyhostel.utils import (
    get_pixels_per_mm,
    get_framerate
)

from .bin import bin_features
from .frames import build_frames_parallel
from .features import extract_features_timeseries


@functools.lru_cache
def extract_social_features(experiment, step_seconds=1, min_time=None, max_time=None, bin_seconds=None, on_overlap="raise", _version="v2"):
   
    px_per_mm=get_pixels_per_mm(experiment)
    fps=get_framerate(experiment)
    pose_features=load_experiment_features(
        experiment = experiment,
        extra_bodyparts  = ["lW", "rW"],
        min_time = min_time,
        max_time = max_time,
        on_overlap = on_overlap,
    )

    n_workers=30

    frames = build_frames_parallel(pose_features, fps=fps, step_seconds=step_seconds, n_workers=n_workers)
    
    # 1BL = 2.5 mm
    X_arr, keys=extract_features_timeseries(
        frames,
        contact_radius = 1 * px_per_mm, # mm
        local_radius = 5 * px_per_mm,
        n_workers=n_workers
    )

    X=pd.DataFrame(
        X_arr,
        columns=keys,
        index=pd.MultiIndex.from_tuples(
            [(gf.t, gf.frame_number) for gf in frames],
            names=["t", "frame_number"]
        )        
    )

    drop_features=[
        "pw_prob_to_body_min",
        "pw_prob_to_body_mean",
        "pw_prob_to_body_max",
        "pw_prob_to_body_std",
        "pw_min_leg_dist_min",
        "pw_min_leg_dist_mean",
        "pw_min_leg_dist_max",
        "pw_min_leg_dist_std"
    ]
    X.drop(drop_features, axis=1, inplace=True)
    X_raw=X.copy()
    X=X[~(np.isnan(X_raw).any(axis=1))]

    if X.shape[0]==0:
        import ipdb; ipdb.set_trace()

    X["experiment"]=experiment
    X=X.reset_index().set_index(["t", "frame_number", "experiment"])


    keys=X.columns
    times = X.index.get_level_values(X.index.names.index("t"))
    frame_numbers = X.index.get_level_values(X.index.names.index("frame_number"))
    if bin_seconds is not None:
        X_binned, keys_binned, bin_times = bin_features(
            X         = X.values,
            keys      = keys,
            frame_ids = (times, frame_numbers),       # the subsampled time array from build_frames_parallel
            fps       = fps / int(fps * step_seconds),
            bin_seconds = bin_seconds
        )
        X_binned=pd.DataFrame(X_binned, columns = keys_binned)

        X_binned["t"]=bin_times[:, 0]
        X_binned["frame_number"]=bin_times[:, 1]
        
        X_binned["experiment"]=experiment
        X_binned.set_index(["experiment", "t", "frame_number"], inplace=True)
        
        X=X_binned
    else:
        pass


    return X



