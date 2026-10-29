"""
Sleep of a fly, from its behavior predictions or from its centroid (centroid method, below). This is the only
definition of sleep in flyhostel (sleep_from_centroid_velocity is the velocity based alternative for arrays of
speed, used by the social pipeline and simulations):

    1. every frame is inactive or active depending on its prediction2:
       active = walk, feed, groom and background (ACTIVE_STATES), the rest is inactive (INACTIVE_STATES)
    2. every second (time_window_length) is immobile if more than 1/3 (INACTIVE_FRACTION) of its frames are inactive.
       The frames of a second are the usual number of frames per second of the fly (the median), so that the rule
       is the same whatever the framerate of the behavior: > 10 of 30 frames, > 15.7 of 47
    3. the fly is asleep during every run of at least min_time_immobile (300) immobile seconds,
       from the start of the run (ethoscopy's sleep_contiguous, as sleepr in R)
    4. the binned trace is the fraction of every bin_size (300) seconds spent asleep

In arena experiments (sleep_inactive_states), feed is inactive too: their behavior model has no food_distance, and
feed is mostly what the rules make of long proboscis extensions (inactive+pe), so it would break sleep bouts with PE.
Walking still breaks them. Sleep is somewhat overestimated, by any immobile feeding

Other immobility durations (e.g. the 10 s of the interactions pipeline) come from the windowed_var of the sleep file
(immobility_bouts, load_sleep_file), so that they follow this definition too

The centroid method (annotate_sleep_from_centroids, compute-sleep --method centroid) skips the behavior: steps 1 and 2
become "a second is immobile when the centroid moved <= min_displacement_mm (MIN_DISPLACEMENT_MM) since the previous
second", comparing the median position of the frames of each second. Steps 3 and 4 are the same. It is meant for the
arena experiments, where the pose (and so the behavior) is unreliable when the fly is on the wall and seen sideways,
and it needs no pose, wavelets or behavior model. Micromovements (grooming, PE) barely move the centroid, so they do
not break sleep bouts. A second without centroid is skipped, like a second without behavior; the first second is
not immobile

compute_sleep_file does all of it for one fly, and saves

    <basedir>/sleep/<NN>/<experiment>__<NN>_sleep5min.feather         1 row per second
    <basedir>/sleep/<NN>/<experiment>__<NN>_sleep5min_bin300.feather   1 row per bin

or, with the centroid method, _sleep5min_centroid.feather and _sleep5min_centroid_bin300.feather, so that the two
are never mixed up (load_sleep_data(method="centroid") reads them).

The behavior based sleep runs at the end of the behavior_prediction pipeline (compute-sleep) and in the analysis code,
when a sleep file is missing or older than the behavior file. The centroid based one runs in the sleep_centroid.nf
pipeline of behavior_prediction

Columns:
    inactive_states: number of inactive frames in the second (behavior method)
    displacement_mm: distance moved by the centroid since the previous second (centroid method)
    windowed_var: the second is immobile
    inactive_rule: the fly is asleep (in the binned trace: fraction of the bin asleep)
"""
import argparse
import logging
import os

import numpy as np
import pandas as pd
from ethoscopy.analyse import sleep_contiguous
from flyhostel.data.pose.ethogram.utils import annotate_bouts, annotate_bout_duration
from flyhostel.utils.experiment import split_arena

logger=logging.getLogger(__name__)

ACTIVE_STATES=["walk", "feed", "groom", "background"]
INACTIVE_STATES=["inactive", "inactive+pe", "inactive+micromovement", "inactive+rejection"]
ARENA_INACTIVE_STATES=INACTIVE_STATES+["feed"]

TIME_WINDOW_LENGTH=1 # seconds
INACTIVE_FRACTION=1/3
MIN_TIME_IMMOBILE=300 # seconds
BIN_SIZE=300 # seconds
# centroid method: a second is immobile when the centroid moved this much or less since the previous second.
# About 1/3 of a body length, and the speed above which a fly moves in flyhostel.data.social (MovementConfig). CALIBRATE
MIN_DISPLACEMENT_MM=1.0
METHODS=["behavior", "centroid"]


def sleep_path(basedir, animal, min_time_immobile=MIN_TIME_IMMOBILE, bin_size=None, method="behavior"):
    """
    Arguments:
        basedir (str): Folder of the experiment
        animal (str): experiment__NN e.g. FlyHostel1_3X_2024-12-10_16-00-00__01
        bin_size (int): If None, path to the 1 s trace. Otherwise to the trace binned in windows of bin_size seconds
        method (str): behavior or centroid (see the module docstring)
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method}")
    identity=animal.split("__")[-1]
    basename=f"{animal}_sleep{min_time_immobile//60}min"
    if method!="behavior":
        basename+=f"_{method}"
    if bin_size is not None:
        basename+=f"_bin{bin_size}"
    return os.path.join(basedir, "sleep", identity, basename + ".feather")


def behavior_path(basedir, animal):
    identity=animal.split("__")[-1]
    return os.path.join(basedir, "motionmapper", identity, animal + ".feather")


def sleep_inactive_states(animal):
    """
    Behaviors that count as inactive in the sleep of a fly (experiment__NN): ARENA_INACTIVE_STATES in arena
    experiments (see the module docstring), INACTIVE_STATES otherwise
    """
    experiment=animal.split("__")[0]
    if split_arena(experiment)[1] is not None:
        return ARENA_INACTIVE_STATES
    return INACTIVE_STATES


def load_behavior(path, basedir):
    """
    Load the behavior of a fly to compute its sleep

    Seconds next to a frame listed in <basedir>/flyhostel/validation/in_frame_index_mistakes.csv
    take the behavior of the previous frame
    """
    behavior=pd.read_feather(path, columns=["id", "t", "frame_number", "prediction2"])
    duplicated=behavior["t"].duplicated()
    if duplicated.any():
        logger.warning("%s duplicated timestamps in %s", duplicated.sum(), path)
        behavior=behavior.loc[~duplicated]
    return mask_unreliable_frames(behavior, basedir)


def mask_unreliable_frames(behavior, basedir):
    """
    in_frame_index_mistakes.csv, if it exists, lists frames where in_frame_index repeats in the same frame due to an
    error in the validation (bug of the in_frame_index assigner, detected on 2026-01-07).
    The identity of the flies is not reliable in these frames, and neither is their behavior in the surrounding seconds:
    the behavior there is replaced by the last reliable one
    """
    unreliable=unreliable_rows(behavior, basedir)
    if unreliable.any():
        logger.warning("%s unreliable frames in %s. Taking the behavior from the past", unreliable.sum(), basedir)
        behavior=behavior.copy()
        behavior.loc[unreliable, "prediction2"]=np.nan
        behavior["prediction2"]=behavior["prediction2"].ffill()
    return behavior


def unreliable_rows(data, basedir):
    """
    Rows of data (columns t and frame_number) in the seconds around a frame listed in in_frame_index_mistakes.csv
    (see mask_unreliable_frames). All False if the file does not exist
    """
    mistakes_path=os.path.join(basedir, "flyhostel", "validation", "in_frame_index_mistakes.csv")
    if not os.path.exists(mistakes_path):
        return pd.Series(False, index=data.index)

    unreliable_frames=pd.read_csv(mistakes_path)["frame_number"].drop_duplicates()
    unreliable_t=data.loc[data["frame_number"].isin(unreliable_frames), "t"].drop_duplicates().values.round().astype(int)
    unreliable_t=set(unreliable_t-1) | set(unreliable_t) | set(unreliable_t+1)
    return (data["t"]//1).astype(int).isin(unreliable_t)


def annotate_sleep(behavior, min_time_immobile=MIN_TIME_IMMOBILE, time_window_length=TIME_WINDOW_LENGTH, inactive_fraction=INACTIVE_FRACTION, inactive_states=INACTIVE_STATES):
    """
    Arguments:
        behavior (pd.DataFrame): Behavior of one or more flies, with columns id, t, frame_number, prediction2
        min_time_immobile (int): Seconds of immobility needed to call sleep. Other values give other immobility
            durations e.g. 10 s
        inactive_fraction (float): A window is immobile when more than this fraction of its frames are inactive
        inactive_states (list): Behaviors that are inactive (see sleep_inactive_states)

    Returns:
        pd.DataFrame with one row per fly and window of time_window_length seconds
    """
    unknown=set(behavior["prediction2"].dropna().unique()) - set(ACTIVE_STATES) - set(inactive_states)
    if unknown:
        logger.warning("Behaviors %s are neither active nor inactive: they count as active", sorted(unknown))

    data=behavior[["id", "t", "frame_number"]].copy()
    data["inactive_states"]=behavior["prediction2"].isin(inactive_states)
    data["t_round"]=time_window_length*(data["t"]//time_window_length)

    windows=data.groupby(["id", "t_round"]).agg(
        inactive_states=("inactive_states", "sum"),
        n_frames=("inactive_states", "size"),
        frame_number=("frame_number", "first"),
    ).reset_index()
    windows["t"]=windows["t_round"]
    # usual number of frames per window of each fly: a window with a few missing frames is judged like the others
    frames_per_window=windows.groupby("id")["n_frames"].transform("median")
    windows["windowed_var"]=windows["inactive_states"]>inactive_fraction*frames_per_window
    windows["inactive_rule"]=apply_min_time_immobile(windows, time_window_length, min_time_immobile)
    return windows[["id", "t", "t_round", "frame_number", "inactive_states", "windowed_var", "inactive_rule"]]


def apply_min_time_immobile(windows, time_window_length, min_time_immobile):
    """
    Asleep (bool array) during every run of at least min_time_immobile seconds of immobile windows (windowed_var)
    of each fly. windows must be sorted by id (grouped) and time
    """
    return np.concatenate([
        sleep_contiguous(~fly["windowed_var"], 1/time_window_length, min_valid_time=min_time_immobile)
        for _, fly in windows.groupby("id", sort=False)
    ]).astype(bool)


def annotate_sleep_from_centroids(centroids, min_displacement_mm=MIN_DISPLACEMENT_MM, min_time_immobile=MIN_TIME_IMMOBILE, time_window_length=TIME_WINDOW_LENGTH):
    """
    Sleep from the displacement of the centroid, instead of the behavior (centroid method, see the module docstring)

    Arguments:
        centroids (pd.DataFrame): Centroid of one or more flies, with columns id, t, frame_number, x_mm, y_mm
            NaN positions are ignored
        min_displacement_mm (float): A window is immobile when the median centroid of its frames is at most
            this far from the one of the previous window (with centroid) of the fly

    Returns:
        pd.DataFrame with one row per fly and window of time_window_length seconds with centroid
    """
    data=centroids[["id", "t", "frame_number", "x_mm", "y_mm"]].dropna(subset=["x_mm", "y_mm"]).copy()
    data["t_round"]=time_window_length*(data["t"]//time_window_length)

    windows=data.groupby(["id", "t_round"], observed=True).agg(
        x_mm=("x_mm", "median"),
        y_mm=("y_mm", "median"),
        n_frames=("x_mm", "size"),
        frame_number=("frame_number", "first"),
    ).reset_index()
    windows["t"]=windows["t_round"]
    step=windows.groupby("id", sort=False, observed=True)[["x_mm", "y_mm"]].diff()
    windows["displacement_mm"]=np.hypot(step["x_mm"], step["y_mm"])
    # NaN (first window of the fly) is not immobile
    windows["windowed_var"]=windows["displacement_mm"]<=min_displacement_mm
    windows["inactive_rule"]=apply_min_time_immobile(windows, time_window_length, min_time_immobile)
    return windows[["id", "t", "t_round", "frame_number", "n_frames", "displacement_mm", "windowed_var", "inactive_rule"]]


def load_centroids(basedir, animal):
    """
    Centroid of a fly (experiment__NN) in mm, with columns id, t (ZT seconds), frame_number, x_mm, y_mm,
    loaded from its dbfile like in predict-behavior. Frames in in_frame_index_mistakes.csv (see
    mask_unreliable_frames) are dropped
    """
    # imported here: flyhostel.data.pose.main imports this module
    from flyhostel.data.pose.main import FlyHostelLoader

    experiment, identity=animal.split("__")
    loader=FlyHostelLoader(experiment=experiment, identity=int(identity), chunks=range(0, 400))
    loader.load_centroid_data(identity_table=loader.identity_table, roi_0_table=loader.roi_0_table)
    if loader.dt is None:
        raise ValueError(f"No centroid data for {animal}")

    dt=loader.dt
    centroids=pd.DataFrame({
        "id": dt["id"].astype(str).values,
        "t": dt["t"].values,
        "frame_number": dt["frame_number"].values,
        "x_mm": (dt["center_x"]/loader.pixels_per_mm).values,
        "y_mm": (dt["center_y"]/loader.pixels_per_mm).values,
    }).sort_values("frame_number", ignore_index=True)

    unreliable=unreliable_rows(centroids, basedir)
    if unreliable.any():
        logger.warning("%s unreliable frames in %s. Ignoring their centroid", unreliable.sum(), basedir)
        centroids=centroids.loc[~unreliable]
    return centroids


def immobility_bouts(dt_sleep, min_time_immobile, time_window_length=TIME_WINDOW_LENGTH):
    """
    Runs of at least min_time_immobile immobile seconds (windowed_var) of a sleep trace made by annotate_sleep
    (e.g. a sleep file), for immobility durations other than the one of the file (e.g. 10 s)

    Returns:
        np.ndarray of bool, aligned to the rows of dt_sleep (sorted by id and t)
    """
    return np.concatenate([
        sleep_contiguous(~fly["windowed_var"], 1/time_window_length, min_valid_time=min_time_immobile)
        for _, fly in dt_sleep.groupby("id", sort=False)
    ])


def load_sleep_file(basedir, animal):
    """
    Sleep of a fly, 1 row per second (see the module docstring), computed first if it does not exist yet
    """
    path=sleep_path(basedir, animal)
    if not os.path.exists(path):
        compute_sleep_file(basedir, animal)
    return pd.read_feather(path)


def bin_sleep(dt_sleep, bin_size=BIN_SIZE):
    """
    Fraction of every bin of bin_size seconds spent asleep, and the first frame of the bin
    """
    data=dt_sleep[["id", "t", "frame_number", "inactive_rule"]].copy()
    data["t_round"]=bin_size*(data["t"]//bin_size)
    return data.groupby(["id", "t_round"]).agg({
        "inactive_rule": "mean",
        "frame_number": "first",
    }).reset_index()


def compute_sleep_file(basedir, animal, input=None, output=None, method="behavior", min_displacement_mm=MIN_DISPLACEMENT_MM):
    """
    Compute the sleep of a fly from its behavior file (or its centroid) and save it (see the module docstring)

    Arguments:
        basedir (str): Folder of the experiment
        animal (str): experiment__NN
        input (str): Behavior file. Default <basedir>/motionmapper/<NN>/<animal>.feather. Not used by the centroid method
        output (str): Folder of the sleep files. Default <basedir>/sleep/<NN>
        method (str): behavior or centroid
        min_displacement_mm (float): Only used by the centroid method, see annotate_sleep_from_centroids

    Returns:
        Paths to the 1 s and to the binned sleep traces
    """
    if method=="centroid":
        if input is not None:
            raise ValueError("The centroid method reads the centroid from the dbfile, it takes no input file")
        centroids=load_centroids(basedir, animal)
        if centroids.shape[0]==0:
            raise ValueError(f"No centroid data for {animal}")
        dt_sleep=annotate_sleep_from_centroids(centroids, min_displacement_mm=min_displacement_mm)
    elif method=="behavior":
        if input is None:
            input=behavior_path(basedir, animal)

        behavior=load_behavior(input, basedir)
        if behavior.shape[0]==0:
            raise ValueError(f"No behavior data in {input}")

        dt_sleep=annotate_sleep(behavior, inactive_states=sleep_inactive_states(animal))
    else:
        raise ValueError(f"method must be one of {METHODS}, not {method}")
    dt_bin=bin_sleep(dt_sleep)

    paths=[sleep_path(basedir, animal, method=method), sleep_path(basedir, animal, bin_size=BIN_SIZE, method=method)]
    if output is not None:
        paths=[os.path.join(output, os.path.basename(path)) for path in paths]

    # the binned trace is written last: it tells the analysis code that the sleep of the fly is complete
    for df, path in zip([dt_sleep, dt_bin], paths):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp=path + ".tmp"
        df.reset_index(drop=True).to_feather(tmp, compression="zstd")
        os.replace(tmp, path)
        logger.info("Saved %s", path)
    return paths


def sleep_from_centroid_velocity(speed, fps, speed_threshold, min_time_immobile=MIN_TIME_IMMOBILE, min_time_moving=0):
    """
    Sleep from the speed of the centroid of the fly, NOT from its pose based behavior (that is annotate_sleep).
    For data without behavior predictions: the social pipeline (flyhostel.data.social) and simulated flies
    (flyhostel.modelling)

        1. the fly is moving while its speed is above speed_threshold (unknown speed counts as moving:
           immobility that was not observed is not assumed)
        2. moving runs shorter than min_time_moving seconds are ignored (tracking blips)
        3. the fly is asleep during every immobile run of at least min_time_immobile seconds

    Arguments:
        speed (np.ndarray): (T,) speed of one fly or (T, N) of N flies, sampled at fps
        fps (float): Samples per second
        speed_threshold (float): Speed above which the fly moves, in the units of speed

    Returns:
        moving, asleep (np.ndarray): bool arrays with the shape of speed
    """
    speed=np.asarray(speed, dtype=float)
    if speed.ndim==2:
        out=[sleep_from_centroid_velocity(speed[:, i], fps, speed_threshold, min_time_immobile, min_time_moving) for i in range(speed.shape[1])]
        return np.stack([o[0] for o in out], axis=1), np.stack([o[1] for o in out], axis=1)

    with np.errstate(invalid="ignore"):
        moving=~(speed<=speed_threshold)
    moving=_keep_long_runs(moving, max(0, int(round(min_time_moving*fps))))
    asleep=_keep_long_runs(~moving, max(1, int(round(min_time_immobile*fps))))
    return moving, asleep


def _keep_long_runs(mask, min_len):
    """Set the True runs of mask shorter than min_len samples to False"""
    mask=np.asarray(mask, dtype=bool).copy()
    if min_len<=1:
        return mask
    edges=np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    for start, end in zip(np.flatnonzero(edges==1), np.flatnonzero(edges==-1)):
        if end-start<min_len:
            mask[start:end]=False
    return mask


def compute_time_asleep(df, column="asleep"):
    """
    Compute time asleep at each timepoint
    time asleep = seconds since the start of the current bout of sleep
    Arguments:
        df (pd.DataFrame): Contains columns id, t, and asleep
    Returns:
        df (pd.DataFrame): Contains columns time_asleep and others
    """

    df=annotate_bout_duration(annotate_bouts(df, variable=column), fps=1)
    df[f"time_{column}"]=df["bout_in"].copy()
    df.loc[~df[column], f"time_{column}"]=-1
    df.drop(["bout_out"], axis=1, inplace=True)
    df.rename({"duration": f"duration_{column}"}, axis=1, inplace=True)
    return df


def get_parser():
    ap=argparse.ArgumentParser(description="Compute the sleep of a fly from its behavior file or its centroid (see flyhostel.data.sleep)")
    ap.add_argument("--basedir", required=True, help="Folder of the experiment")
    ap.add_argument("--animal", required=True, help="experiment__NN e.g. FlyHostel1_3X_2024-12-10_16-00-00__01")
    ap.add_argument("--input", default=None, help="Behavior file. Default <basedir>/motionmapper/<NN>/<animal>.feather")
    ap.add_argument("--output", default=None, help="Folder of the sleep files. Default <basedir>/sleep/<NN>")
    ap.add_argument("--method", choices=METHODS, default="behavior", help="Immobility from the behavior or from the displacement of the centroid")
    ap.add_argument(
        "--min-displacement-mm", type=float, default=MIN_DISPLACEMENT_MM,
        help="centroid method: a second is immobile when the centroid moved at most this much since the previous second"
    )
    return ap


def main():
    logging.basicConfig(level=logging.INFO)
    args=get_parser().parse_args()
    compute_sleep_file(
        args.basedir, args.animal, input=args.input, output=args.output,
        method=args.method, min_displacement_mm=args.min_displacement_mm,
    )
