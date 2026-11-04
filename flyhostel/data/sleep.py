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
are never mixed up (load_sleep_data(method="centroid") reads them). The centroid method also saves the activity
ethogram of the fly (activity_ethogram), read by the Ethograms tab of the validator for arena flies

    <basedir>/sleep/<NN>/<experiment>__<NN>_ethogram_centroid.feather    1 row per second: active / inactive

compute-sleep --ethogram-only makes it from the sleep file already saved, without computing the sleep again.

The blobs method (annotate_sleep_from_blobs, compute-sleep --method blobs) is a proxy of the sleep of a group of flies, one more
metric next to the others, for a group whose identities are not tracked (e.g. many flies with a drug on one side of a splitter).
It gives no sleep per fly, but the fraction of the group asleep. A sleeping fly does not move, so it needs no identity: every
second, the blobs (idtrackerai blobs in ROI_0, identified or not) are linked to the nearest blob of the previous second at most
min_displacement_mm away, as in the centroid method, and every run of at least min_time_immobile seconds is a fly asleep.
Runs survive up to MAX_BLOB_GAP seconds without a blob (e.g. a fly walks on top of a sleeping one). Its files are named
after the group instead of a fly: experiment__side1 and experiment__side2 for the 2 sides of a splitter (as in split_identities.py,
1 is left / top, 2 is right / bottom), experiment__all for all the flies

    <basedir>/sleep/side1/<experiment>__side1_sleep5min_blobs.feather         1 row per second
    <basedir>/sleep/side1/<experiment>__side1_sleep5min_blobs_bin300.feather  1 row per bin

The proxy is validated in flies that touch rarely. Flies that stay merged in a blob while asleep are counted as one

The behavior based sleep runs at the end of the behavior_prediction pipeline (compute-sleep) and in the analysis code,
when a sleep file is missing or older than the behavior file. The centroid based one runs in the sleep_centroid.nf
pipeline of behavior_prediction

Columns:
    inactive_states: number of inactive frames in the second (behavior method)
    displacement_mm: distance moved by the centroid since the previous second (centroid method)
    windowed_var: the second is immobile
    inactive_rule: the fly is asleep (in the binned trace: fraction of the bin asleep).
        Blobs method: fraction of the flies of the group asleep
    n_blobs, n_asleep, number_of_flies: blobs in the second, runs of immobility covering it, flies in the group (blobs method)
"""
import argparse
import logging
import os
import re
import sqlite3

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
# blobs method: seconds without a blob that do not end a run of immobility
MAX_BLOB_GAP=5
METHODS=["behavior", "centroid", "blobs"]


def sleep_path(basedir, animal, min_time_immobile=MIN_TIME_IMMOBILE, bin_size=None, method="behavior"):
    """
    Arguments:
        basedir (str): Folder of the experiment
        animal (str): experiment__NN e.g. FlyHostel1_3X_2024-12-10_16-00-00__01
            Blobs method: the group instead of a fly, experiment__side1, experiment__side2 or experiment__all
        bin_size (int): If None, path to the 1 s trace. Otherwise to the trace binned in windows of bin_size seconds
        method (str): behavior, centroid or blobs (see the module docstring)
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


def ethogram_path(basedir, animal, method="centroid"):
    """
    Path to the activity ethogram of a fly (experiment__NN), see activity_ethogram. Only the centroid method makes one
    """
    if method!="centroid":
        raise ValueError(f"Only the centroid method makes an activity ethogram, not {method}")
    identity=animal.split("__")[-1]
    return os.path.join(basedir, "sleep", identity, f"{animal}_ethogram_{method}.feather")


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


def link_immobile_runs(seconds, positions, min_displacement_mm=MIN_DISPLACEMENT_MM, min_time_immobile=MIN_TIME_IMMOBILE, max_gap=MAX_BLOB_GAP):
    """
    Runs of immobility of blobs without identity (blobs method, see the module docstring)

    Arguments:
        seconds (np.ndarray): Sorted seconds with blobs
        positions (list): For every second, array blobs x 2 with their position in mm

    Returns:
        list of (first second, last second) of the runs of at least min_time_immobile seconds
    """
    # [x, y, first second, last second], of the runs that can still go on
    tracks=[]
    runs=[]

    def close(track):
        if track[3]-track[2]>=min_time_immobile:
            runs.append((track[2], track[3]))

    for second, xy in zip(seconds, positions):
        # a run longer without a blob ended, even if a blob shows up in the same place
        for track in tracks:
            if second-track[3]>max_gap:
                close(track)
        tracks=[track for track in tracks if second-track[3]<=max_gap]
        linked=np.zeros(len(xy), dtype=bool)
        if tracks and len(xy):
            last=np.array([track[:2] for track in tracks])
            distance=np.hypot(last[:, None, 0]-xy[:, 0], last[:, None, 1]-xy[:, 1])
            taken=np.zeros(len(tracks), dtype=bool)
            # the closest pairs are linked first
            for i, j in zip(*np.unravel_index(np.argsort(distance, axis=None), distance.shape)):
                if distance[i, j]>min_displacement_mm:
                    break
                if not taken[i] and not linked[j]:
                    taken[i]=linked[j]=True
                    tracks[i][0], tracks[i][1]=xy[j]
                    tracks[i][3]=second
        tracks+=[[xy[j, 0], xy[j, 1], second, second] for j in np.flatnonzero(~linked)]
    for track in tracks:
        close(track)
    return runs


def annotate_sleep_from_blobs(blobs, number_of_flies=None, min_displacement_mm=MIN_DISPLACEMENT_MM, min_time_immobile=MIN_TIME_IMMOBILE, max_gap=MAX_BLOB_GAP):
    """
    Fraction of a group of flies asleep from its blobs, without identities (blobs method, see the module docstring)

    Arguments:
        blobs (pd.DataFrame): Blobs of one group, with columns id, t, frame_number, x_mm, y_mm. One frame per second is enough
        number_of_flies (int): Flies in the group. Default: the usual number of blobs in a second with blobs

    Returns:
        pd.DataFrame with one row per second between the first and the last blob, columns id, t, t_round, frame_number,
        n_blobs, n_asleep, number_of_flies, inactive_rule
    """
    if blobs["id"].nunique()!=1:
        raise ValueError("annotate_sleep_from_blobs takes the blobs of one group")
    data=blobs[["t", "frame_number", "x_mm", "y_mm"]].dropna(subset=["x_mm", "y_mm"]).copy()
    data["second"]=(data["t"]//TIME_WINDOW_LENGTH).astype(np.int64)
    # one frame per second, or the same blob would be seen more than once in the second
    data=data.loc[data["frame_number"]==data.groupby("second")["frame_number"].transform("first")]
    by_second=dict(tuple(data.groupby("second")))
    seconds=np.array(sorted(by_second))
    positions=[by_second[second][["x_mm", "y_mm"]].to_numpy() for second in seconds]
    runs=link_immobile_runs(seconds, positions, min_displacement_mm, min_time_immobile, max_gap)

    grid=np.arange(seconds[0], seconds[-1]+1)
    n_asleep=np.zeros(len(grid), dtype=np.int32)
    for first, last in runs:
        n_asleep[first-grid[0]:last-grid[0]+1]+=1
    n_blobs=np.zeros(len(grid), dtype=np.int32)
    n_blobs[seconds-grid[0]]=[len(xy) for xy in positions]
    if number_of_flies is None:
        number_of_flies=int(pd.Series(n_blobs[n_blobs>0]).mode().iloc[0])
    frames=data.groupby("second")["frame_number"].first().to_numpy()
    return pd.DataFrame({
        "id": blobs["id"].iloc[0], "t": grid.astype(float), "t_round": grid.astype(float),
        "frame_number": np.interp(grid, seconds, frames).astype(np.int64), "n_blobs": n_blobs, "n_asleep": n_asleep,
        "number_of_flies": number_of_flies, "inactive_rule": np.minimum(n_asleep, number_of_flies)/number_of_flies,
    })


def read_splitter(basedir):
    """[[x1, y1], [x2, y2]] of the splitter in <basedir>/landmarks.toml, in pixels of the frame. None if it has none"""
    landmarks=os.path.join(basedir, "landmarks.toml")
    if not os.path.exists(landmarks):
        return None
    with open(landmarks, "r") as handle:
        match=re.search(r"^\s*splitter\s*=\s*(.*)$", handle.read(), flags=re.MULTILINE)
    if match is None:
        return None
    return np.array(re.findall(r"-?\d+(?:\.\d+)?", match.group(1)), dtype=float).reshape(2, 2)


def splitter_side(points, x, y):
    """1 (left or top) or 2 (right or bottom) of every point, same as side_of in split_identities.py of splitter.nf"""
    (x1, y1), (x2, y2)=points
    dx, dy=x2-x1, y2-y1
    cross=lambda x, y: dx*(y-y1)-dy*(x-x1)
    middle_x, middle_y=(x1+x2)/2, (y1+y2)/2
    reference=(middle_x-1e6, middle_y) if abs(dy)>=abs(dx) else (middle_x, middle_y-1e6)
    return np.where(np.sign(cross(x, y))==np.sign(cross(*reference)), 1, 2)


def load_blobs(basedir, animal):
    """
    Blobs of a group (experiment__side1, experiment__side2 or experiment__all), identified or not, in the first frame of
    every second of its dbfile, with columns id, t (ZT seconds), frame_number, x_mm, y_mm, area. Frames in
    in_frame_index_mistakes.csv (see mask_unreliable_frames) are dropped
    """
    # imported here: flyhostel.utils.utils is slow to import
    from flyhostel.utils.utils import load_meta_info

    experiment, group=animal.split("__")
    if group not in ("all", "side1", "side2"):
        raise ValueError(f"The group of {animal} must be all, side1 or side2")
    dbfile=os.path.join(basedir, f"{experiment}.db")
    with sqlite3.connect(f"file:{dbfile}?mode=ro", uri=True) as connection:
        metadata=dict(connection.execute("SELECT field, value FROM METADATA WHERE field IN ('framerate', 'pixels_per_cm')").fetchall())
        framerate=int(float(metadata["framerate"]))
        blobs=pd.read_sql(f"SELECT frame_number, x, y, area FROM ROI_0 WHERE frame_number % {framerate} = 0", connection)
        times=pd.read_sql(f"SELECT frame_number, frame_time FROM STORE_INDEX WHERE frame_number % {framerate} = 0", connection)
    blobs=blobs.merge(times, on="frame_number", how="inner")

    if group!="all":
        points=read_splitter(basedir)
        if points is None:
            raise ValueError(f"{animal} is a side of a splitter, but {basedir}/landmarks.toml has none")
        blobs=blobs.loc[splitter_side(points, blobs["x"].values, blobs["y"].values)==int(group[-1])]

    pixels_per_mm=float(metadata["pixels_per_cm"])/10
    blobs=pd.DataFrame({
        "id": animal,
        "t": blobs["frame_time"].values/1000+load_meta_info(dbfile, identity=1)["t_after_ref"],
        "frame_number": blobs["frame_number"].values,
        "x_mm": blobs["x"].values/pixels_per_mm,
        "y_mm": blobs["y"].values/pixels_per_mm,
        "area": blobs["area"].values,
    }).sort_values("frame_number", ignore_index=True)
    unreliable=unreliable_rows(blobs, basedir)
    if unreliable.any():
        logger.warning("%s unreliable frames in %s. Ignoring their blobs", unreliable.sum(), basedir)
        blobs=blobs.loc[~unreliable]
    return blobs


ACTIVITY_STATES=["active", "inactive"]


def activity_ethogram(dt_sleep):
    """
    Activity ethogram of a sleep trace made by annotate_sleep_from_centroids: every second is inactive
    (windowed_var: the centroid moved <= min_displacement_mm) or active. A second without displacement
    (first second of the fly) has no behavior (None)

    Returns:
        pd.DataFrame with columns frame_number (first frame of the second), t (ZT s), behavior (ACTIVITY_STATES),
        asleep, displacement_mm
    """
    behavior=np.where(dt_sleep["windowed_var"], "inactive", "active").astype(object)
    behavior[dt_sleep["displacement_mm"].isna().values]=None
    return pd.DataFrame({
        "frame_number": dt_sleep["frame_number"].astype(np.int64).values,
        "t": dt_sleep["t"].values,
        "behavior": pd.Categorical(behavior, categories=ACTIVITY_STATES),
        "asleep": dt_sleep["inactive_rule"].astype(bool).values,
        "displacement_mm": dt_sleep["displacement_mm"].values,
    })


def save_feather(df, path):
    """Write atomically (temporary file + os.replace)"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp=path + ".tmp"
    df.reset_index(drop=True).to_feather(tmp, compression="zstd")
    os.replace(tmp, path)
    logger.info("Saved %s", path)


def make_activity_ethogram(basedir, animal, output=None):
    """
    Activity ethogram of a fly from its centroid sleep file, already saved (compute-sleep --ethogram-only)

    Returns:
        Path to the ethogram
    """
    source=sleep_path(basedir, animal, method="centroid")
    if output is not None:
        source=os.path.join(output, os.path.basename(source))
    path=ethogram_path(basedir, animal)
    if output is not None:
        path=os.path.join(output, os.path.basename(path))
    save_feather(activity_ethogram(pd.read_feather(source)), path)
    return path


def has_splitter(basedir):
    """
    True if <basedir>/landmarks.toml has a splitter (the line between 2 flies, see split_identities.py in splitter.nf)
    """
    landmarks=os.path.join(basedir, "landmarks.toml")
    if not os.path.exists(landmarks):
        return False
    with open(landmarks, "r") as handle:
        return any(re.match(r"^\s*splitter\s*=", line) for line in handle)


def load_centroids(basedir, animal):
    """
    Centroid of a fly (experiment__NN) in mm, with columns id, t (ZT seconds), frame_number, x_mm, y_mm,
    loaded from its dbfile like in predict-behavior. Frames in in_frame_index_mistakes.csv (see
    mask_unreliable_frames) are dropped

    Experiments with 2 flies separated by a splitter (splitter = ... in <basedir>/landmarks.toml, tracked by
    splitter.nf) are read without validation: the identity of a fly is the side of the splitter it is on,
    so it cannot be swapped
    """
    # imported here: flyhostel.data.pose.main imports this module
    from flyhostel.data.pose.main import FlyHostelLoader

    experiment, identity=animal.split("__")
    ensure_validated=not has_splitter(basedir)
    if not ensure_validated:
        logger.info("%s has a splitter: reading its identities without validation", basedir)
    loader=FlyHostelLoader(
        experiment=experiment, identity=int(identity), chunks=range(0, 400), ensure_validated=ensure_validated
    )
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


def compute_sleep_file(basedir, animal, input=None, output=None, method="behavior", min_displacement_mm=MIN_DISPLACEMENT_MM, number_of_flies=None):
    """
    Compute the sleep of a fly from its behavior file (or its centroid) and save it (see the module docstring)

    Arguments:
        basedir (str): Folder of the experiment
        animal (str): experiment__NN. Blobs method: experiment__side1, experiment__side2 or experiment__all
        input (str): Behavior file. Default <basedir>/motionmapper/<NN>/<animal>.feather. Only used by the behavior method
        output (str): Folder of the sleep files. Default <basedir>/sleep/<NN>
        method (str): behavior, centroid or blobs
        min_displacement_mm (float): Not used by the behavior method, see annotate_sleep_from_centroids
        number_of_flies (int): Only used by the blobs method, see annotate_sleep_from_blobs

    Returns:
        Paths to the 1 s and to the binned sleep traces (and the activity ethogram with the centroid method)
    """
    if method=="centroid":
        if input is not None:
            raise ValueError("The centroid method reads the centroid from the dbfile, it takes no input file")
        centroids=load_centroids(basedir, animal)
        if centroids.shape[0]==0:
            raise ValueError(f"No centroid data for {animal}")
        dt_sleep=annotate_sleep_from_centroids(centroids, min_displacement_mm=min_displacement_mm)
    elif method=="blobs":
        if input is not None:
            raise ValueError("The blobs method reads the blobs from the dbfile, it takes no input file")
        blobs=load_blobs(basedir, animal)
        if blobs.shape[0]==0:
            raise ValueError(f"No blobs for {animal}")
        dt_sleep=annotate_sleep_from_blobs(blobs, number_of_flies=number_of_flies, min_displacement_mm=min_displacement_mm)
    elif method=="behavior":
        if input is None:
            input=behavior_path(basedir, animal)

        behavior=load_behavior(input, basedir)
        if behavior.shape[0]==0:
            raise ValueError(f"No behavior data in {input}")

        dt_sleep=annotate_sleep(behavior, inactive_states=sleep_inactive_states(animal))
    else:
        raise ValueError(f"method must be one of {METHODS}, not {method}")
    tables=[dt_sleep]
    paths=[sleep_path(basedir, animal, method=method)]
    if method=="centroid":
        tables.append(activity_ethogram(dt_sleep))
        paths.append(ethogram_path(basedir, animal, method=method))
    # the binned trace is written last: it tells the analysis code that the sleep of the fly is complete
    tables.append(bin_sleep(dt_sleep))
    paths.append(sleep_path(basedir, animal, bin_size=BIN_SIZE, method=method))
    if output is not None:
        paths=[os.path.join(output, os.path.basename(path)) for path in paths]

    for df, path in zip(tables, paths):
        save_feather(df, path)
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
    ap=argparse.ArgumentParser(description="Compute the sleep of a fly from its behavior file or its centroid, or of a group from its blobs (see flyhostel.data.sleep)")
    ap.add_argument("--basedir", required=True, help="Folder of the experiment")
    ap.add_argument("--animal", required=True, help="experiment__NN e.g. FlyHostel1_3X_2024-12-10_16-00-00__01. blobs method: experiment__side1, experiment__side2 or experiment__all")
    ap.add_argument("--input", default=None, help="Behavior file. Default <basedir>/motionmapper/<NN>/<animal>.feather")
    ap.add_argument("--output", default=None, help="Folder of the sleep files. Default <basedir>/sleep/<NN>")
    ap.add_argument("--method", choices=METHODS, default="behavior", help="Immobility from the behavior, from the displacement of the centroid, or from the blobs of a group")
    ap.add_argument(
        "--min-displacement-mm", type=float, default=MIN_DISPLACEMENT_MM,
        help="centroid and blobs methods: a second is immobile when the centroid moved at most this much since the previous second"
    )
    ap.add_argument("--number-of-flies", type=int, default=None, help="blobs method: flies in the group. Default: the usual number of blobs per second")
    ap.add_argument(
        "--ethogram-only", action="store_true",
        help="centroid method: only make the activity ethogram, from the sleep file already in --output (default <basedir>/sleep/<NN>)"
    )
    return ap


def main():
    logging.basicConfig(level=logging.INFO)
    args=get_parser().parse_args()
    if args.ethogram_only:
        if args.method!="centroid":
            raise ValueError("--ethogram-only needs --method centroid")
        make_activity_ethogram(args.basedir, args.animal, output=args.output)
        return
    compute_sleep_file(
        args.basedir, args.animal, input=args.input, output=args.output,
        method=args.method, min_displacement_mm=args.min_displacement_mm, number_of_flies=args.number_of_flies,
    )
