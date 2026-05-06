from pathlib import Path
import logging
import os
import joblib
import pandas as pd
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.utils import get_basedir, get_number_of_animals
from flyhostel.data.interactions.sociability.main import (
    process_all_experiments,
    load_experiments
)
raise DeprecationWarning
import numpy as np

ANIMALS_CSV="/home/vibflysleep/opt/vsc-scripts/nextflow/pipelines/behavior_prediction/animals.csv"
GROUPS_CSV="/home/vibflysleep/opt/vsc-scripts/nextflow/pipelines/interaction_detection/fly_groups.csv"

N_CLUSTERS=5
WINDOW_S=1
MIN_TIME=6*3600
MAX_TIME=30*3600
TIME_WINDOW_LENGTH=1
sociability_logger=logging.getLogger("flyhostel.data.interactions.sociability.sociability")
sociability_logger.setLevel(logging.INFO)
logger=logging.getLogger(__name__)

from flyhostel.data.sleep import sleep_annotation_rf, PURE_INACTIVE_STATES
from flyhostel.data.pose.ethogram.utils import annotate_bouts, annotate_bout_duration
from tqdm.auto import tqdm

sleep_periods=[(10, "longImmobile"), (600, "longAsleep"), (300, "asleep")]
sleep_names=[f[1] for f in sleep_periods]

BEHAVIOR_FRAMERATE=30
STEP=FRAMERATE//BEHAVIOR_FRAMERATE

def load_sleep_data(loader, min_time, max_time, sleep_periods):
    """
    Annotate inactivity duration where the minimum time inactive is variable

    Arguments:
        loader (FlyHostelLoader)
        min_time (int): Min ZT
        max_time (int): Max ZT
        sleep_periods (list): Collection of tuples of length 2, containing a min_time_immobile (int) and a name for the inactivity (str)

    Returns
        loader

    loader.sleep has been populated with columns foo, foo_duration, foo_bout_in and immobile,
    where foo is each of the inactivity period names
    
    """
    loader.load_behavior_data(loader.experiment, loader.identity, min_time=min_time, max_time=max_time)
    loader.behavior["inactive_states"]=loader.behavior["prediction2"].isin(PURE_INACTIVE_STATES)
    for i, (min_time_immobile, period) in enumerate(sleep_periods):
        sleep=sleep_annotation_rf(loader.behavior, min_time_immobile=min_time_immobile).rename({"inactive_rule": period, "windowed_var": "immobile"}, axis=1).drop(["t"], axis=1)
        if i==0:
            loader.sleep=sleep
        else:
            loader.sleep=loader.sleep.merge(sleep[["id", "t_round", period]], on=["id", "t_round"])

        loader.sleep=annotate_bout_duration(annotate_bouts(loader.sleep, variable=period), fps=1).reset_index(drop=True).rename({
            "bout_in": f"{period}_bout_in",
            "duration": f"{period}_duration"
        }, axis=1) 
    return loader

def immobility_annotation(loader, interactions_database, sleep_columns=None, rename_dict=None):
    """
    Annotate interactions database with immobility states

    t_ref: Time representative of the interaction
    t_raw: Time at which contact is closest 
    """
    interactions_database["t_round"]=interactions_database["t_ref"]//1
    interactions_database["t_till_next"]=np.nan
    interactions_database["t_till_next"].iloc[:-1]=np.diff(interactions_database["t_raw"])
    if sleep_columns is None:
        sleep_columns=sleep.columns.tolist()
    elif "t_round" not in sleep_columns:
        sleep_columns=["t_round"]+sleep_columns
        
    sleep=loader.sleep[sleep_columns].copy()

    if rename_dict is not None:
        sleep.rename(rename_dict, axis=1, inplace=True)
    
    interactions_database=interactions_database.merge(sleep, on="t_round").sort_values("frame_number")   
    return interactions_database

def immobility_annotations(loader, interactions_database, framerate):
    """
    Annotate immobility state before and after the interaction start and end resp.
    """

    # Find immobility state before (PRE) the interaction
    interactions_database["t_ref"]=interactions_database["t_raw"]-(interactions_database["frame_number"]-interactions_database["first_frame"])/framerate - 1
    sleep_columns=sleep_names + [f"{feat}_bout_in" for feat in sleep_names] + [f"{feat}_duration" for feat in sleep_names]
    rename_dict={col: f"pre_{col}" for col in sleep_columns}
    interactions_database=immobility_annotation(
        loader, interactions_database,
        sleep_columns=sleep_columns,
        rename_dict=rename_dict
    )
    
    # Find immobility state after (POST) the interaction
    interactions_database["t_ref"]=interactions_database["t_raw"]+(interactions_database["last_frame_number"]-interactions_database["frame_number"])/framerate - 1
    rename_dict={col: f"post_{col}" for col in sleep_columns}
    interactions_database=immobility_annotation(
        loader, interactions_database, framerate,
        sleep_columns=sleep_columns,
        rename_dict=rename_dict
    )
    for sleep_name in sleep_names:
        interactions_database[f"pre_{sleep_name}_time_before_interaction"]=interactions_database[f"pre_{sleep_name}_bout_in"]*TIME_WINDOW_LENGTH
        interactions_database[f"post_{sleep_name}_time_after_interaction"]=interactions_database[f"post_{sleep_name}_duration"]-interactions_database[f"post_{sleep_name}_bout_in"]*TIME_WINDOW_LENGTH
    return interactions_database



def annotate_behavior_database_id(interactions_database, behavior):
    # annotate row_id_start and row_id_end
    behavior["row_id"]=np.arange(behavior.shape[0])
    interactions_database=interactions_database\
    .merge(behavior[["frame_number", "row_id"]].rename({
        "frame_number": "frame_number_start",
        "row_id": "row_id_start"
    }, axis=1), on="frame_number_start", how="inner")\
    .merge(behavior[["frame_number", "row_id"]].rename({
        "frame_number": "frame_number_end",
        "row_id": "row_id_end"
    }, axis=1), on="frame_number_end", how="inner")
    return interactions_database


def compute_behavior_features(interaction_database, behavior, half_period_s=5):
    """
    Compute numerical descriptors of each interaction as a function of the behaviors produced during the interaction

    Descriptors:
        inactive+rejection_max
        inactive+rejection_mean        
        centroid_speed        
    """

    idx["frame_number_raw"]=interaction_database["frame_number"].copy()
    interactions_database["frame_number"]=STEP*(interactions_database["frame_number_raw"]//STEP)

    interactions_database["frame_number_start"]=interactions_database["frame_number"]-half_period_s*FRAMERATE
    interactions_database["frame_number_end"]=interactions_database["frame_number"]+half_period_s*FRAMERATE

    interactions_database=annotate_behavior_database_id(interactions_database, behavior)
    stats=[]
    intervals=zip(interactions_database["row_id_start"], interactions_database["row_id_end"])
    prob_features=[]
    for interval in tqdm(intervals, total=interactions_database.shape[0]):
        probs=behavior["inactive+rejection"].iloc[interval[0]:interval[1]]
        speed=behavior["centroid_speed"].iloc[interval[0]:interval[1]]
        stats.append([probs.max(), probs.mean(), speed.sum()])
        probs=probs.T
        prob_features.append(probs.tolist())

    stats=pd.DataFrame.from_records(
        stats,
        columns=["inactive+rejection_max", "inactive+rejection_mean", "centroid_speed"],
        index=interactions_database.index
    )
    interactions_database=pd.concat([interactions_database, stats], axis=1)
    return interaction_database, prob_features


def annotate_interaction_database_using_behavior_data(interactions_database, loaders, half_period_s=5):
    """
    Add to the features associated to an interaction database
    the features computed by compute_behavior_features
    Add to the interaction database single-number descriptors of these features

    Example: add pre_longImmobile and post_longImmobile
    """
    
    interactions_database["t_raw"]=interactions_database["t"]
    interactions_database_non_nan=interactions_database.loc[interactions_database["keep"]==True]

    # a list to collect one element per animal in this experiment
    interactions_database_l=[]
    features_l=[]
    probs_l=[]
    experiment=loaders[0].experiment
    
    for loader in tqdm(loaders, desc=f"Annotating {experiment}"):
        focal_fly_indices=interactions_database["id"]==loader.ids[0]
        load_sleep_data(loader, min_time=MIN_TIME, max_time=MAX_TIME, sleep_periods=sleep_periods)
        idx=immobility_annotations(
            loader,
            interactions_database.loc[focal_fly_indices],
            FRAMERATE
        )
        focal_database, prob_features=compute_behavior_features(idx, loader.behavior, half_period_s=half_period_s)

        indices_non_nan=interactions_database_non_nan["id"]==loader.ids[0]
        features_l.append(features.iloc[np.where(indices_non_nan)])
        interactions_database_l.append(focal_database)
        probs_l.extend(prob_features)
    
    interactions_database=pd.concat(interactions_database_l, axis=0).reset_index(drop=True)
    prob_features=pd.DataFrame.from_records(probs_l)
    prob_features=prob_features.loc[interactions_database["keep"]==True]
    n_steps=half_period_s*2*BEHAVIOR_FRAMERATE
    
    prob_features.columns=pd.MultiIndex.from_arrays([
        ["probability",]*n_steps,
        np.arange(-half_period_s*FRAMERATE, half_period_s*FRAMERATE,5),
        ["1",]*n_steps
    ], names=["feature", "position", "timepoint"]
    )
    
    features=pd.concat(features_l, axis=0)
    nrows=features.shape[0]
    
    features=pd.concat([
        features.reset_index(drop=True),
        prob_features.reset_index(drop=True)
    ], axis=1)
    assert features.shape[0]==nrows
    return interactions_database, features
 

def process_experiment(experiment, number_of_animals, half_period_s=5):

    """
    Annotate immobility state of each fly involved in interactions detected in a flyhostel experiment
    """
    
    index_csv=get_basedir(experiment) + f"/interactions/{experiment}_index.csv"
    features_pkl=get_basedir(experiment) + f"/interactions/{experiment}_features.pkl"
    
    if not os.path.exists(index_csv):
        logger.error("%s not found", index_csv)
        return None
        
    identities=list(range(1, number_of_animals+1))
    loaders=[FlyHostelLoader(experiment=experiment, identity=identity) for identity in identities]
    interactions_database=pd.read_csv(index_csv)
    features=pd.read_pickle(features_pkl)
    interactions_database, features=annotate_interaction_database_using_behavior_data(
        interactions_database,
        loaders,
        half_period_s=half_period_s
    )
    return interactions_database, features, loaders

