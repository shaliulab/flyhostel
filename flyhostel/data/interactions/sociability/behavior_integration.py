import os.path
import pickle
import logging

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from flyhostel.data.sleep import sleep_annotation_rf
from flyhostel.data.sleep import INACTIVE_STATES
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.ethogram.utils import annotate_bouts, annotate_bout_duration
from flyhostel.utils import get_basedir, get_number_of_animals
from flyhostel.data.pose.main import FlyHostelLoader

sleep_periods=[(10, "longImmobile"), (600, "longAsleep"), (300, "asleep")]
sleep_names=[f[1] for f in sleep_periods]

BEHAVIOR_FRAMERATE=30
STEP=FRAMERATE//BEHAVIOR_FRAMERATE
WINDOW_S=1
MIN_TIME=6*3600
MAX_TIME=30*3600
TIME_WINDOW_LENGTH=1

logger=logging.getLogger(__name__)

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
    loader.load_behavior_data(min_time=min_time, max_time=max_time)
    loader.behavior["inactive_states"]=loader.behavior["prediction2"].isin(INACTIVE_STATES)
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

def immobility_annotation(loader, interactions_database, framerate, sleep_columns=None, rename_dict=None):
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
    Annotate immobility state before and after the interaction start and end respectively

    Calls immobility_annotation
    """

    # Find immobility state before (PRE) the interaction
    interactions_database["t_ref"]=interactions_database["t_raw"]-(interactions_database["frame_number"]-interactions_database["first_frame"])/framerate - 1
    sleep_columns=sleep_names + [f"{feat}_bout_in" for feat in sleep_names] + [f"{feat}_duration" for feat in sleep_names]
    rename_dict={col: f"pre_{col}" for col in sleep_columns}
    interactions_database=immobility_annotation(
        loader, interactions_database, framerate,
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


def compute_behavior_features(interactions_database, behavior, window_s):
    """
    Compute numerical descriptors of each interaction as a function of the behaviors produced during the interaction

    Descriptors:
        inactive+rejection_max
        inactive+rejection_mean        
        centroid_speed        
    """

    interactions_database["frame_number_raw"]=interactions_database["frame_number"].copy()
    interactions_database["frame_number"]=STEP*(interactions_database["frame_number_raw"]//STEP)

    interactions_database["frame_number_start"]=interactions_database["frame_number"]-window_s/2*FRAMERATE
    interactions_database["frame_number_end"]=interactions_database["frame_number"]+window_s/2*FRAMERATE

    interactions_database=annotate_behavior_database_id(interactions_database, behavior)
    stats=[]
    intervals=zip(interactions_database["row_id_start"], interactions_database["row_id_end"])
    prob_features=[]
    for interval in tqdm(intervals, total=interactions_database.shape[0]):
        probs=behavior["inactive+rejection"].iloc[interval[0]:interval[1]]
        speed=behavior["centroid_speed"].iloc[interval[0]:interval[1]]
        stats.append([
            probs.max(),
            np.round(probs.mean(), 3),
            np.round(speed.sum(), 3)
        ])
        probs=probs.T
        prob_features.append(probs.tolist())

    stats=pd.DataFrame.from_records(
        stats,
        columns=[
            "inactive+rejection_max",
            "inactive+rejection_mean",
            "centroid_speed"
        ],
        index=interactions_database.index
    )
    interactions_database=pd.concat([interactions_database, stats], axis=1)
    
    diff=interactions_database["inactive+rejection_max"]-np.array([np.max(e) for e in prob_features]) # OK
    return interactions_database, prob_features


def annotate_interaction_database_using_behavior_data(interactions_database, features, loaders, window_s=10):
    """
    Add to the features associated to an interaction database
    the features computed by compute_behavior_features
    Add to the interaction database single-number descriptors of these features 
    """

    assert len(loaders)>0
    
    interactions_database["t_raw"]=interactions_database["t"]
    # interactions_database_non_nan=interactions_database.loc[interactions_database["keep"]==True]

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
        focal_database, prob_features=compute_behavior_features(idx, loader.behavior, window_s)
        interactions_database_l.append(focal_database)
        probs_l.extend(prob_features)
        features_l.append(features.iloc[np.where(focal_fly_indices)])
    
    interactions_database=pd.concat(interactions_database_l, axis=0).reset_index(drop=True)
    prob_features=pd.DataFrame.from_records(probs_l)
    features=pd.concat(features_l, axis=0).reset_index(drop=True)

    assert interactions_database.shape[0]==prob_features.shape[0]
    
    # prob_features=prob_features.loc[interactions_database["keep"]==True]
    n_steps=window_s*BEHAVIOR_FRAMERATE
    
    prob_features.columns=pd.MultiIndex.from_arrays([
            ["probability",]*n_steps,
            np.arange(-window_s/2*FRAMERATE, window_s/2*FRAMERATE, FRAMERATE//BEHAVIOR_FRAMERATE),
            ["1",]*n_steps
        ],
        names=["feature", "position", "timepoint"]
    )
    
    nrows=features.shape[0]
    features_all=pd.concat([
        features,
        prob_features
    ], axis=1)
    assert features_all.shape[0]==nrows
    diff=interactions_database["inactive+rejection_max"]-features_all.iloc[:, -prob_features.shape[1]:].max(axis=1) # OK

    return interactions_database, features_all





COLUMNS=[
    "idx", "index", "label", "first_frame", "frame_number", "last_frame_number", "t", "centroid_speed", "inactive+rejection_max", "inactive+rejection_mean",
    "pre_longImmobile", "pre_longImmobile_duration", "post_longImmobile", "post_longImmobile_duration", "interaction_duration", "id", "nn",
]

def process_experiment(experiment, number_of_animals, window_s=10):
    """
    Annotate immobility state of each fly involved in interactions detected in a flyhostel experiment
    """
    
    # index_csv=get_basedir(experiment) + f"/interactions/{experiment}_index.csv"
    # features_pkl=get_basedir(experiment) + f"/interactions/{experiment}_features.pkl"
    index_csv="index.csv"
    features_pkl="features.pkl"
    
    if not os.path.exists(index_csv):
        logger.error("%s not found", index_csv)
        return None
        
    identities=list(range(1, number_of_animals+1))
    loaders=[FlyHostelLoader(experiment=experiment, identity=identity) for identity in identities]
    interactions_database=pd.read_csv(index_csv)
    features=pd.read_pickle(features_pkl)
    interactions_database, features=annotate_interaction_database_using_behavior_data(
        interactions_database,
        features,
        loaders,
        window_s=window_s
    )
    interactions_database["interaction_duration"]=(interactions_database["last_frame_number"]-interactions_database["first_frame"])/FRAMERATE
    interactions_database["idx"]=interactions_database["id"] + "_" + interactions_database["nn"] + "_" + interactions_database["frame_number"].astype(str)

    for sleep_name in sleep_names:
        interactions_database[f"pre_{sleep_name}_time_before_interaction"]=interactions_database[f"pre_{sleep_name}_bout_in"]*TIME_WINDOW_LENGTH
        interactions_database[f"post_{sleep_name}_time_after_interaction"]=interactions_database[f"post_{sleep_name}_duration"]-interactions_database[f"post_{sleep_name}_bout_in"]*TIME_WINDOW_LENGTH

    interactions_database["label"]=np.nan
    interactions_database.reset_index(inplace=True)

    putative_rejections=interactions_database.loc[
        (interactions_database["experiment"]==experiment) & \
        (interactions_database["pre_longImmobile"]==True)
    ].sort_values(
        "inactive+rejection_max", ascending=False
    )[COLUMNS].drop_duplicates(["id", "frame_number", "nn"])

    putative_rejections.set_index("idx", inplace=True)
    putative_rejections.to_csv("rejections.csv")
    features.to_hdf("features.hdf5", key="features")

    with open("features2.pkl", "wb") as handle:
        pickle.dump(features, handle)

    return interactions_database, features, loaders

def load_rejections(experiment):
    csv_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_rejections.csv"
    )
    index_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_index.csv"
    )
    features_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_features.hdf5"
    )
    features=pd.read_hdf(features_file)
    rejections=pd.read_csv(csv_file)
    return rejections, features
