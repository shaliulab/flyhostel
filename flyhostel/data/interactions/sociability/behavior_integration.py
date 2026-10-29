"""
Last steps of the interactions-from-centroids pipeline: annotate the interactions found in an experiment
(index.csv and features.pkl, made by sociability.process_experiment) with

1. the behavior of the flies (integrate_experiment_behavior): rejection probabilities and centroid speed around the
   interaction. Does not depend on sleep. Saves behavior.feather and features.hdf5

2. the immobility / sleep state of the flies before and after the interaction (annotate_experiment_sleep).
   The only step that depends on sleep, which comes from the sleep files of the flies (flyhostel.data.sleep), so
   a new sleep definition only needs this step to be run again. Saves database.feather, rejections.csv and rejected.csv
"""
import logging
import os.path
import itertools
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from flyhostel.data.sleep import load_sleep_file, immobility_bouts
from flyhostel.data.pose.ethogram.utils import annotate_bouts, annotate_bout_duration
from flyhostel.utils import get_basedir
from flyhostel.data.pose.main import FlyHostelLoader

SLEEP_PERIODS=[(10, "longImmobile"), (600, "longAsleep"), (300, "asleep")]

WINDOW_S=1
MIN_TIME=6*3600
MAX_TIME=30*3600
TIME_WINDOW_LENGTH=1
SLEEP_FRAMERATE=1/TIME_WINDOW_LENGTH

logger=logging.getLogger(__name__)


def behavior_database_path(experiment):
    """Interactions of an experiment annotated with behavior, but not with sleep (integrate_experiment_behavior)"""
    return os.path.join(get_basedir(experiment), "interactions", f"{experiment}_behavior.feather")


def load_sleep_data(loader, sleep_periods=[(300, "asleep")]):
    """
    Annotate inactivity duration where the minimum time inactive is variable,
    from the immobility (windowed_var) of the sleep file of the fly

    Arguments:
        loader (FlyHostelLoader)
        sleep_periods (list): Collection of tuples of length 2, containing a min_time_immobile (int) and a name for the inactivity (str)

    Returns
        loader

    loader.sleep has been populated with columns foo, foo_duration, foo_bout_in and immobile,
    where foo is each of the inactivity period names
    """
    animal=f"{loader.experiment}__{str(loader.identity).zfill(2)}"
    sleep=load_sleep_file(loader.basedir, animal).sort_values(["id", "t"]).reset_index(drop=True)
    sleep=sleep.rename({"windowed_var": "immobile"}, axis=1)
    for min_time_immobile, period in sleep_periods:
        sleep[period]=immobility_bouts(sleep.rename({"immobile": "windowed_var"}, axis=1), min_time_immobile)
        sleep=annotate_bout_duration(annotate_bouts(sleep, variable=period), fps=SLEEP_FRAMERATE).reset_index(drop=True).rename({
            "bout_in": f"{period}_bout_in",
            "bout_out": f"{period}_bout_out",
            "duration": f"{period}_duration"
        }, axis=1)
    loader.sleep=sleep.drop(["t", "inactive_rule"], axis=1)
    return loader


def immobility_annotation(loader, interactions_database, sleep_columns=None, rename_dict=None):
    """
    Annotate interactions database with immobility states.
    Keeps every interaction, in the same order (NaN where the fly has no sleep annotation)

    t_ref: Time representative of the interaction
    t: Time at which contact is closest
    """
    interactions_database["t_round"]=interactions_database["t_ref"]//1
    # time until the next interaction (NaN for the last one)
    interactions_database["t_till_next"]=interactions_database["t"].diff().shift(-1)
    assert "t_round" not in sleep_columns
    all_columns=["t_round"] + sleep_columns
    sleep=loader.sleep[all_columns].copy()

    if rename_dict is not None:
        sleep.rename(rename_dict, axis=1, inplace=True)

    interactions_database=interactions_database.merge(sleep, on="t_round", how="left")
    interactions_database["t_round"]=interactions_database["t"]//1
    return interactions_database


def immobility_annotations(loader, interactions_database, sleep_names):
    """
    Annotate immobility state before and after the interaction start and end respectively

    Calls immobility_annotation
    """

    # Find immobility state before (PRE) the interaction
    interactions_database["t_ref"]=interactions_database["first_t"]
    sleep_columns=sleep_names + list(itertools.chain(*[[f"{feat}_bout_in", f"{feat}_bout_out"] for feat in sleep_names])) #+ [f"{feat}_duration" for feat in sleep_names]
    rename_dict={col: f"pre_{col}" for col in sleep_columns}
    interactions_database=immobility_annotation(
        loader, interactions_database,
        sleep_columns=sleep_columns,
        rename_dict=rename_dict
    )
    del interactions_database["t_ref"]

    # Find immobility state after (POST) the interaction
    interactions_database["t_ref"]=interactions_database["last_t"]
    rename_dict={col: f"post_{col}" for col in sleep_columns}
    interactions_database=immobility_annotation(
        loader, interactions_database,
        sleep_columns=sleep_columns,
        rename_dict=rename_dict
    )
    del interactions_database["t_ref"]

    for sleep_name in sleep_names:
        interactions_database[f"pre_{sleep_name}_duration"]=interactions_database[f"pre_{sleep_name}_bout_in"]/SLEEP_FRAMERATE
        interactions_database[f"post_{sleep_name}_duration"]=interactions_database[f"post_{sleep_name}_bout_out"]/SLEEP_FRAMERATE

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


def compute_behavior_features(interactions_database, behavior, window_s, framerate):
    """
    Compute numerical descriptors of each interaction as a function of the behaviors produced during the interaction
    Interactions without behavior at the start or end of the window are dropped

    Descriptors:
        inactive+rejection_max
        inactive+rejection_mean
        centroid_speed
    """
    interactions_database["frame_number_start"]=np.ceil(interactions_database["frame_number"]-window_s/2*framerate)
    interactions_database["frame_number_end"]=np.floor(interactions_database["frame_number"]+window_s/2*framerate)

    interactions_database=annotate_behavior_database_id(interactions_database, behavior)
    if "inactive+rejection" in behavior.columns:
        probs_all=behavior["inactive+rejection"].values
    else:
        probs_all=np.zeros(behavior.shape[0])
    speed_all=behavior["centroid_speed"].values

    stats=[]
    prob_features=[]
    intervals=zip(interactions_database["row_id_start"], interactions_database["row_id_end"])
    for start, end in tqdm(intervals, total=interactions_database.shape[0]):
        probs=probs_all[start:end]
        stats.append([
            probs.max(),
            np.round(probs.mean(), 3),
            np.round(speed_all[start:end].sum(), 3)
        ])
        prob_features.append(probs.tolist())

    # DataFrame and not from_records, which fails when the fly has no interactions
    stats=pd.DataFrame(
        stats,
        columns=[
            "inactive+rejection_max",
            "inactive+rejection_mean",
            "centroid_speed"
        ],
        index=interactions_database.index
    )
    interactions_database=pd.concat([interactions_database, stats], axis=1)
    return interactions_database, prob_features


def annotate_interaction_database_using_behavior_data(interactions_database, features, loaders, window_s=10):
    """
    Add to the features associated to an interaction database
    the features computed by compute_behavior_features
    Add to the interaction database single-number descriptors of these features

    Returns the interactions with behavior, and their features (same rows, same order)
    """

    assert len(loaders)>0
    assert interactions_database.shape[0]==features.shape[0], "Interaction index is not aligned to features database"
    # row of every interaction in features, so they stay aligned when interactions are dropped
    interactions_database=interactions_database.copy()
    interactions_database["features_row"]=np.arange(interactions_database.shape[0])

    interactions_database_l=[]
    probs_l=[]

    experiment=loaders[0].experiment

    for loader in tqdm(loaders, desc=f"Annotating {experiment}"):
        focal=interactions_database.loc[interactions_database["id"]==loader.ids[0]].sort_values("frame_number")
        loader.load_behavior_data(min_time=MIN_TIME, max_time=MAX_TIME)
        focal_database, prob_features=compute_behavior_features(focal, loader.behavior, window_s, loader.framerate)
        interactions_database_l.append(focal_database)
        probs_l.extend(prob_features)

    framerate=loaders[0].framerate

    interactions_database=pd.concat(interactions_database_l, axis=0).reset_index(drop=True)
    prob_features=pd.DataFrame.from_records(probs_l)
    features=features.iloc[interactions_database["features_row"].values].reset_index(drop=True)
    del interactions_database["features_row"]

    assert interactions_database.shape[0]==prob_features.shape[0]==features.shape[0]
    n_steps=prob_features.shape[1]

    prob_features.columns=pd.MultiIndex.from_arrays([
            ["probability",]*n_steps,
            np.arange(n_steps) - (window_s/2)*framerate,
            ["1",]*n_steps
        ],
        names=["feature", "position", "timepoint"]
    )

    features_all=pd.concat([
        features,
        prob_features
    ], axis=1)
    assert features_all.shape[0]==interactions_database.shape[0]

    interactions_database["interaction_duration"]=(interactions_database["last_frame_number"]-interactions_database["first_frame"])/framerate
    interactions_database["idx"]=interactions_database["id"] + "_" + interactions_database["nn"] + "_" + interactions_database["frame_number"].astype(str)
    interactions_database["label"]=np.nan
    # index: row of the interaction, a column of the rejections (COLUMNS)
    interactions_database=interactions_database.reset_index()
    return interactions_database, features_all


def annotate_sleep_states(interactions_database, loaders, sleep_periods=SLEEP_PERIODS):
    """
    Annotate the immobility / sleep state of the focal fly before (pre_) and after (post_) every interaction,
    from the sleep files of the flies. The only use of sleep in the interactions pipeline

    Returns the interactions in the same order
    """
    interactions_database=interactions_database.copy()
    interactions_database["order__"]=np.arange(interactions_database.shape[0])
    annotated=[]
    for loader in tqdm(loaders, desc=f"Annotating sleep of {loaders[0].experiment}"):
        focal=interactions_database.loc[interactions_database["id"]==loader.ids[0]].sort_values("frame_number")
        if focal.shape[0]==0:
            continue
        load_sleep_data(loader, sleep_periods=sleep_periods)
        annotated.append(immobility_annotations(loader, focal, sleep_names=[f[1] for f in sleep_periods]))
    interactions_database=pd.concat(annotated, axis=0).sort_values("order__").reset_index(drop=True)
    del interactions_database["order__"]
    return interactions_database


COLUMNS=[
    "idx", "index", "label", "first_frame", "frame_number", "last_frame_number", "t", "centroid_speed", "inactive+rejection_max", "inactive+rejection_mean",
    "pre_longImmobile", "pre_longImmobile_duration", "post_longImmobile", "post_longImmobile_duration", "interaction_duration", "id", "nn",
]

def putative_rejections(interactions_database, experiment):
    """
    Interactions where the focal fly was immobile before (rejections), and the same interactions seen from the partner (rejected)
    """
    putative_rejections=interactions_database.loc[
        (interactions_database["experiment"]==experiment) & \
        (interactions_database["pre_longImmobile"]==True)
    ].sort_values(
        "inactive+rejection_max", ascending=False
    )[COLUMNS].drop_duplicates(["id", "frame_number", "nn"])

    putative_rejected=interactions_database.merge(
        putative_rejections[["nn", "frame_number"]].rename({"nn": "id"}, axis=1),
        on=["id", "frame_number"], how="inner"
    ).sort_values(
        "inactive+rejection_max", ascending=False
    )[COLUMNS].drop_duplicates(["id", "frame_number", "nn"])
    return putative_rejections.set_index("idx"), putative_rejected.set_index("idx")


def init_loaders(experiment, number_of_animals):
    identities=list(range(1, number_of_animals+1))
    return [FlyHostelLoader(experiment=experiment, identity=identity) for identity in identities]


def integrate_experiment_behavior(experiment, number_of_animals, window_s=10, index_csv="index.csv", features_pkl="features.pkl"):
    """
    Step 1 (no sleep): annotate the interactions of an experiment with the behavior of the flies.
    Saves behavior.feather and features.hdf5 in the working directory
    """
    if not os.path.exists(index_csv):
        logger.error("%s not found", index_csv)
        return None

    loaders=init_loaders(experiment, number_of_animals)
    interactions_database=pd.read_csv(index_csv)
    features=pd.read_pickle(features_pkl)
    interactions_database, features=annotate_interaction_database_using_behavior_data(
        interactions_database, features, loaders, window_s=window_s
    )
    interactions_database.reset_index(drop=True).to_feather("behavior.feather")
    features.to_hdf("features.hdf5", key="features")
    return interactions_database, features


def annotate_experiment_sleep(experiment, number_of_animals, behavior_feather="behavior.feather"):
    """
    Step 2 (sleep): annotate the interactions annotated with behavior (integrate_experiment_behavior)
    with the sleep state of the flies. Saves database.feather, rejections.csv and rejected.csv in the working directory
    """
    loaders=init_loaders(experiment, number_of_animals)
    interactions_database=annotate_sleep_states(pd.read_feather(behavior_feather), loaders)
    interactions_database.reset_index(drop=True).to_feather("database.feather")
    rejections, rejected=putative_rejections(interactions_database, experiment)
    rejections.to_csv("rejections.csv")
    rejected.to_csv("rejected.csv")
    return interactions_database


def process_experiment(experiment, number_of_animals, window_s=10):
    """
    Both steps: annotate the interactions of an experiment with behavior and sleep
    """
    if integrate_experiment_behavior(experiment, number_of_animals, window_s=window_s) is None:
        return None
    return annotate_experiment_sleep(experiment, number_of_animals)


def load_rejections(experiment):
    csv_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_rejections.csv"
    )
    features_file=os.path.join(
        get_basedir(experiment), "interactions", f"{experiment}_features.hdf5"
    )
    features=pd.read_hdf(features_file)
    rejections=pd.read_csv(csv_file)
    return rejections, features
