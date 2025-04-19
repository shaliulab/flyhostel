import os
from pathlib import Path

import joblib
import pandas as pd
from flyhostel.data.pose.constants import framerate as FRAMERATE
from .sociability import process_experiment

ANIMALS_CSV="/home/vibflysleep/opt/vsc-scripts/nextflow/pipelines/behavior_prediction/animals.csv"
GROUPS_CSV="/home/vibflysleep/opt/vsc-scripts/nextflow/pipelines/interaction_detection/fly_groups.csv"

N_CLUSTERS=20
WINDOW_S=1
MIN_TIME=6*3600
MAX_TIME=30*360


def load_experiments(number_of_animals=6):
    """
    Load experiment name for all experiments
    whose behavior and interaction pipelines are complete
    """
    metadata_beh=pd.read_csv(ANIMALS_CSV, header=None)
    metadata_beh.columns=["experiment", "basedir", "identity", "date_completed", "status", "select"]
    metadata_beh["number_of_animals"]=metadata_beh["basedir"].str.slice(34, 35).astype(int)
    metadata_inters=pd.read_csv(GROUPS_CSV, header=None)
    metadata_inters.columns=["basedir", "experiment", "number_of_animals", "status", "select"]
    metadata_inters=metadata_inters.loc[metadata_inters["select"]=="SELECT"]
    if number_of_animals>1:
        metadata=metadata_inters[["experiment"]].merge(metadata_beh, on="experiment", how="inner")
    else:
        metadata=metadata_beh
    metadata.loc[(metadata["number_of_animals"]==1), "status"]="SELECT"
    metadata=metadata.loc[~(metadata["select"].isna())]
    metadata=metadata.loc[metadata["select"]=="SELECT"]
    experiments=metadata.loc[
        (metadata["number_of_animals"]==number_of_animals),
        "experiment"
    ].unique().tolist()

    return experiments


def main():

    outputs_dir = Path("./outputs/")
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)
    window_f=WINDOW_S*FRAMERATE

    number_of_animals=6
    experiments=load_experiments(number_of_animals)
    identities=[1, 2, 3, 4, 5, 6]
    # identities=list(range(1, number_of_animals+1))

    timepoints=("first_frame", "frame_number", "last_frame_number")

    process_all_experiments(
        experiments, identities, window_s=WINDOW_S, n_jobs=1,
        min_time=MIN_TIME, max_time=MAX_TIME, timepoints=timepoints
    )


def process_all_experiments(
        experiments, identities, window_s,
        n_jobs=1,
        max_workers=1,
        min_time=None, max_time=None,
        timepoints=("frame_number", )
    ):
    all_features=[]
    indices=[]
    n_jobs=min(len(experiments), n_jobs)
    if n_jobs==1:
        pass
    else:
        max_workers=1

    out=joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            process_experiment
        )(
            experiment, identities,
            min_time=min_time, max_time=max_time,
            window_s=window_s, cache=False,
            timepoints=timepoints,
            max_workers=max_workers
        )
        for experiment in experiments
    )
    loaders=[]
    for experiment, (features, index), experiment_loaders in out:
        if features is not None:
            index=index.loc[index["keep"]]
            assert features.shape[0]==index.shape[0], f"{features.shape[0]} != {index.shape[0]}"
            index["experiment"]=experiment
            indices.append(index)
            all_features.append(features)
            loaders.extend(experiment_loaders)

    features=pd.concat(all_features, axis=0).reset_index(drop=True)
    index=pd.concat(indices, axis=0).reset_index(drop=True)
    assert features.shape[0]==index.shape[0], f"{features.shape[0]} != {index.shape[0]}"


    return index, features, loaders
