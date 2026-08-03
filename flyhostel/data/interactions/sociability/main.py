import os
import logging
from pathlib import Path

import joblib
import pandas as pd
from .sociability import process_experiment
from flyhostel.utils import load_experiments as load_experiments_
N_CLUSTERS=20
WINDOW_S=1
MIN_TIME=6*3600
MAX_TIME=30*360

logger=logging.getLogger(__name__)

def load_experiments(*args, **kwargs):
    logger.warning("Please replace flyhostel.data.interactions.sociability.load_experiments with flyhostel.utils.load_experiments")
    return load_experiments_(*args, **kwargs)


def main():

    outputs_dir = Path("./outputs/")
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(exist_ok=True, parents=True)

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
