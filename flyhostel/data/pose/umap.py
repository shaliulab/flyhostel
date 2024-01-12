import datetime
import json
import logging
import os.path
import pickle
import time

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from umap import UMAP
from flyhostel.data.pose.ethogram import annotate_bout_duration, annotate_bouts
from flyhostel.data.pose.main import FlyHostelLoader



LTA_DATA=os.environ["LTA_DATA"]
MOTIONMAPPER_DATA=os.environ["MOTIONMAPPER_DATA"]
OUTPUT_FOLDER=os.path.join(MOTIONMAPPER_DATA, "output")
MODELS_FOLDER=os.path.join(MOTIONMAPPER_DATA, "models")

logger=logging.getLogger(__name__)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
STRIDE=5

def remove_bout_ends_from_dataset(dataset, n, fps):
    """
    Remove beginning and end of each bout, to avoid confusion between human label and wavelet defined behavior

    Arguments:

        dataset (pd.DataFrame): contains a column called behavior and is sorted chronologically.
            All rows are equidistant in time. A single animal is present
        n (int): How many points to remove at beginning or end of each bout
        fps (int): Number of points in this dataset that are contained within one second of recording

    Returns:
        dataset (pd.DataFrame): rows at beginning or end of bouts are removed
    """
    dataset=annotate_bouts(dataset, variable="behavior")
    dataset=annotate_bout_duration(dataset, fps=fps)
    dataset = dataset.loc[(dataset["bout_in"] > n) & (dataset["bout_out"] > n)]
    del dataset["bout_in"]
    del dataset["bout_out"]
    return dataset


def train_umap(input="experiments.txt", run_on_unknown=False, output=OUTPUT_FOLDER):
    logger.info("Output will be saved in %s", output)

    loaders={}

    assert os.path.exists(input)
    with open(input, "r", encoding="utf-8") as handle:
        lines=handle.readlines()
        experiments=[line.strip("\n") for line in lines]

    for experiment in experiments:
        loader = FlyHostelLoader(experiment, chunks=range(0, 400))
        loader.load_and_process_data(
            stride=STRIDE,
            cache="/flyhostel_data/cache",
            filters=None,
            useGPU=0
        )
        loaders[experiment]=loader

    datasets=[]
    unknown_datasets=[]

    for experiment in experiments:
        out=loaders[experiment].load_dataset()
        if out is None:
            continue
        labeled_dataset, unknown_dataset, (frequencies, freq_names)=out
        assert freq_names is not None

        labeled_dataset=remove_bout_ends_from_dataset(labeled_dataset, 1, fps=6)
        datasets.append(labeled_dataset)
        unknown_datasets.append(unknown_dataset)

    del loaders
    labeled_dataset=pd.concat(datasets, axis=0)
    labeled_dataset=labeled_dataset.iloc[::5]
    train_idx, test_idx=train_test_split(np.arange(labeled_dataset.shape[0]), train_size=0.7)

    split={"train": train_idx.tolist(), "test": test_idx.tolist()}
    with open(os.path.join(output, f"{timestamp}_split.json"), "w", encoding="utf-8") as handle:
        json.dump(split, handle)


    training_set=labeled_dataset.iloc[train_idx]
    test_set=labeled_dataset.iloc[test_idx]


    # train the UMAP model
    # and use it to project the training and the test set
    model=UMAP()
    umap_set=training_set[freq_names].values
    logger.debug("Fitting UMAP with data of shape %s", umap_set.shape)
    before=time.time()
    model.fit(umap_set)
    after=time.time()
    logger.debug("Done fitting UMAP in %s seconds", round(after-before, 1))
    timestamp=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_path=os.path.join(output, f"{timestamp}_UMAP.pkl")
    
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)

    with open(os.path.join(output, f"{timestamp}_freqs.pkl"), "wb") as handle:
        pickle.dump((frequencies, freq_names), handle)


    logger.debug("Transforming labeled dataset of shape %s", training_set.shape)
    projection=model.transform(training_set[freq_names].values)
    training_set["C_1"]=projection[:,0]
    training_set["C_2"]=projection[:,1]
    test_set["C_1"]=np.nan
    test_set["C_2"]=np.nan
    

    labeled_dataset=pd.concat([training_set, test_set], axis=0).sort_values(["id", "frame_number"])
    labeled_dataset.reset_index().to_feather(os.path.join(output, f"{timestamp}_dataset.feather"))



    logger.debug("Generating visualization")
    fig=px.scatter(
        training_set.loc[training_set["behavior"].isin(["pe_inactive", "feed", "groom", "inactive", "walk"])], x="C_1", y="C_2", color="behavior",
        hover_data=["id", "chunk", "frame_idx", "frame_number", "behavior"],
    )

    px_path=os.path.join(output, f"{timestamp}_UMAP_by_behavior.html")
    logger.debug("Saving to ---> %s", px_path)
    fig.write_html(px_path)
    fig.show()


    if run_on_unknown:
        unknown_dataset=pd.concat(unknown_datasets, axis=0)
        unknown_dataset_subset=unknown_dataset.iloc[::5]

        logger.debug("Transforming unknown dataset of shape %s", unknown_dataset_subset.shape)
        before=time.time()
        unknown_projection=model.transform(unknown_dataset_subset[freq_names].values)
        after=time.time()
        logger.debug("Done transforming unknown dataset of shape %s in %s seconds", unknown_dataset_subset.shape, round(after-before, 1))
        unknown_dataset_subset["C_1"]=unknown_projection[:,0]
        unknown_dataset_subset["C_2"]=unknown_projection[:,1]
        unknown_dataset_subset.reset_index().to_feather(os.path.join(output, "unknown_set.feather"))


def generate_umap_dataset(pose_annotated, groupby="behavior", min_per_group=1000):
    
    behavior_target_count="pe_inactive"
    target_count=(pose_annotated[groupby]==behavior_target_count).sum()
    target_count=max(min_per_group, target_count)
    logger.debug("Keeping %s points per %s", target_count, groupby)
    pose_annotated_shuf=pose_annotated.sample(frac=1).reset_index(drop=True)

    pose_annotated_shuf = pose_annotated_shuf.groupby(groupby).apply(lambda x: x.iloc[:target_count]).reset_index(drop=True)
    return pose_annotated_shuf
