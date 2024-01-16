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
from motionmapperpy import setRunParameters



LTA_DATA=os.environ["LTA_DATA"]
MOTIONMAPPER_DATA=os.environ["MOTIONMAPPER_DATA"]
OUTPUT_FOLDER=os.path.join(MOTIONMAPPER_DATA, "output")
MODELS_FOLDER=os.path.join(MOTIONMAPPER_DATA, "models")

logger=logging.getLogger(__name__)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
STRIDE=setRunParameters().wavelet_downsample

def remove_bout_ends_from_dataset(dataset, n_points, fps):
    """
    Set beginning and end of each bout to background, to avoid confusion between human label and wavelet defined behavior

    If the labeling strategy has more temporal resolution than the algorithm used to infer them
    there can be some artifacts where the signal inferred from a previous or future bout (very close temporally) spills over into the present bout
    This means the ground truth and inference are less likely to agree at transitions, and such frames should be labeled as such
    by setting the behavior to background (aka transition)
     

    Arguments:

        dataset (pd.DataFrame): contains a column called behavior and is sorted chronologically.
            All rows are equidistant in time. A single animal is present.
        n_points (int): How many points to remove at beginning AND end of each bout.
        fps (int): Number of points in this dataset that are contained within one second of recording.

    Returns:
        dataset (pd.DataFrame): rows at beginning or end of bouts are removed
    """
    dataset=annotate_bouts(dataset, variable="behavior")
    dataset=annotate_bout_duration(dataset, fps=fps)
    short_behaviors=["pe_inactive"]
    dataset.loc[((dataset["bout_in"] <= n_points) & (dataset["bout_out"] <= n_points)) | np.bitwise_not(dataset["behavior"].isin(short_behaviors)), "behavior"]="background"
    del dataset["bout_in"]
    del dataset["bout_out"]
    return dataset


NUMBER_OF_SAMPLES={"walk": 30_000, "inactive": 10_000, "groom": 30_000}

def sample_informative_behaviors(pose_annotated_with_wavelets):

    # generate a dataset of wavelets and the ground truth for all behaviors
    ##########################################################################
    pe_inactive=pose_annotated_with_wavelets.loc[pose_annotated_with_wavelets["behavior"]=="pe_inactive"]
    behaviors=np.unique(pose_annotated_with_wavelets["behavior"]).tolist()
    for behav in ["unknown", "pe_inactive"]:
        if behav in behaviors:
            behaviors.pop(behaviors.index(behav))


    dfs=[pe_inactive]
    for behav in behaviors:
        d=pose_annotated_with_wavelets.loc[pose_annotated_with_wavelets["behavior"]==behav].sample(frac=1).reset_index(drop=True)
        samples_available=d.shape[0]
        if behav=="pe_inactive":
            n_max=samples_available
        else:
            max_seconds=60
            n_max=6*max_seconds
        number_of_samples=NUMBER_OF_SAMPLES.get(behav, n_max)
        dfs.append(
            d.iloc[:number_of_samples]
        )

    labeled_dataset = pd.concat(dfs, axis=0)
    return labeled_dataset


def train_umap(input="experiments.txt", run_on_unknown=False, output=OUTPUT_FOLDER):
    logger.info("Output will be saved in %s", output)
    timestamp=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    assert os.path.exists(input)
    with open(input, "r", encoding="utf-8") as handle:
        lines=handle.readlines()
        experiments=[line.strip("\n") for line in lines]

    datasets=[]
    unknown_datasets=[]


    for experiment in experiments:
        loader = FlyHostelLoader(experiment, chunks=range(0, 400))
        load=False
        for datasetname in loader.datasetnames:
            if not os.path.exists(loader.get_matfile(datasetname)):
                logger.warning("Skipping %s", datasetname)
                continue
            else:
                load=True

        if not load:
            logger.error("Skipping %s", experiment)
            continue


        loader.load_and_process_data(
            stride=STRIDE,
            cache="/flyhostel_data/cache",
            filters=None,
            useGPU=0
        )
      
        out=loader.load_dataset()
        if out is None:
            continue
        labeled_dataset, unknown_dataset, (frequencies, freq_names)=out
        assert freq_names is not None

        datasets.append(labeled_dataset)
        unknown_datasets.append(unknown_dataset)
    
    del loader

    all_labeled_datasets=pd.concat(datasets, axis=0).sort_values(["id", "frame_number"]).reset_index(drop=True)
    
    # labeled_datasets=sample_informative_behaviors(all_labeled_datasets)
    labeled_datasets=all_labeled_datasets
    labeled_datasets=labeled_datasets.iloc[::5]
    
    labeled_datasets=annotate_bouts(labeled_datasets, variable="behavior")
    bouts=labeled_datasets["bout_count"].unique()

    train_bout, test_bout=train_test_split(bouts, train_size=0.7)
    train_idx=sorted(np.where(labeled_datasets["bout_count"].isin(train_bout))[0].tolist())
    test_idx=sorted(np.where(labeled_datasets["bout_count"].isin(test_bout))[0].tolist())
    
    split={"train": train_idx, "test": test_idx}
    with open(os.path.join(output, f"{timestamp}_split.json"), "w", encoding="utf-8") as handle:
        json.dump(split, handle)


    training_set=labeled_datasets.iloc[train_idx].sort_values(["id", "frame_number"])
    test_set=labeled_datasets.iloc[test_idx].sort_values(["id", "frame_number"])


    training_set_clipped=remove_bout_ends_from_dataset(training_set, n_points=3, fps=6)


    # train the UMAP model
    # and use it to project the training and the test set
    model=UMAP()
    umap_set=training_set_clipped[freq_names].values
    logger.debug("Fitting UMAP with data of shape %s", umap_set.shape)
    before=time.time()
    model.fit(umap_set)
    after=time.time()
    logger.debug("Done fitting UMAP in %s seconds", round(after-before, 1))
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
    

    labeled_datasets=pd.concat([training_set, test_set], axis=0).sort_values(["id", "frame_number"])
    labeled_datasets.reset_index().to_feather(os.path.join(output, f"{timestamp}_dataset.feather"))
    all_labeled_datasets.reset_index().to_feather(os.path.join(output, f"{timestamp}_all_dataset.feather"))

    color_mapping = {
        "pe_inactive": "yellow",
        "feed": "orange",
        "groom": "green",
        "inactive": "blue",
        "walk": "red",
    }

    logger.debug("Generating visualization")
    fig=px.scatter(
        training_set.loc[training_set["behavior"].isin(["pe_inactive", "feed", "groom", "inactive", "walk"])], x="C_1", y="C_2", color="behavior",
        hover_data=["id", "chunk", "frame_idx", "frame_number", "behavior"],
        color_discrete_map=color_mapping,
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
