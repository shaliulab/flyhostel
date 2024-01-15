import datetime
import json
import logging
import os.path
import pickle
import time

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from flyhostel.data.pose.ethogram import annotate_bout_duration, annotate_bouts
from flyhostel.data.pose.main import FlyHostelLoader
from motionmapperpy import setRunParameters
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc


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


def train_rf(input="experiments.txt", run_on_unknown=False, output=OUTPUT_FOLDER):
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

    features=freq_names

    # training_set_clipped=remove_bout_ends_from_dataset(training_set, n_points=3, fps=6)

    X_train=training_set[features].values
    y_train=training_set["behavior"].values
    X_test=test_set[features].values
    y_test=test_set["behavior"].values

    # Create a Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)


    logger.debug("Fitting RF with data of shape %s", X_train.shape)
    before=time.time()
    rf_model.fit(X_train, y_train)
    after=time.time()
    logger.debug("Done fitting RF in %s seconds", round(after-before, 1))
    model_path=os.path.join(output, "models", f"{timestamp}_rf.pkl")
    
    with open(model_path, "wb") as handle:
        pickle.dump((rf_model, features), handle)

    with open(os.path.join(output, f"{timestamp}_freqs.pkl"), "wb") as handle:
        pickle.dump((frequencies, freq_names), handle)

    labeled_datasets.reset_index().to_feather(os.path.join(output, f"{timestamp}_dataset.feather"))


def evaluate_model(timestamp, input=OUTPUT_FOLDER):

    model_path=os.path.join(input, f"{timestamp}_rf.pkl")

    with open(model_path, "rb") as handle:
        rf_model=pickle.load(handle)


    dataset=pd.read_feather(os.path.join(input, f"{timestamp}_dataset.feather"))

    with open(os.path.join(input, f"{timestamp}_split.json"), "r") as handle:
        split=json.load(handle)
    with open(os.path.join(input, f"{timestamp}_freqs.pkl"), "rb") as handle:
        freqs, freq_names=pickle.load(handle)
        
    train_idx, test_idx = split["train"], split["test"]
    training_set=dataset.iloc[train_idx]
    test_set=dataset.iloc[test_idx]


    X_train=training_set[freq_names].values
    y_train=training_set["behavior"].values
    X_test=test_set[freq_names].values
    y_test=test_set["behavior"].values

    logger.debug("Transforming labeled dataset of shape %s", training_set.shape)

    y_prob = rf_model.predict_proba(X_test)

    behaviors=rf_model.classes_

    for label in behaviors:
        probs_label=y_prob[:, behaviors.tolist().index(label)]
        fpr, tpr, thresholds=roc_curve(y_test, probs_label, pos_label=label)
        # Calculate the AUC (Area under the ROC Curve)
        roc_auc = auc(fpr, tpr)
        
        # Plot the ROC Curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title(f'Behavior {label} ROC')
        plt.legend(loc="lower right")
        plt.show()

        path=os.path.join(input, "roc", f"{timestamp}_{label}_ROC.png")
        logger.debug("Saving ---> %s", path)
        plt.savefig(path)
