import datetime
import json
import logging
import os.path
import pickle
import itertools
from functools import lru_cache

from tqdm.auto import tqdm
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from flyhostel.data.pose.constants import get_bodyparts, chunksize
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.distances import add_speed_features, add_interdistance_features
from flyhostel.utils.utils import (
    get_wavelet_downsample,
    restore_cache,
    save_cache
)
from sklearn.metrics import confusion_matrix, roc_curve, auc


OUTPUT_FOLDER="output"

logger=logging.getLogger(__name__)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


NUMBER_OF_SAMPLES={"walk": 30_000, "inactive": 10_000, "groom": 30_000}

def sample_informative_behaviors(pose_annotated_with_wavelets):

    # generate a dataset of wavelets and the ground truth for all behaviors
    ##########################################################################
    inactive_pe=pose_annotated_with_wavelets.loc[pose_annotated_with_wavelets["behavior"]=="inactive+pe"]
    behaviors=np.unique(pose_annotated_with_wavelets["behavior"]).tolist()
    for behav in ["unknown", "inactive+pe"]:
        if behav in behaviors:
            behaviors.pop(behaviors.index(behav))


    dfs=[inactive_pe]
    for behav in behaviors:
        d=pose_annotated_with_wavelets.loc[pose_annotated_with_wavelets["behavior"]==behav].sample(frac=1).reset_index(drop=True)
        samples_available=d.shape[0]
        if behav=="inactive+pe":
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


# def downsample_dataset(labeled_datasets):
#     return labeled_datasets.iloc[::5]

def downsample_dataset(labeled_datasets, n, n_min=0):
    behaviors=labeled_datasets["behavior"].unique()
    out=[]
    for behavior in behaviors:
        labeled_dataset_behav=labeled_datasets.loc[labeled_datasets["behavior"]==behavior]
        n_samples=min(n, labeled_dataset_behav.shape[0])
        if n_samples < n_min:
            continue
        logger.info("N %s = %s", behavior, n_samples)
        out.append(
            labeled_dataset_behav.sample(n=n_samples)
        )
    out=pd.concat(out, axis=0)
    return out


def load_one_animal(experiment, identity, feature_types, pose_key="pose", behavior_key="deg", segregate=True, bodyparts=None, cache=None):

    loader = FlyHostelLoader(experiment=experiment, identity=identity, chunks=range(0, 400))

    wavelet_downsample=get_wavelet_downsample(experiment)
    
    if cache is not None:
        path=os.path.join(cache, experiment + "__" + str(identity).zfill(2) + "_" + "-".join(sorted(feature_types)) + "_" + str(segregate) + "_classifier_data.pkl")
        ret, val=restore_cache(path)
        if ret:
            logger.debug("Restored cache %s", path)
            loader.load_deg_data(identity=identity, ground_truth=True, stride=1, verbose=False)

            if segregate:
                val0=val[0].drop("behavior", axis=1).merge(loader.deg[["frame_number", "behavior"]], on="frame_number")
                val1=val[1].drop("behavior", axis=1).merge(loader.deg[["frame_number", "behavior"]], on="frame_number")
                val=(val0, val1, *val[2:])
            else:
                val0=val[0].drop("behavior", axis=1).merge(loader.deg[["frame_number", "behavior"]], on="frame_number")
                val=(val0, *val[1:])
            return val

    if pose_key=="pose":
        loader.load_data(
            stride=1,
            cache="/flyhostel_data/cache",
            load_behavior=False,
        )
    elif pose_key=="pose_boxcar":
        loader.load_and_process_data(
            stride=1,
            cache="/flyhostel_data/cache",
            filters=None,
            useGPU=0
        )
    out=loader.load_dataset(
        pose=getattr(loader, pose_key).iloc[::wavelet_downsample],
        feature_types=feature_types, deg=getattr(loader, behavior_key), wavelets=None,
        segregate=segregate
    )


    if bodyparts is None:
        bodyparts=get_bodyparts()

    legs=[bp for bp in bodyparts if "L" in bp]
    leg_tips=[leg for leg in legs if "J" not in leg]
    
    if segregate:
        if out is None:
            return None, None, (None, None, None)
        labeled_dataset, unknown_dataset, (frequencies, freq_names, features)=out
        datasets=[labeled_dataset, unknown_dataset]
    else:
        if out is None:
            return None, (None, None, None)
        pose_with_wavelets, (frequencies, freq_names, features)=out
        datasets=[pose_with_wavelets]

    assert features is not None

    out_datasets=[]
    
    for dataset in datasets:
        if dataset.shape[0]>0:
            if feature_types is None or "speed" in feature_types:
                logger.debug("Computing body parts speed in dataset of shape %s", dataset.shape)
                dataset, features=add_speed_features(dataset, features, legs + ["proboscis"])

            if feature_types is None or "distance" in feature_types:
                logger.debug("Computing inter body part distance in dataset of shape %s", dataset.shape)
                dataset, features=add_interdistance_features(dataset, features, leg_tips)
                dataset, features=add_interdistance_features(dataset, features, ["head", "proboscis"])

            dataset["experiment"]=experiment
            dataset["identity"]=int(identity)
            dataset["animal"]=f"{experiment}__{str(identity).zfill(2)}"
            dataset["frame_idx"]=dataset["frame_number"]%chunksize
            dataset["chunk"]=dataset["frame_number"]//chunksize
            
            if behavior_key=="deg":
                dataset=dataset.loc[~dataset["frame_idx"].isin([0, chunksize-1])]
        
        out_datasets.append(dataset)

    features=np.unique(features).tolist()

    for dataset in out_datasets:
        assert all((feat in dataset.columns for feat in features))

    out=(*out_datasets, (frequencies, freq_names, features))

    if cache is not None:
        save_cache(path, out)

    return out



def train_test_split_behavior(data, features, critical_behaviors=None, train_size=0.7, output=None, timestamp=None):
    videos=data[["id", "chunk"]].drop_duplicates().values.tolist()
    if critical_behaviors is None:
        stratify=None
    else:
        stratify=data.groupby(["id", "chunk"]).apply(lambda df: df["behavior"].isin(critical_behaviors).any()).reset_index()[0].values.tolist()

    train_videos, test_videos=train_test_split(videos, train_size=train_size, stratify=stratify)

    test_idx=[]
    train_idx=[]

    for id, chunk in train_videos:
        train_idx.extend(
            np.where((data["id"]==id) & (data["chunk"]==chunk))[0].tolist()
        )

    for id, chunk in test_videos:
        test_idx.extend(
            np.where((data["id"]==id) & (data["chunk"]==chunk))[0].tolist()
        )

    split={"train": train_idx, "test": test_idx}
    if output is not None:
        with open(os.path.join(output, f"{timestamp}_split.json"), "w", encoding="utf-8") as handle:
            json.dump(split, handle)


    logger.debug("Preparing data...")
    training_set=data.iloc[train_idx]#.sort_values(["id", "frame_number"])
    test_set=data.iloc[train_idx]#.sort_values(["id", "frame_number"])
    
    return training_set, test_set


def load_data_for_behavior_model(
        animals, timestamp, pose_key="pose", behavior_key="deg", output=OUTPUT_FOLDER, bodyparts=None,
        n_max=1000,
        n_min=0,
        train_size=0.7, n_jobs=1,
        feature_types=["wavelets"],
        critical_behaviors=None,
        cache=None,
        with_unknown=True,
        segregate=True,
    ):

    logger.info("Output will be saved in %s", output)
    experiments=[animal.split("__")[0] for animal in animals]
    identities=[int(animal.split("__")[1]) for animal in animals]

    datasets=[]
    unknown_datasets=[]

    par_out=joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            load_one_animal
        )(
            experiment, identity, pose_key=pose_key, behavior_key=behavior_key,
            bodyparts=bodyparts, segregate=segregate,
            cache=cache, feature_types=feature_types
        )
        for experiment, identity in zip(experiments, identities)
    )

    for row in par_out:
        if segregate:
            labeled_dataset, unknown_dataset, (frequencies, freq_names, features) = row
            datasets.append(labeled_dataset)
            unknown_datasets.append(unknown_dataset)
        else:
            data, (frequencies, freq_names, features) = row 
            datasets.append(data)

    logger.debug("Concatenating datasets")
    all_labeled_datasets=pd.concat(datasets, axis=0).reset_index(drop=True)
    if with_unknown:
        all_unknown_datasets=pd.concat(unknown_datasets, axis=0).reset_index(drop=True)
    else:
        all_unknown_datasets=None

   
    logger.debug("Downsampling abundant classes")
    train_test_dataset=downsample_dataset(all_labeled_datasets, n=n_max, n_min=n_min)
    logger.debug("Performing train-test split by video")


    training_set, test_set=train_test_split_behavior(
        data=train_test_dataset, features=features, train_size=train_size, output=output,
        critical_behaviors=critical_behaviors, timestamp=timestamp
    )
    
    X_train=training_set[features].values
    y_train=training_set["behavior"].values

    logger.debug("Saving datasets")
    all_labeled_datasets.reset_index().to_feather(os.path.join(output, f"{timestamp}_dataset.feather"))
    if with_unknown:
        all_unknown_datasets.reset_index().to_feather(os.path.join(output, f"{timestamp}_unknown_dataset.feather"))
    train_test_dataset.reset_index().to_feather(os.path.join(output, f"{timestamp}_train_test.feather"))

    with open(os.path.join(output, f"{timestamp}_features.txt"), "w") as handle:
        for feat in features:
            handle.write(f"{feat}\n")

    return (X_train, y_train), (all_labeled_datasets, all_unknown_datasets, train_test_dataset), (freq_names, frequencies, features)


def evaluate_model(timestamp, input=OUTPUT_FOLDER, suffix="ml_classifier", behaviors=None):

    model_path=os.path.join(input, "models", f"{timestamp}_{suffix}.pkl")

    with open(model_path, "rb") as handle:
        model, features=pickle.load(handle)


    dataset=pd.read_feather(os.path.join(input, f"{timestamp}_dataset.feather"))

    with open(os.path.join(input, f"{timestamp}_split.json"), "r") as handle:
        split=json.load(handle)

    train_idx, test_idx = split["train"], split["test"]
    training_set=dataset.iloc[train_idx]
    test_set=dataset.iloc[test_idx]

    X_test=test_set[features].values
    y_test=test_set["behavior"].values

    keep=[]
    if behaviors is not None:
        for i, label in enumerate(y_test):
            if label in behaviors:
                keep.append(i)

        X_test=X_test[keep, :]
        y_test=y_test[keep]
    else:
        behaviors=model.classes_.tolist()


    logger.debug("Transforming labeled dataset of shape %s", training_set.shape)

    y_prob = model.predict_proba(X_test)


    for label in behaviors:
        probs_label=y_prob[:, behaviors.index(label)]
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
