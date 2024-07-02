import shutil
import os.path
import traceback
import itertools
import logging
import pickle


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from xgboost import XGBClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from flyhostel.data.pose.ethogram.inference import inference_ as inference
from flyhostel.data.pose.ethogram.plot import save_confusion_matrix
from flyhostel.data.pose.ethogram.utils import (
    annotate_active_state,
)

MODELS={"RandomForestClassifier": RandomForestClassifier, "XGBClassifier": XGBClassifier, "ExplainableBoostingClassifier": ExplainableBoostingClassifier}

from sklearn.metrics import ConfusionMatrixDisplay, balanced_accuracy_score, top_k_accuracy_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

from motionmapperpy.motionmapper import generate_frequencies
from flyhostel.data.pose.constants import bodyparts as BODYPARTS
from flyhostel.data.pose.constants import bodyparts_xy as BODYPARTS_XY
from flyhostel.data.pose.constants import WAVELET_DOWNSAMPLE, MOTIONMAPPER_PARAMS
from flyhostel.data.pose.constants import DEFAULT_FILTERS
DEFAULT_FILTERS="-".join(DEFAULT_FILTERS)
from flyhostel.data.deg import LABELS
from flyhostel.data.pose.ethogram.utils import load_train_test_split
from flyhostel.data.pose.ethogram.loader import load_animals, DISTANCE_FEATURES_PAIRS, document_provenance, validate_animals_data



def get_frequencies(params):
    return np.round(generate_frequencies(params.minF, params.maxF, params.numPeriods), 4)


def get_frequencies_bps(params=MOTIONMAPPER_PARAMS, freqs=None, bodyparts_xy=BODYPARTS_XY):
    if freqs is None:
        freqs=get_frequencies(params)

    freq_names=[f"{e0}_{e1}" for e0, e1 in itertools.product(bodyparts_xy, freqs) if not e0.startswith("thorax") and not e0.startswith("head")]
    return freq_names


def get_features(
        freqs=None, bodyparts_xy=BODYPARTS_XY,
        distance_features_pairs=DISTANCE_FEATURES_PAIRS,
        probabilities=None,
        speed_features=None,
        landmarks=None,
        ):
    freq_names=get_frequencies_bps(freqs=freqs, bodyparts_xy=bodyparts_xy)
    features=freq_names + [f"{p0}_{p1}_distance" for p0, p1 in distance_features_pairs]
    if speed_features is not None:
        features+=speed_features    
    if probabilities is not None:
        features+=probabilities
    if landmarks is not None:
        features+=landmarks
    return features

logger=logging.getLogger(__name__)



def get_sample_weights(y):
    # Compute class weights
    classes=np.unique(y)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    weight_dict = {classes[i]: class_weights[i] for i in range(len(class_weights))}
    sample_weights = np.array([weight_dict[cls] for cls in y])
    return sample_weights


def build_eval_params(eval_params):
    eval_params["downsample_train"]=eval_params.get("downsample_train", 1)
    eval_params["probabilities"]=eval_params.get("probabilities", None)
    eval_params["landmarks"]=eval_params.get("landmarks", None)
    eval_params["speed_features"]=eval_params.get("speed_features", None)
    eval_params["downsample_test"]=eval_params.get("downsample_test", 1)
    eval_params["bodyparts"]=eval_params.get("bodyparts", BODYPARTS)
    eval_params["freqs"]=eval_params.get("freqs", get_frequencies(MOTIONMAPPER_PARAMS))
    eval_params["distance_features_pairs"]=eval_params.get("distance_features_pairs", DISTANCE_FEATURES_PAIRS)
    return eval_params


def train(
        train_set_animals, test_set_animals, norm=True, output_folder=".", label="behavior", n_jobs_load=1, model_arch="RandomForestClassifier",
        eval_params={}, cache=None, refresh_cache=True, on_fail="raise", **kwargs):
    f"""
    Train a multiclass classifier to predict a behavior using the WT features and the head-proboscis distance

    Arguments:
       train_set_animals (list): Animals in experiment__identity format used for training
       test_set_animals (list): Animals in experiment__identity format used for test
       norm (bool): If True, input to the model is z-values computed based on standard deviation of train set features
       output_folder (str): Where to save outputs
       label (str): If behavior, the original behavior is the target, if label, several behaviors are grouped into micromovement
       n_jobs_load (int): How many animals can be loaded in parallel?
       model_arch (str): One of the models in {MODELS}. The model objects must implement fit(X, y), predict_proba(X) and the attribute classes_

       eval_params (dict): Parameters used to evaluate the efficiency of this routine (not used in the final run)
           * downsample_train (int): How many times less data should be used in the train set. 1 means all, 2 means half and so on
           * downsample_test (int): How many times less data should be used in the test set. 1 means all, 2 means half and so on
           * distance_features_pairs (list): List of body part pairs whose distance in every frame will be used as feature for the model
           * bodyparts (list): Bodyparts whose wavelets will be used
           * freqs (list): Frequencies whose wavelet will be used
       
       **kwargs: extra arguments to the model

    Returns:
       Saves to output folder
            1. test_confusion_matrix.png
               visualization of the specificity and sensitivity of the model in the test set
            2. test_predictions.feather
               test set data annotated by the model
            3. model.pkl
                contains the tuple (model, scaler, features)
            4. split.yaml
                the train-test split used during this training run
            5. metrics.yaml
                some of the metrics mentioned here https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
                achieved by the model on the test set
    """

    if model_arch not in MODELS:
        raise ValueError(f"Please pass a model arch which is one of {MODELS.keys()}")


    os.makedirs(output_folder)
    split={"train": train_set_animals, "test": test_set_animals}
    with open(os.path.join(output_folder, "split.yaml"), "w") as handle:
        yaml.dump(split, handle, sort_keys=False)


    provenance=document_provenance()
    eval_params=build_eval_params(eval_params)
    bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in eval_params["bodyparts"]]))
    provenance.update(eval_params)

    provenance["freqs"]=[float(np.round(freq, 4)) for freq in provenance["freqs"] if freq is not None]
    
    with open(os.path.join(output_folder, "provenance.yaml"), "w") as handle:
        yaml.safe_dump(provenance, handle)

    features=get_features(
        freqs=eval_params["freqs"],
        bodyparts_xy=bodyparts_xy,
        distance_features_pairs=eval_params["distance_features_pairs"],
        speed_features=eval_params["speed_features"],
        probabilities=eval_params["probabilities"],
        landmarks=eval_params["landmarks"],

    )

    validate_animals_data(train_set_animals+test_set_animals)
    load_scores_data=True
    train_data=load_animals(train_set_animals, load_deg_data=True, load_scores_data=load_scores_data, cache=cache, refresh_cache=refresh_cache, n_jobs=n_jobs_load, filters=DEFAULT_FILTERS, on_fail=on_fail)
    test_data=load_animals(test_set_animals, load_deg_data=True, load_scores_data=load_scores_data, cache=cache, refresh_cache=refresh_cache, n_jobs=n_jobs_load, filters=DEFAULT_FILTERS, on_fail=on_fail)
    train_data=downsample_dataset(train_data, eval_params["downsample_train"])
    test_data=downsample_dataset(test_data, eval_params["downsample_test"])


    # check all behaviors in the training set are also present in the test set
    trained_behaviors=train_data[label].unique()
    for behavior in test_data[label].unique():
        assert behavior in trained_behaviors, f"{behavior} is present in test but not train set"
        

    # check all frequencies have been computed and present in the train data
    missing_freqs=[feat for feat in features if not feat in train_data.columns]
    
    if len(missing_freqs)>0:
        logger.warning("Some of the frequencies produced by LTA are not present in the data")
        logger.warning("Missing frequencies %s", missing_freqs)
        logger.warning("Available frequecies: %s", train_data.columns)
        

    # Downsample so that least abundant classes are not so underrepresented
    n_samples_inactive_pe=train_data.loc[train_data["behavior"]=="inactive+pe"].shape[0]
    train_data_downsampled=train_data.groupby("behavior").apply(lambda df: df.sample(n=min(n_samples_inactive_pe, df.shape[0]))).reset_index(drop=True)
    assert train_data_downsampled["head_proboscis_distance"].isna().sum()==0

    train_data_downsampled.to_feather(os.path.join(output_folder, "train_set.feather"))

    y_train=train_data_downsampled[label].values

    # Compute class weights
    x_train=train_data_downsampled[features].values

    missing_data=np.isnan(x_train).sum(axis=1)

    # check there is no missing data
    if (missing_data==0).all():
        pass

    elif (missing_data>0).sum()<100:
        logger.error("Missing data:")
        print(
            train_data_downsampled.loc[missing_data]
        )
        x_train=train_data_downsampled[features].ffill(axis=0).values

    else:
        logger.error("Missing data:")
        print(
            train_data_downsampled.loc[missing_data]
        )
        raise Exception()
    
    scaler=StandardScaler()
    if norm:
        z_train=scaler.fit_transform(x_train)
    else:
        z_train=x_train
        scaler.mean_ = np.zeros(x_train.shape[1])  # Zeros mean
        scaler.scale_ = np.ones(x_train.shape[1])  # Ones scale


    # train
    model=MODELS[model_arch](**kwargs)
    logger.info("Training %s on dataset of size %s", model, z_train.shape)
    sample_weights=get_sample_weights(y_train)
    model.fit(z_train, y_train, sample_weight=sample_weights)
    model_path=os.path.join(output_folder, f"{model_arch}.pkl")
    with open(model_path, "wb") as handle:
        pickle.dump((model, features, scaler), handle)

    try:
        preds=inference(test_data, label, model_path, inactive_states=["inactive", "inactive+pe", "inactive+rejection", "inactive+micromovement"])
    
    except Exception as error:
        test_data.reset_index().to_feather(os.path.join(output_folder, "test_predictions.feather"))
        raise error
    else:
        preds.reset_index().to_feather(os.path.join(output_folder, "test_predictions.feather"))

    # Generate visualizations
    title=[]
    for k, v in eval_params.items():
        if k == "freqs":
            v=", ".join([str(e) for e in v])
        
        title.append(f"{k}: {v}")
    title="\n".join(title)

    save_confusion_matrix(preds, label, "prediction", output_folder, "test_confusion_matrix.png",  labels=LABELS, title=title)
    title+=" with rules"
    save_confusion_matrix(preds, label, "prediction2", output_folder, "test_confusion_matrix_with_rules.png",  labels=LABELS, title=title)
    
    preds=annotate_active_state(
        preds, y_true="behavior", y_pred="prediction2",
        inactive_states=["inactive", "inactive+micromovement", "inactive+pe", "inactive+rejection"]
    )
    save_confusion_matrix(preds, "active.gt", "active.pr", output_folder, "test_confusion_matrix_inactive.png", labels=None, title=title)

    y_true=preds[label].values
    y_pred=preds["prediction"].values
    columns=[f"{behavior}_prob" for behavior in np.unique(y_true)]
    y_score=preds[columns].values
    sample_weights=get_sample_weights(y_true)
    
    # Generate metrics
    try:
        metrics = {
            "accuracy_score": accuracy_score                         (y_true=y_true, y_pred=y_pred),
            "balanced_accuracy_score": balanced_accuracy_score       (y_true=y_true, y_pred=y_pred, sample_weight=sample_weights),
            "balanced_accuracy_score_adj": balanced_accuracy_score   (y_true=y_true, y_pred=y_pred, sample_weight=sample_weights, adjusted=True),
            "top_2_accuracy_score": top_k_accuracy_score             (y_true=y_true, y_score=y_score, sample_weight=sample_weights, k=2),
            "top_3_accuracy_score": top_k_accuracy_score             (y_true=y_true, y_score=y_score, sample_weight=sample_weights, k=3),
        }
    except Exception as error:
        print(error)
        print(traceback.print_exc())
        import ipdb; ipdb.set_trace()
    metrics={
        k: str(np.round(v, 2)) for k, v in metrics.items()
    }

    with open(os.path.join(output_folder, "metrics.yaml"), "w") as handle:
        yaml.safe_dump(metrics, handle)


def downsample_dataset(data, downsample):
    """
    Keeps 1/downsample of the chunk-ids

    Removes at random some of the rows in the dataset,
    in blocks of data from the same chunk and animal.
    The probability of keeping a block is 1/downsample
    So if downsample=1, all data is kept (p=1) and the higher downsample,
    the lower the p
    """
    
    p_keep = 1 / downsample

    data["chunk_lid"]=data["chunk"]+data["local_identity"]
    chunk_lids=data["chunk_lid"].drop_duplicates().values.tolist()
    kept=[]

    for chunk_lid in chunk_lids:
        if np.random.uniform(0, 1) < p_keep:
            kept.append(chunk_lid)

    data=data.loc[data["chunk_lid"].isin(kept)]
    return data



def main(**kwargs):

    assert os.path.exists("split.yaml")
    animals=load_train_test_split()
    train(animals["train"], animals["test"], norm=True, output_folder="normalization_output", **kwargs)
    train(animals["train"], animals["test"], norm=False, output_folder="raw_output", **kwargs)