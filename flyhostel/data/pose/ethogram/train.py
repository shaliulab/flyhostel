import glob
import yaml
import os.path
import datetime
import itertools
import logging
import h5py
import pickle
import joblib

logger=logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import sklearn
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_rows", 1200)
from motionmapperpy.motionmapper import setRunParameters, generate_frequencies

from flyhostel.data.pose.constants import chunksize, framerate, bodyparts_xy, bodyparts
from sklearn.metrics import ConfusionMatrixDisplay
from flyhostel.data.pose.distances import compute_distance_features_pairs
from sklearn.utils.class_weight import compute_class_weight
from flyhostel.data.pose.ethogram.utils import annotate_bout_duration, annotate_bouts
from sklearn.preprocessing import StandardScaler

from flyhostel.data.pose.ethogram.loader import load_animals


filters="rle-jump"
params=setRunParameters()
wavelet_downsample=params.wavelet_downsample
freqs=np.round(generate_frequencies(params.minF, params.maxF, params.numPeriods), 4)
freq_names=[f"{e0}_{e1}" for e0, e1 in itertools.product(bodyparts_xy, freqs) if not e0.startswith("thorax") and not e0.startswith("head")]

features=freq_names + ["head_proboscis_distance"]

def train(train_set_animals, test_set_animals, norm=True, output_folder=".", label="label", n_jobs_load=1, n_jobs_model=1, **kwargs):
    """
    Train a RandomForest classifier to predict a behavior using the WT features and the head-proboscis distance

    Arguments:
       train_set_animals (list): Animals in experiment__identity format used for training
       test_set_animals (list): Animals in experiment__identity format used for test
       norm (bool): If True, input to the model is z-values computed based on standard deviation of train set features
       output_folder (str): Where to save outputs
       label (str): If behavior, the original behavior is the target, if label, several behaviors are grouped into micromovement
       n_jobs_load (int): How many animals can be loaded in parallel?
       n_jobs_model (int): How many jobs can the model use for predictions?
       **kwargs: extra arguments to the model

    Returns:
       Saves to output folder
           1. test_confusion_matrix.png
               visualization of the specificity and sensitivity of the model in the test set
           2. test_predictions.feather
               test set data annotated by the model
           3. random_forest.pkl
                contains the tuple (model, scaler, features) 
    """
    train_data=load_animals(train_set_animals, n_jobs=n_jobs_load)
    test_data=load_animals(test_set_animals, n_jobs=n_jobs_load)

    # check all behaviors in the training set are also present in the test set
    trained_behaviors=train_data[label].unique()
    for behavior in test_data[label].unique():
        assert behavior in trained_behaviors, f"{behavior} is present in test but not train set"
        
    # check all frequencies have been computed and present in the train data
    missing_freqs=[feat for feat in features if not feat in train_data.columns]
    assert len(missing_freqs)==0, f"Some of the frequencies produced by LTA are not present in the data ({missing_freqs})"

    # Downsample so that least abundant classes are not so underrepresented
    n_samples_inactive_pe=train_data.loc[train_data["behavior"]=="inactive+pe"].shape[0]
    train_data_downsampled=train_data.groupby("behavior").apply(lambda df: df.sample(n=min(n_samples_inactive_pe, df.shape[0]))).reset_index(drop=True)
    assert train_data_downsampled["head_proboscis_distance"].isna().sum()==0

    y_train=train_data_downsampled[label].values

    # Compute class weights
    classes=np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    weight_dict = {classes[i]: class_weights[i] for i in range(len(class_weights))}
    sample_weights = np.array([weight_dict[cls] for cls in y_train])

    x_train=train_data_downsampled[features].values

    # check there is no missing data
    assert (np.isnan(x_train).sum(axis=0)==0).all()
    scaler=StandardScaler()

    if norm:
        z_train=scaler.fit_transform(x_train)
    else:
        z_train=x_train

    # train
    model=RandomForestClassifier(n_jobs=n_jobs_model, **kwargs)
    model.fit(z_train, y_train, sample_weight=sample_weights)

    x_test=test_data[features].values
    z_test=scaler.transform(x_test)
    probabilities=model.predict_proba(z_test)
    predictions=[model.classes_[i] for i in probabilities.argmax(axis=1)]
    confidence=probabilities.max(axis=1)
    prob_sorted=np.sort(probabilities, axis=1)
    contrast=prob_sorted[:, -1] - prob_sorted[:, -2]

    preds=test_data.copy()
    preds["prediction"]=predictions
    preds["confidence"]=confidence
    preds["contrast"]=contrast
    preds["correct"]=preds[label]==preds["prediction"]
    
    bp_probabilities=[cl + "_prob" for cl in model.classes_]
    
    index=preds.index
    preds=pd.concat([
        preds,
        pd.DataFrame(probabilities, index=preds.index, columns=bp_probabilities),
    ], axis=1)
    preds=preds.merge(
        preds.groupby("bout_count").agg({"correct": np.mean}).reset_index().rename({"correct": "correct_bout_fraction"}, axis=1),
        on=["bout_count"], how="left"
    )
    preds.index=index

    ground_truth=preds[label].values
    ConfusionMatrixDisplay.from_predictions(y_true=ground_truth, y_pred=predictions, normalize="true", xticks_rotation=45)
    plt.savefig(os.path.join(output_folder, "test_confusion_matrix.png"))
    preds.reset_index().to_feather(os.path.join(output_folder, "test_predictions.feather"))
    with open(os.path.join(output_folder, "random_forest.pkl"), "wb") as handle:
        pickle.dump((model, features, scaler), handle)
        

def load_train_test_split():
    with open("split.yaml", "r") as handle:
        train_test_split=yaml.safe_load(handle)
    return train_test_split


def main(**kwargs):

    assert os.path.exists("split.yaml")
    animals=load_train_test_split()
    train(animals["train"], animals["test"], norm=True, output_folder="normalization_output", **kwargs)
    train(animals["train"], animals["test"], norm=False, output_folder="raw_output", **kwargs)