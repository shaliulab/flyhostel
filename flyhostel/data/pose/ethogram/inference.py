import os.path
import pickle
import logging

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score

from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.ethogram.loader import process_animal
from flyhostel.data.deg import LABELS
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.constants import BEHAVIOR_IDX_MAP
from flyhostel.data.pose.constants import WAVELET_DOWNSAMPLE
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.data.pose.ethogram.export import save_deg_prediction_file
from flyhostel.data.pose.ethogram.utils import (
    annotate_active_state,
    annotate_bout_info,
)
from flyhostel.data.pose.ethogram.rules import main as apply_rules
from flyhostel.data.pose.ethogram.postprocess import postprocess_behaviors
from flyhostel.data.pose.ethogram.plot import save_confusion_matrix
logger=logging.getLogger(__name__)


def inference(
        experiment, identity, model_path, output="./",
        frame_numbers=None, postprocess=True, **kwargs
    ):

    if not postprocess:
        logger.warning("Postprocessing behaviors is required")
    micromovement_behavior="inactive+micromovement"

    
    loader=FlyHostelLoader(experiment=experiment, identity=identity, chunks=range(0, 400))
    assert loader.identity_table is not None
    assert loader.roi_0_table is not None
    
    dataset=process_animal(loader=loader, **kwargs, load_deg_data=False).reset_index()

    # dataset=dataset.loc[
    #     (dataset["frame_number"]>6948465)&\
    #     (dataset["frame_number"]<=6950465)
    # ]

    print(np.round(loader.dt.loc[loader.dt["frame_number"]==6949465, ["x", "y"]].values, decimals=2))


    logger.debug("Read dataset of shape %s", dataset.shape)
    dataset["chunk"]=dataset["frame_number"]//CHUNKSIZE
    dataset["frame_idx"]=dataset["frame_number"]%CHUNKSIZE
    dataset["zt"]=(dataset["t"]/3600).round(2)
    if frame_numbers is not None:
        dataset=dataset.loc[
            dataset["frame_number"].isin(frame_numbers)
        ]

    with open(model_path, "rb") as handle:
        model, features, scaler= pickle.load(handle)
   
    behaviors=model.classes_
    output_cols=[
        "id", "local_identity", "identity",
        "t", "frame_number", "chunk", "frame_idx",
        "x", "y","food_distance", "notch_distance",
        "score", "prediction", "rule", "prediction2",
        "bout_in_pred", "bout_out_pred", "bout_count_pred", "duration_pred",
        "proboscis", "head_proboscis_distance"
        ] + behaviors.tolist()

    if "behavior" in dataset.columns:
        has_gt=True
        output_cols.append("behavior")
    else:
        has_gt=False
    
    # 1. RF inference
    # import ipdb; ipdb.set_trace()
    predictions=inference_(
        dataset, "behavior", model_path=model_path,
        inactive_states=["inactive", micromovement_behavior, "inactive+pe", "inactive+rejection"]
    )
    # Save results
    feather_out=os.path.join(output, "dataset.feather")
    # feather_input=os.path.join(output, "input.feather")
    behaviors_suffix="_prob"
    for col in predictions.columns:
        if col.endswith(behaviors_suffix):
            predictions[col.replace(behaviors_suffix, "")]=predictions[col]


    logger.info("Saving to ---> %s", feather_out)
    final_cols=[]
    for col in output_cols:
        if col in predictions.columns:
            final_cols.append(col)
        else:
            logger.warning("Ignoring field %s", col)
    predictions[final_cols].reset_index(drop=True).to_feather(feather_out)

    logger.debug("Saving prediction files")
    save_deg_prediction_file(experiment, predictions, features, column="prediction2")


def inference_(dataset, label, model_path, inactive_states):

    with open(model_path, "rb") as handle:
        model, features, scaler= pickle.load(handle)

    print(f"Loaded {model_path}. Will predict among the following classes : {model.classes_}")
    dataset.drop(["bout_count_pred", "bout_in_pred", "bout_out_pred", "duration_pred", "correct_bout_fraction"], axis=1, errors="ignore", inplace=True)

    missing_counts=dataset[features].isna().sum(axis=0)
    if (missing_counts>0).any():
        print("Missing values count: ", np.array(features)[missing_counts>0], missing_counts[missing_counts>0])
        dataset[features]=dataset[features].ffill(axis=0)
    x_test=dataset[features].values
    logger.debug("Predicting %s points", x_test.shape[0])
    z_test=scaler.transform(x_test)
    
    probabilities=model.predict_proba(z_test)
    predictions=[model.classes_[i] for i in probabilities.argmax(axis=1)]
    confidence=probabilities.max(axis=1)
    prob_sorted=np.sort(probabilities, axis=1)
    contrast=prob_sorted[:, -1] - prob_sorted[:, -2]

    preds=dataset.copy()
    preds["prediction"]=predictions
    preds["confidence"]=confidence
    preds["contrast"]=contrast
    preds["score"]=confidence
    try:
        preds["correct"]=preds[label]==preds["prediction"]
        preds=preds.merge(
            preds.groupby("bout_count").agg({"correct": np.mean}).reset_index().rename({"correct": "correct_bout_fraction"}, axis=1),
            on=["bout_count"], how="left"
        )
        has_gt=True
        behavior_col="behavior"
    except KeyError:
        has_gt=False
        behavior_col=None

    bp_probabilities=[cl + "_prob" for cl in model.classes_]
    index=preds.index
    preds=pd.concat([
        preds.drop(bp_probabilities, axis=1, errors="ignore"),
        pd.DataFrame(probabilities, index=preds.index, columns=bp_probabilities),
    ], axis=1)

    preds.index=index

    # 2. Remove very short bouts (<1 percentile distribution in GT data)
    logger.debug("Postprocessing behaviors")
    preds=postprocess_behaviors(preds, percentile=1, column="prediction", behaviors=BEHAVIOR_IDX_MAP)

    preds.reset_index().to_feather("backup.feather")

    # 3. Apply rules
    logger.debug("Applying logic rules")
    preds=apply_rules(preds, micromovement_behavior="inactive+micromovement")
    
    logger.debug("Annotating bout structure")
    preds.drop(["bout_in_pred", "bout_out_pred","bout_count_pred","duration_pred"], axis=1, inplace=True, errors="ignore")
    preds=preds.groupby("id").apply(lambda df: annotate_bout_info(df, FRAMERATE//WAVELET_DOWNSAMPLE, behavior=behavior_col, prediction="prediction2")).reset_index(drop=True)
    
    return preds


def evaluate_model_on_test_set(test_set, model_path, output_folder, micromovement_behavior="inactive+micromovement", by_fly=False):

    predictions=inference(
        test_set, "behavior", model_path=model_path,
        inactive_states=["inactive", micromovement_behavior, "inactive+pe", "inactive+rejection"]
    )
    predictions=postprocess_behaviors(predictions, percentile=1, column="prediction", behaviors={
        "background": (0,),
        "walk": (1,),
        "groom": (2,),
        "feed": (3,),
        "inactive": (5,),
        "inactive+pe": (4,5),
        "inactive+micromovement": (5,7),
        "inactive+rejection": (5,7,9),
    })

    predictions=apply_inactive_pe_requirement(predictions, "inactive")
    predictions=predictions.groupby("id").apply(lambda df: annotate_bout_info(df, 30, prediction="prediction")).reset_index(drop=True)   
    predictions["prediction2"]=predictions["prediction"].copy()
    predictions=apply_sequence_rules(predictions, micromovement_behavior)
    predictions=predictions.groupby("id").apply(lambda df: annotate_bout_info(df, 30, prediction="prediction2")).reset_index(drop=True)
    predictions=apply_inactive_micromovement_limit(predictions, micromovement_behavior, idx=1.6)
    predictions=predictions.groupby("id").apply(lambda df: annotate_bout_info(df, 30, prediction="prediction2")).reset_index(drop=True)
    
    predictions.drop(["bout_in_pred", "bout_out_pred","bout_count_pred","duration_pred"], axis=1, inplace=True, errors="ignore")
    
    predictions=annotate_active_state(predictions, y_true="behavior", y_pred="prediction2", inactive_states=["inactive", micromovement_behavior, "inactive+pe", "inactive+rejection"])

    scores_records=[]
    for behavior in test_set["behavior"].unique():
        scores_records.append((
            behavior, f1_score(y_pred=predictions["prediction2"]==behavior, y_true=predictions["behavior"]==behavior)
        ))

    scores_records.append((
        "activity", f1_score(y_pred=predictions["active.pr"]=="active", y_true=predictions["active.gt"]=="active")
    ))

    scores=pd.DataFrame.from_records(scores_records, columns=["behavior", "f1"])
    scores.to_csv(os.path.join(output_folder, "f1.csv"))

    save_confusion_matrix(
        predictions,  y_true="active.gt", y_pred="active.pr",
        output_folder=output_folder, name="test_confusion_matrix_activity.png",
        title=model_path, labels=None
    )
    save_confusion_matrix(
        predictions, y_true="behavior", y_pred="prediction2",
        output_folder=output_folder, name="test_confusion_matrix.png",
        title=model_path, labels=LABELS
    )

    ids=predictions["id"].unique()
    if by_fly:
        for id in ids:
            save_confusion_matrix(
                predictions.loc[predictions["id"]==id],  y_true="active.gt", y_pred="active.pr",
                output_folder=output_folder, name=f"test_confusion_matrix_activity_{id}.png",
                title=model_path, labels=None
            )
            save_confusion_matrix(
                predictions.loc[predictions["id"]==id], y_true="behavior", y_pred="prediction2",
                output_folder=output_folder, name=f"test_confusion_matrix_{id}.png",
                title=model_path, labels=LABELS
            )
    return predictions