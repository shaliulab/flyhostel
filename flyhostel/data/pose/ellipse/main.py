import argparse
import logging
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from flyhostel.data.pose.main import FlyHostelLoader


from .ellipse import (
    project_to_absolute,
    compute_ellipse_parameters,
    preprocess_ellipses_mp
)
from .context import quantify_activity_in_context
from .dr import main as run_dimred
from .hungarian import hungarian_matching
from .utils import select_loader
from .distance import compute_min_distance
from flyhostel.data.interactions.classifier.inter_orientation import compute_inter_orientation_one_pair
from flyhostel.data.pose.constants import framerate as FRAMERATE
logger=logging.getLogger(__name__)

from sync_paper.sleep import sleep_annotation_rf
from sync_paper.constants import INACTIVE_STATES

def load_fly_data(loader, min_time, max_time, n_jobs=1, framerate=50, **kwargs):
    loader.load_centroid_data(min_time=min_time, max_time=max_time, n_jobs=n_jobs)
    loader.load_pose_data(min_time=min_time, max_time=max_time)
    loader.pose=loader.pose.merge(loader.dt[["frame_number", "center_x", "center_y"]], on="frame_number")
    loader.load_interaction_data(framerate=framerate, **kwargs)
    loader.load_behavior_data(min_time=min_time, max_time=max_time)
    loader.behavior.sort_values(["frame_number"], inplace=True)
    print("Computing sleep")
    loader.behavior["inactive_states"]=loader.behavior["prediction2"].isin(INACTIVE_STATES)
    loader.sleep=sleep_annotation_rf(loader.behavior)


def process_experiment(experiment, identities, min_time, max_time, sources=["opencv", "pose"], n_jobs=1, n_jobs_ethoscopy=1, **kwargs):
    loaders=load_data(experiment, identities, min_time=min_time, max_time=max_time, n_jobs=n_jobs_ethoscopy, **kwargs)
    temporal_features=process_data(loaders, sources=sources, n_jobs=n_jobs)
    
    dataset=[]
    dts=[]
    for loader in loaders:
        dts.append(loader.dt)
        dataset.append(loader.interaction_ellipse3)
    dataset=pd.concat(dataset, axis=0).reset_index(drop=True)
    sleep=pd.concat([loader.sleep for loader in loaders], axis=0).reset_index()
    dts=pd.concat(dts, axis=0).reset_index(drop=True)

    time_index=dts[["frame_number", "t"]]
    time_index["t_round"]=1*time_index["t"]//1
    time_index=time_index.groupby("t_round").first().reset_index().drop("t", axis=1).rename({"t_round": "t"}, axis=1)
    sleep=sleep.merge(time_index, on="t", how="left")

    dataset.to_feather("dataset.feather")
    sleep.to_feather("sleep.feather")

    index=dataset.groupby(["interaction", "id", "nn"]).first()
    min_distance=dataset.groupby(["interaction", "id", "nn"]).agg({"distance_ellipse_mm": np.min}).reset_index().rename({
        "distance_ellipse_mm": "distance_ellipse_mm_min"
    }, axis=1)


    index["interaction_name"]=np.nan
    index=index.merge(min_distance, on=["interaction", "id", "nn"], how="left")
    index.loc[(index["id_distance_pre_10"]<1.5)&(index["nn_distance_pre_10"]<1.5), "interaction_name"]="QW"
    index.loc[(index["id_distance_pre_10"]>1.5)&(index["nn_distance_pre_10"]<1.5)&(index["id_distance_post_10"]>1.5)&(index["nn_distance_post_10"]<1.5), "interaction_name"]="nn_resist"
    index.loc[(index["id_distance_pre_10"]>1.5)&(index["nn_distance_pre_10"]<1.5)&(index["id_distance_post_10"]>1.5)&(index["nn_distance_post_10"]>1.5), "interaction_name"]="id_A_nn"
    index.loc[(index["id_distance_pre_10"]>1.5)&(index["nn_distance_pre_10"]<1.5)&(index["id_distance_post_10"]<1.5)&(index["nn_distance_post_10"]<1.5), "interaction_name"]="nn_S_id"
    index.loc[(index["id_distance_pre_10"]<1.5)&(index["nn_distance_pre_10"]>1.5)&(index["id_distance_post_10"]<1.5)&(index["nn_distance_post_10"]>1.5), "interaction_name"]="id_resist"
    index.loc[(index["id_distance_pre_10"]<1.5)&(index["nn_distance_pre_10"]>1.5)&(index["id_distance_post_10"]>1.5)&(index["nn_distance_post_10"]>1.5), "interaction_name"]="nn_A_id"
    index.loc[(index["id_distance_pre_10"]<1.5)&(index["nn_distance_pre_10"]>1.5)&(index["id_distance_post_10"]<1.5)&(index["nn_distance_post_10"]<1.5), "interaction_name"]="id_S_nn"
    index.loc[(index["id_distance_pre_10"]>1.5)&(index["nn_distance_pre_10"]>1.5), "interaction_name"]="W"
    
    features=temporal_features + ["duration", "distance_ellipse_mm_min"]
    index, model, scaler=run_dimred(index, features=features, algorithm="UMAP")
    index.to_csv("index.csv")
    with open("ml.pkl", "wb") as handle:
        pickle.dump((model, scaler), handle)


def process_data(loaders, sources, n_jobs):
    ellipse_data=model_ellipses(loaders, sources, n_jobs=n_jobs)
    describe_interactions_between_ellipses(loaders, ellipse_data)
    temporal_features=quantify_activity_in_context(loaders, [10, 60, 300], FRAMERATE)
    return temporal_features


def load_data(experiment, identities, min_time, max_time, n_jobs, **kwargs):

    loaders=[FlyHostelLoader(experiment=experiment, identity=identity) for identity in identities]
    for loader in loaders:
        load_fly_data(loader, min_time, max_time, n_jobs=n_jobs, **kwargs)
    
    return loaders

def model_ellipses(loaders, sources, n_jobs):
    frame_numbers=[]
    for loader in loaders:
        frame_numbers.extend(
            loader.interaction["frame_number"].tolist()
        )
    frame_numbers=sorted(np.unique(frame_numbers).tolist())
    ellipse_data_cv=None
    ellipse_data_pose=None

    if "opencv" in sources:
        ellipse_data_cv=get_ellipses_from_opencv(loaders, frame_numbers, n_jobs=n_jobs)
        index=ellipse_data_cv.loc[ellipse_data_cv["id"].isna()].groupby("frame_number").size()
        # get the frame number where not all ellipses got an id
        frame_numbers=index.loc[index!=0].index.tolist()
    if "pose" in sources:
        ellipse_data_pose=get_ellipses_from_pose(loaders, frame_numbers)
    
    if ellipse_data_pose is None:
        ellipse_data=ellipse_data_cv
    elif ellipse_data_cv is None:
        ellipse_data=ellipse_data_pose
    else:    
        ellipse_data=pd.concat([
            ellipse_data_cv.loc[~ellipse_data_cv["id"].isna()],
            ellipse_data_pose[["x", "y", "major", "minor", "angle", "frame_number", "id", "source"]]
        ], axis=0).reset_index(drop=True).drop_duplicates(["id", "frame_number"], keep="first")\
        .sort_values(["frame_number", "id"])
    return ellipse_data


def describe_interactions_between_ellipses(loaders, ellipse_data):
    for loader in loaders:
        loader.interaction_ellipse=loader.interaction.merge(ellipse_data, on=["id", "frame_number"], how="left")
        loader.interaction_ellipse["distance_ellipse_mm"]=np.nan

    inter_orientation_l=[]
    dt_l=[]
    for loader1 in loaders:
        print(loader1)
        loader2=None
        dt_l.append(loader1.dt)
        for i in range(loader1.interaction_ellipse.shape[0]):
            row=loader1.interaction_ellipse.iloc[i]
            fn=row["frame_number"]
            id1=loader1.ids[0]
            id2=row["nn"]
            loader2=select_loader(loaders, id2)
            
            if ~np.isnan(row["distance_ellipse_mm"]):
                continue

            if loader2 is None:
                continue

            interaction1=loader1.interaction_ellipse
            interaction2=loader2.interaction_ellipse
            ellipse1=interaction1.loc[(interaction1["frame_number"]==fn)&(interaction1["nn"]==id2)].squeeze()
            ellipse2=interaction2.loc[(interaction2["frame_number"]==fn)&(interaction2["nn"]==id1)].squeeze()

            px_per_mm=loader1.px_per_mm
            distance = compute_min_distance(ellipse1, ellipse2, debug=False) / px_per_mm
            interaction1.loc[(interaction1["frame_number"]==fn)&(interaction1["nn"]==id2), "distance_ellipse_mm"]=distance
            interaction2.loc[(interaction2["frame_number"]==fn)&(interaction2["nn"]==id1), "distance_ellipse_mm"]=distance
        
    dt=pd.concat(dt_l, axis=0).reset_index(drop=True)
    del dt_l

    for loader1 in tqdm(loaders, desc="Computing inter orientation"):
        for loader2 in loaders:
            if loader1==loader2:
                continue
            inter_orientation=compute_inter_orientation_one_pair(loader1, loader2)
            inter_orientation["id"]=loader1.ids[0]
            inter_orientation["nn"]=loader2.ids[0]
            inter_orientation_l.append(inter_orientation)
    
    inter_orientation=pd.concat(inter_orientation_l, axis=0).reset_index(drop=True)
    for loader in loaders:
        loader.interaction_ellipse2=loader.interaction_ellipse.merge(
            inter_orientation,
            on=["id", "nn", "frame_number"],
            how="left"
        )
        loader.interaction_ellipse2=loader.interaction_ellipse2.merge(
            dt[["distance", "id", "frame_number"]].rename({"distance": "id_distance"}, axis=1),
            on=["id", "frame_number"],
            how="left"
        ).merge(
            dt[["distance", "id", "frame_number"]].rename({"id": "nn", "distance": "nn_distance"}, axis=1),
            on=["nn", "frame_number"],
            how="left"
        )
        loader.interaction_ellipse2.sort_values(["frame_number", "id"], inplace=True)
    del dt


def get_ellipses_from_pose(loaders, frame_numbers):
    """
    Compute ellipse formed by head, abdomen, and middle legs on either side
    """
    
    out=[]
    for loader in loaders:
        df_pose=project_to_absolute(loader, ["head", "abdomen", "mRL", "mLL"])
        df_pose=df_pose.loc[df_pose["frame_number"].isin(frame_numbers)]
        df_pose=compute_ellipse_parameters(df_pose)
        out.append(df_pose)
    df_pose=pd.concat(out, axis=0).reset_index(drop=True)
    df_ellipses=df_pose.loc[
        df_pose["frame_number"].isin(frame_numbers)
    ]
    df_ellipses["source"]="pose"
    del out
    return df_ellipses


def get_ellipses_from_opencv(loaders, frame_numbers, n_jobs=1):
    """
    Compute ellipse formed by the contour detected by opencv at segmentation

    Because the contour is not saved anywhere, it is recomputed when calling preprocess_ellipses_mp
    """

    basedir = loaders[0].basedir
    ellipse_data, contours=preprocess_ellipses_mp(
        basedir,
        frame_numbers,
        n_jobs=n_jobs
    )
    assert ellipse_data is not None

    centroids=[]
    for loader in loaders:
        xy=loader.dt[["id", "frame_number", "center_x", "center_y"]].copy()
        xy=xy.loc[xy["frame_number"].isin(frame_numbers)]
        duplicated_locations=xy.duplicated().sum()
        if duplicated_locations>0:
            logger.warning("%s duplicated rows in loader.dt found", duplicated_locations)
            xy.drop_duplicates(inplace=True)

        centroids.append(xy)
    centroids=pd.concat(centroids, axis=0).reset_index(drop=True)
    ellipse_data=hungarian_matching(ellipse_data, centroids)
    ellipse_data["source"]="opencv"
    return ellipse_data, contours

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment")
    ap.add_argument("--identities", nargs="+", type=int, default=None)
    ap.add_argument("--min-time", type=int, default=None)
    ap.add_argument("--max-time", type=int, default=None)
    return ap


def main():
    ap=get_parser()
    args=ap.parse_args()
    if args.identities is None:
        number_of_animals=int(args.experiment.split("_")[1].replace("X", ""))
        identities=list(range(1, number_of_animals+1))
    else:
        identities=args.identities

    process_experiment(args.experiment, identities, min_time=args.min_time, max_time=args.max_time)

if __name__ == "__main__":
    main()