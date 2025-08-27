import argparse
import logging
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from umap import UMAP
import hdbscan
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA

print(sklearn.__version__)

from flyhostel.utils import get_basedir
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.ethogram.plot import bin_behavior_table_v2
from flyhostel.data.pose.constants import BEHAVIOR_IDX_MAP
from flyhostel.data.pose.constants import framerate as FRAMERATE
from sync_paper.sleep import sleep_annotation_rf
from sync_paper.constants import PURE_INACTIVE_STATES

from .inter_orientation import calculate_angles_with_vertical_batch, compute_inter_orientation
from .dtw import  compress_interaction
from .utils import select_loader

logger=logging.getLogger(__name__)

def annotate_last_behavior(loader, time_window_length=1):
    # prediction window captures the most common behavior
    # in the last time_window_length seconds prior to each frame
    loader.behavior["t_round"]=time_window_length*(loader.behavior["t"]//time_window_length)
    (behavior_windows, _, _) = bin_behavior_table_v2(
        loader.behavior[["id", "t", "frame_number", "prediction2", "score"] + list(BEHAVIOR_IDX_MAP.keys())].copy(),
        time_window_length=time_window_length, x_var=None, t0=None, behavior_col="prediction2"
    )
    behavior_windows["prediction_window"]=behavior_windows["prediction2"]
    behavior_windows["t_round"]=behavior_windows["t"]
    loader.behavior=loader.behavior.drop(["prediction_window", "score"], axis=1, errors="ignore").merge(behavior_windows[[
        "t_round", "id", "prediction_window", "score"
    ]], on=["t_round", "id"])


def load_fly_data(loader, min_time, max_time):
    print("Loading centroid")
    loader.load_centroid_data(identity=loader.identity, min_time=min_time, max_time=max_time, n_jobs=1, identity_table="IDENTITY_VAL", roi_0_table="ROI_0_VAL")
    files=[(
        loader.get_pose_file_h5py(pose_name="filter_rle-jump"),
        loader.get_pose_file_h5py(pose_name="raw")
    )]
    print("Loading pose")
    loader.load_pose_data(identity=loader.identity, files=files, min_time=min_time, max_time=max_time)
    points_batch=loader.pose[["head_x", "head_y", "thorax_x", "thorax_y"]].values.reshape((loader.pose.shape[0], 2, 2))
    print("Computing orientation")
    loader.pose["orientation"]=calculate_angles_with_vertical_batch(points_batch)

    print("Informing pose on location")
    xy=loader.dt[["frame_number", "x", "y"]]
    duplicated_locations=xy.duplicated().sum()
    if duplicated_locations>0:
        logger.warning("%s duplicated rows in loader.dt found", duplicated_locations)
        xy.drop_duplicates(inplace=True)
    xy["x"]*=loader.roi_width
    
    # this is not a bug, x and y are normalized with the same number (width)
    # anyway it doesn't matter because the roi is square
    xy["y"]*=loader.roi_width
    loader.pose=loader.pose.merge(
        xy, on="frame_number"
    )

    print("Loading behavior")
    loader.load_behavior_data(loader.experiment, loader.identity, min_time=min_time, max_time=max_time)

    print("Computing sleep")
    loader.behavior["inactive_states"]=loader.behavior["prediction2"].isin(PURE_INACTIVE_STATES)
    loader.sleep=sleep_annotation_rf(loader.behavior)


def load_interactions(experiment, identities, time_index, min_time, max_time, dist_max_mm=None):
    """
    Load cached interactions detected for this experiment
    """
    basedir=get_basedir(experiment)
    interaction_features=["id", "nn", "distance_mm", "frame_number"]

    df_=pd.read_csv(os.path.join(basedir, "interactions", experiment + "_interactions.csv"))[interaction_features]
    
    if dist_max_mm is not None:
        df_=df_.loc[df_["dist_max_mm"]<=dist_max_mm]
    
    df_["identity1"]=[int(e) for e in df_["id"].str.slice(start=-2)]
    df_["identity2"]=[int(e) for e in df_["nn"].str.slice(start=-2)]

    df_=df_.loc[
        (df_["identity1"].isin(identities)) & (df_["identity2"].isin(identities))
    ]
    df_["pair"]=df_["id"]+" " + df_["nn"]

    df_=df_.sort_values(["pair", "frame_number"])
    df_["interaction"]=[0] + (np.cumsum(np.diff(df_["frame_number"])!=10)).tolist()
    df_=df_.merge(time_index, on="frame_number", how="left")
    df_=df_.loc[(df_["t"]>=min_time)&(df_["t"]<max_time)]
    # add global interaction features
    interaction_index=df_.groupby(["id", "interaction"]).agg({"frame_number": [np.min, np.max], "distance_mm": np.min}).reset_index()
    interaction_index.columns=["id", "interaction", "frame_number_start", "frame_number_end", "distance_mm_min"]
    interaction_index["duration"]=(interaction_index["frame_number_end"]-interaction_index["frame_number_start"])/FRAMERATE
    df_=df_.merge(interaction_index, on=["id", "interaction"])
    return df_


def combine_datasets(loader, df_, max_distance_mm_min=4, min_duration=1):
    """
    Put together centroid (location), pose, behavior, sleep and interaction data
    of each animal
    """
    print(loader.identity)
    beh=loader.behavior[["id", "frame_number", "prediction_window", "score", "centroid_speed_1s", "food_distance", "notch_distance"]]
    pose=loader.pose[["id", "frame_number", "orientation"]]
    centroid=loader.dt[["id", "frame_number", "t", "x", "y"]]
    interactions=df_[["id", "interaction", "frame_number", "distance_mm", "inter_orientation", "frame_number_start", "frame_number_end", "duration", "distance_mm_min", "nn", "pair"]]
    interactions=interactions.loc[(interactions["duration"]>=min_duration) & (interactions["distance_mm_min"]<max_distance_mm_min)]
    out = beh.merge(pose, on=["id", "frame_number"], how="inner")\
      .merge(interactions, on=["id", "frame_number"], how="inner")\
      .merge(centroid, on=["id", "frame_number"], how="inner")

    out["bout_in"]=out["frame_number"]-out["frame_number_start"]
    out["bout_in_fraction"]=(out["frame_number"]-out["frame_number_start"])/(out["frame_number_end"]-out["frame_number_start"])


    loader.combined_dataset=pd.merge_asof(
        out,
        loader.sleep.rename({"inactive_rule": "asleep"}, axis=1)[["t", "asleep"]],
        on=["t"],
        tolerance=1,
        direction="backward"
    )    

def one_hot_encoding(loader, encoder, fit=False, col_name="prediction_window"):
    """
    Transform a categorical column into a bunch of columns with 1/0
    one column for every category in the original column 
    """
    dataset=loader.combined_dataset.copy()
    series=dataset[col_name]
    series_reshaped = series.values.reshape(-1, 1)
    try:
        if fit:
            one_hot = encoder.fit_transform(series_reshaped)
        else:
            one_hot=encoder.transform(series_reshaped)
            
        columns=[f"{col_name}_{c}" for c in encoder.categories[0]]
        one_hot_df = pd.DataFrame(one_hot, columns=columns)
        dataset=pd.concat([dataset, one_hot_df], axis=1)
        loader.full_dataset=dataset

    except Exception as error:
        print(error)
        import ipdb; ipdb.set_trace()

    return columns


def process_pair(loaders, pair, features, target_length=5):
    """
    Generate the compressed interactions dataset for all interactions of this pair

    A compressed interaction is an interaction whose information has become a single row
    """
    compressed=[]
    id1, id2 = pair.split(" ")
    identity1=int(id1.split("|")[1])
    identity2=int(id2.split("|")[1])

    focal_fly_data=select_loader(loaders, identity1).full_dataset
    focal_fly_data=focal_fly_data.loc[focal_fly_data["nn"]==id2]
    
    side_fly_data=select_loader(loaders, identity2).full_dataset
    side_fly_data=side_fly_data.loc[side_fly_data["nn"]==id1]
    assert focal_fly_data.shape == side_fly_data.shape
    for interaction, dfff_focal in tqdm(focal_fly_data.groupby("interaction"), desc="Processing pair interactions"):

        fn0=dfff_focal["frame_number_start"].iloc[0]
        
        # interaction column cannot be used to link focal and side fly
        # because the ith interaction of the focal fly is not the same interaction
        # as the ith interaction of the side fly

        dfff_side=side_fly_data.loc[side_fly_data["frame_number_start"]==fn0]

        assert dfff_side.shape==dfff_focal.shape
        dfff_f_comp=compress_interaction(dfff_focal, features + ["t"], target_length=target_length)
        dfff_s_comp=compress_interaction(dfff_side, features+ ["t"], target_length=target_length)
        dfff_f_comp.columns=[f"{c}_focal" for c in dfff_f_comp.columns]
        dfff_s_comp.columns=[f"{c}_side" for c in dfff_s_comp.columns]
        dfff_comp=pd.concat([dfff_f_comp, dfff_s_comp], axis=1)
        columns=dfff_comp.columns.tolist()
        dfff_comp["frame_number_start"]=fn0
        dfff_comp["frame_number_end"]=dfff_focal["frame_number_end"].iloc[0]
        dfff_comp["duration"]=dfff_focal["duration"].iloc[0]
        dfff_comp["distance_mm_min"]=dfff_focal["distance_mm_min"].iloc[0]
        dfff_comp["pair"]=pair
        dfff_comp["interaction"]=interaction
        dfff_comp["t"]=dfff_focal["t"].iloc[0]
        compressed.append(dfff_comp)
    compressed=pd.concat(compressed, axis=0).reset_index(drop=True)
    return compressed, columns


def process_experiment(experiment, identities, min_time, max_time):
    """
    Create an interactions classifier dataset.
    Each time a pair of flies interacts, two rows are added to the dataset,
    one from each point of view
    The row describes numerically the interaction
    """

    scaler=StandardScaler()
    behaviors=[list(BEHAVIOR_IDX_MAP.keys())]
    encoder = OneHotEncoder(sparse=False, dtype=int, categories=behaviors)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=60)
    pca=PCA()
    manifold=UMAP()

    loaders=[FlyHostelLoader(experiment=experiment, identity=identity) for identity in identities]
    time_index=[]
    for loader in loaders:
        load_fly_data(loader, min_time, max_time)
        annotate_last_behavior(loader, time_window_length=1)
        # annotate time (zt in seconds) of every frame in every interaction
        time_index.append(loader.dt[["frame_number", "t"]])
    time_index=pd.concat(time_index, axis=0).drop_duplicates().sort_values("frame_number")
    df_ = load_interactions(experiment, identities, time_index, min_time, max_time)
    df_ = compute_inter_orientation(loaders, df_)
    
    one_hot_columns=[]
    for i, loader in enumerate(loaders):
        combine_datasets(loader, df_)
        if loader.combined_dataset.shape[0]>0:
            one_hot_columns = one_hot_encoding(loader, encoder=encoder, fit=i==0)

    social_features=["score", "centroid_speed_1s", "food_distance", "notch_distance", "distance_mm", "inter_orientation", "asleep"]+one_hot_columns
    X=[]
    raw_data=[]
    for loader in loaders:
        raw_data.append(loader.full_dataset)
    raw_data=pd.concat(raw_data, axis=0).reset_index(drop=True)
    raw_data.to_feather(f"{experiment}_interactions_classifier_raw_data.feather")

    pairs=df_["pair"].unique().tolist()
    for pair in tqdm(pairs, desc="Processing pair"):
        compressed_, columns=process_pair(loaders, pair, social_features, target_length=5)
        X.append(compressed_)
    X=pd.concat(X, axis=0).reset_index(drop=True)
    X.to_csv(f"{experiment}_interactions_classifier_data.csv")
    features=columns + ["duration"]
    X_p=X.loc[~np.isnan(X[features].values).any(axis=1)]
    X_s=scaler.fit_transform(X_p[features])

    pcs=pca.fit_transform(X_s)
    pcs_df=pd.DataFrame(pcs)
    pcs_df.columns=[f"PC{i}" for i in range(1, pcs_df.shape[1]+1)]
    pcs_df=pd.concat([pcs_df, X], axis=1)
    pcs_df.to_csv("pca.csv")

    projection=manifold.fit_transform(X_s)
    umap_df=pd.DataFrame(projection)
    umap_df.columns=[f"UMAP{i}" for i in range(1, umap_df.shape[1]+1)]

    clusterer.fit(projection)
    umap_df["cluster"]=clusterer.labels_
    umap_df=pd.concat([umap_df, X], axis=1)
    umap_df.to_csv("umap.csv")


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