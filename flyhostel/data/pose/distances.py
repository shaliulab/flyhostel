import itertools
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from flyhostel.data.pose.pose import full_interpolation

def compute_distance_features(pose, part1, part2):
    distance=np.sqrt(
        (pose[f"{part1}_x"]-pose[f"{part2}_x"])**2 + (pose[f"{part1}_y"]-pose[f"{part2}_y"])**2
    )
    return distance

def compute_distance_features_pairs(pose, pairs=[("head", "proboscis"),]):
    
    distance_features={
        f"{part1}_{part2}_distance": compute_distance_features(pose, part1, part2)
        for part1, part2 in pairs
    }

    distance_features=pd.DataFrame(distance_features)
    distance_features["id"]=pose["id"]
    distance_features["frame_number"]=pose["frame_number"]
    pose=pose.merge(distance_features, on=["id", "frame_number"], how="left")
    return pose
    

def add_speed_features(data, features, bodyparts):
    bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in bodyparts]))
    pose=data[["id", "frame_number"] + bodyparts_xy]
    pose=full_interpolation(pose, bodyparts_xy)
    pose_arr=pose[bodyparts_xy].values.reshape((pose.shape[0], -1, 2))
    distance=np.sqrt(np.sum(np.diff(pose_arr, axis=0)**2, axis=2))
    distance_df=pd.DataFrame(distance, columns=bodyparts, index=pose.index[:-1])

    series=(~distance_df[bodyparts].isna().all(axis=0))
    features+=series[series].index.values.tolist()
    data=pd.concat([
        data.iloc[:-1].reset_index(drop=True),
        distance_df.reset_index(drop=True)
    ], axis=1)
    return data, features


def add_interdistance_features(data, features, bodyparts, prefix=""):

    for bp1, bp2 in tqdm(itertools.combinations(bodyparts, 2)):
        feature=f"{prefix}{bp1}__{bp2}"
        diff=(data[[bp1 + "_x", bp1 + "_y"]].values - data[[bp2 + "_x", bp2 + "_y"]].values)
        distance=pd.Series(np.sqrt((diff**2).sum(axis=1)))
        distance.ffill(inplace=True)
        distance.bfill(inplace=True)
        distance=distance.values
        data[feature]=distance
        features.append(feature)
    return data, features