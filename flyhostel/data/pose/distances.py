import numpy as np
import pandas as pd

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
    