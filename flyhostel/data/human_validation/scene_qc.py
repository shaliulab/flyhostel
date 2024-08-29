"""
Quantify difficulty of challenging frames

Run a QC that extracts behavioral features of the animals in a complex scene (requiring AI)
This helps users tell which scenes are very unlikely to have mistakes or are maybe likely
The features extracted are documented in scene_qc
"""

import os.path
import logging
import math
import glob


import yaml
from tqdm.auto import tqdm
import joblib
import numpy as np
import pandas as pd
import cupy as cp


from .qc import all_id_expected_qc
from flyhostel.data.interactions.neighbors_gpu import compute_distance_between_ids, find_closest_pair
from flyhostel.data.pose.constants import chunksize
pd.set_option("display.max_rows", 1000)
logger=logging.getLogger(__name__)
logging.getLogger("flyhostel.data.interactions.neighbors_gpu").setLevel(logging.WARNING)


def all_id_expected_qc_scene(scene, number_of_animals):
    result=scene.groupby("frame_number").apply(lambda df: all_id_expected_qc(df[["id"]].values, number_of_animals=number_of_animals, idx=0)).all().item()
    return result


def scene_qc(scene, number_of_animals):
    """
        Arguments:
            scene (pd.DataFrame): Dataset of animal positions (centroid_x and centroid_y) over time (frame_number) with identity annotation (id) and fragment annotation (fragment)
            number_of_animals (int)

        Returns:
            min_distance (float): Minimum distance observed between any two animals in the same frame
            gap_n_frames (int): Number of frames where the only broken fragment is broken.
                if infinite, more than 1 fragment is broken, if -1, the scene spans >2 chunks
            gap_distance (float): Pixels traveled by the animal in the frames where the only fragment is broken
                if infinite, more than 1 fragment is broken, if -1, the scene spans >2 chunks
            max_velocity (float): Maximum velocity observed by any animal between two consecutive frames
                if infinite, the number of animals in eacah frame does not match the expectation for at least one frame
            between_chunks (bool): If true, the scene spans >2 chunks
            broken (int): Whether all animals are found in all frames (0) or not (1)
            length (int): Number of frames making up the scene
            maintains_id (int): In scene where only 1 fragment is broken (giving 2 fragments), whether the 2 fragments have the same id or not
            n_failed_fragments (int): Number of fragments that dont span the whole scene,
                which indicates they are broken due to some challenging behavior of the animal 
    """

    all_valid_ids=all_id_expected_qc_scene(scene, number_of_animals)
    min_distance, (focal_id, partner_id)=min_distance_between_animals_qc(scene)
    gap_n_frames, gap_distance, between_chunks, maintains_id, n_failed_fragments=fragment_gap_qc(scene)
    scene_length=len(scene["frame_number"].unique())
    # count how many times each id appears
    counts=scene.value_counts("id")
    try:
        # if all available ids appear in all frames
        if (counts==scene_length).all():
            broken=0
            max_velocity=max_velocity_qc_ideal(scene, number_of_animals)
        else:
            broken=1
            max_velocity=max_velocity_qc_not_ideal(scene)
    except Exception as error:
        print(scene)
        raise error
        
    return {
        "all_valid_ids": all_valid_ids,
        "min_distance": min_distance, "max_velocity": max_velocity,
        "gap_n_frames": gap_n_frames, "gap_distance": gap_distance, "maintains_id": maintains_id,
        "between_chunks": between_chunks, "broken": broken, "length": scene_length,
        "n_failed_fragments": n_failed_fragments,
    }

def min_distance_between_animals_qc(scene):
    if isinstance(scene, cp.ndarray):
        nx=cp
    else:
        nx=np

    ids=scene["id"].unique().tolist()
    distance_matrix=compute_distance_between_ids(scene, ids)
    distance, (i, j) = find_closest_pair(distance_matrix, time_axis=2, partner_axis=1)
    i=i.tolist()
    j=j.tolist()
    k=int(nx.argmin(distance))
    if nx is np:
        min_distance=distance_matrix[k, i[k], j[k]].item()
    elif nx is cp:
        min_distance=distance_matrix[k, i[k], j[k]].get().item()

    focal_id=ids[k]

    other_ids=ids.copy()
    other_ids.pop(ids.index(focal_id))
    partner_id=other_ids[i[k]]

    return min_distance, (focal_id, partner_id)

def max_velocity_qc_ideal(scene, number_of_animals):
    """
    Return the maximum distance traveled by any animal
    between any pair of consecutive frames in the scene 
    """

    scene_length=len(scene["frame_number"].unique())
    coords=scene.sort_values(["frame_number", "id"])[["centroid_x", "centroid_y"]].values
    number_of_found_animals=len(scene["id"].unique())
    if number_of_animals!=number_of_found_animals:
        logger.warning("Number of animals found in scene starting in %s != %s", scene["frame_number"].iloc[0], number_of_animals)
        number_of_animals=number_of_found_animals

    try:
        coords=coords.reshape(scene_length, number_of_animals, -1)
    except Exception as error:
        logger.error("max_velocity_qc_ideal has failed processing scene starting in %s", scene["frame_number"].iloc[0])
        raise error
    distance=((cp.diff(coords, axis=0)**2).sum(axis=2)**0.5)
    return distance.max().get().item()

def max_velocity_qc_not_ideal(scene):
    coords_df=scene.sort_values(["frame_number", "id"])
    max_distance=0
    for id, coords in coords_df.groupby("id"):
        if coords.shape[0]==1:
            continue
        coords=coords[["centroid_x", "centroid_y"]].values
        distance=((cp.diff(coords, axis=0)**2).sum(axis=1)**0.5)
        
        max_distance=max(max_distance, distance.max().get().item())
    return max_distance


def fragment_gap_qc(scene):

    chunks=scene["frame_number"]//chunksize
    number_of_missing_frames=-1
    gap_distance=-1
    maintains_id=None
    between_chunks=None
    
    if len(chunks.unique())>1:
        between_chunks=1
        return number_of_missing_frames, gap_distance, between_chunks,maintains_id, None
    between_chunks=0

    fragments=scene["fragment"].unique()
    frame_numbers=scene["frame_number"].unique()
    scene_length=len(frame_numbers)
    
    fragment_qc=[]
    for fragment_identifier in fragments:
        df_fragment=scene.loc[scene["fragment"]==fragment_identifier]
        n_frames=df_fragment.shape[0]
        distance=(cp.stack([
            cp.diff(df_fragment["centroid_x"]),
            cp.diff(df_fragment["centroid_y"])
        ], axis=1)**2).sum(axis=1)**0.5
    
        fragment_qc.append((
            (scene_length==n_frames),
            distance
        ))
    
    failed_fragments=sorted([fragments[i] for i in range(len(fragments)) if not fragment_qc[i][0]])
    if len(failed_fragments)==2:
        last_observed=scene.loc[scene["fragment"]==failed_fragments[0]].iloc[-1]
        first_observed=scene.loc[scene["fragment"]==failed_fragments[1]].iloc[0]
        number_of_missing_frames=int(first_observed["frame_number"]-last_observed["frame_number"])
        gap_distance=(((last_observed[["centroid_x", "centroid_y"]].values-first_observed[["centroid_x", "centroid_y"]].values)**2).sum()**0.5).item()
        maintains_id=(last_observed["id"]==first_observed["id"]).item()*1
    
        return number_of_missing_frames, gap_distance, between_chunks,maintains_id, 2
    elif len(failed_fragments) == 1:
        maintains_id=None
        number_of_missing_frames=-1
        gap_distance=-1
        return number_of_missing_frames, gap_distance, between_chunks,maintains_id, 1
    elif len(failed_fragments) == 0:
        maintains_id=True
        number_of_missing_frames=0
        gap_distance=0
        return number_of_missing_frames, gap_distance, between_chunks,maintains_id, 0
        
    else:
        maintains_id=False
        # logger.debug(f"scene - {scene['frame_number'].iloc[0]}: {len(failed_fragments)} broken fragments found")
        number_of_missing_frames=math.inf
        gap_distance=math.inf
        return number_of_missing_frames, gap_distance, between_chunks,maintains_id, len(failed_fragments)

def run_qc_of_scene(scene, scene_start, manifest, number_of_animals):
    output_yaml = os.path.join(os.path.dirname(manifest), os.path.splitext(os.path.basename(manifest))[0] + "_qc.yaml")
    logging.getLogger("flyhostel.data.interactions.neighbors_gpu").setLevel(logging.WARNING)
    if os.path.exists(output_yaml):
        with open(output_yaml, "r") as handle:
            result=yaml.load(handle, yaml.SafeLoader)
            
    else:
        result=scene_qc(scene, number_of_animals)
        result["scene_start"]=scene_start
        with open(output_yaml, "w") as handle:
            yaml.dump(result, handle, yaml.SafeDumper)

    return result


def run_qc_of_scene_batch(kwargs_all):
    out=[]
    for kwargs in kwargs_all:
        out.append(
            run_qc_of_scene(**kwargs)
        )
    qc=pd.DataFrame(out)
    return qc

def annotate_scene_quality(experiment, folder, n_jobs=-2, sample_size=None):
    tracking_data=pd.read_feather(f"{folder}/{experiment}_tracking_data.feather")
    
    manifests=sorted(glob.glob(f"{folder}/movies/{experiment}*jsonl"))
    if sample_size is not None:
        manifests=manifests[:sample_size]

    number_of_animals=int(experiment.split("_")[1].rstrip("X"))

    kwargs=[]
    manifests_todo=[]
    for manifest in tqdm(manifests):
        output_yaml = os.path.join(os.path.dirname(manifest), os.path.splitext(os.path.basename(manifest))[0] + "_qc.yaml")
        if os.path.exists(output_yaml):
            continue
        
        scene_start=int(os.path.splitext(manifest)[0].split("_")[-1])
        chunk=scene_start//chunksize
        scene_length=len(os.listdir(f"{folder}/movies/{experiment}_{str(chunk).zfill(6)}_{str(scene_start).zfill(10)}"))

        scene=tracking_data.loc[
            (tracking_data["frame_number"] >= scene_start) & (tracking_data["frame_number"] < scene_start+scene_length),
            ["local_identity", "frame_number", "x","y", "fragment"]
        ]
        scene.columns=["id", "frame_number", "centroid_x", "centroid_y", "fragment"]
        kwargs.append({"scene": scene, "scene_start": scene_start, "manifest": manifest, "number_of_animals": number_of_animals})
        manifests_todo.append(manifest)

    if len(manifests_todo) > 0:

        batches=[]
        batch_size=50
        n_batches=len(manifests_todo)//batch_size + 1

        for j in range(n_batches):
            batch=kwargs[(j*batch_size):(j+1)*batch_size]
            if len(batch)>0:
                batches.append(
                    batch
                )

        qc=joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(run_qc_of_scene_batch)(batch)
            for batch in batches
        )

        qc=pd.concat(qc, axis=0)
        qc.to_csv(os.path.join(folder, "scene_qc.csv"))