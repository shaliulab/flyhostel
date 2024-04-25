import glob
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
from sklearn.ensemble import RandomForestClassifier

pd.set_option("display.max_rows", 1200)
from flyhostel.data.pose.constants import get_bodyparts, chunksize, framerate, inactive_states, bodyparts, bodyparts_xy
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.main import FlyHostelLoader
from motionmapperpy import setRunParameters

from sklearn.metrics import ConfusionMatrixDisplay
from flyhostel.data.pose.distances import compute_distance_features_pairs
from sklearn.utils.class_weight import compute_class_weight
from flyhostel.data.pose.ethogram.utils import annotate_bout_duration, annotate_bouts
from sklearn.preprocessing import StandardScaler


wavelet_downsample=setRunParameters().wavelet_downsample
filters="rle-jump"



def compute_distance(pose, bodyparts_xy, time_window_length=1, framerate=30, FUN="sum"):
    
    k=int(time_window_length*framerate)
    
    arr=pose[bodyparts_xy].values.reshape((pose.shape[0], -1, 2))
    dist=np.sqrt(np.sum(np.diff(arr, axis=0)**2, axis=2))
    dist=np.concatenate([
        np.array([[0,] * dist.shape[1]]),
        dist
    ])
    dist[np.isnan(dist)]=0
    bodyparts_distance=[bp.replace("_x", "_distance") for bp in bodyparts_xy[::2]]

    dist_res=dist.reshape((dist.shape[0]//k, k, dist.shape[1]))
    summed_array = getattr(dist_res, FUN)(axis=1)
    final_array = np.repeat(summed_array, k, axis=0)

    
    dist_df=pd.DataFrame(final_array, columns=bodyparts_distance, index=pose.index)
    pose=pd.concat([pose, dist_df], axis=1)
    return pose, bodyparts_distance
         


def load_deg(loader):
    loader.deg=None
    loader.load_deg_data(verbose=False)
    loader.deg.loc[loader.deg["behavior"]=="groom+micromovement", "behavior"]="groom"
    loader.deg.loc[loader.deg["behavior"]=="feed+walk", "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=="feed+groom", "behavior"]="groom"
    loader.deg.loc[loader.deg["behavior"]=="feed+micromovement", "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=="pe", "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=="groom+pe", "behavior"]="groom"
    loader.deg.loc[loader.deg["behavior"]=="inactive+turn+twitch", "behavior"]="inactive+turn"
    loader.deg.loc[loader.deg["behavior"]=="twitch", "behavior"]="background"
    loader.deg.loc[loader.deg["behavior"]=="turn", "behavior"]="background"    
    loader.deg.loc[loader.deg["behavior"]=="micromovement", "behavior"]="background"
    loader.deg.loc[loader.deg["behavior"]=="inactive+pe+turn+twitch", "behavior"]="inactive+turn"
    loader.deg.loc[loader.deg["behavior"]=='inactive+pe+twitch', "behavior"]="inactive+pe"
    loader.deg.loc[loader.deg["behavior"]=='inactive+pe+turn', "behavior"]="inactive+pe"
    loader.deg.loc[loader.deg["behavior"]=='turn+twitch', "behavior"]="background"
    loader.deg.loc[loader.deg["behavior"]=='inactive+micromovement+pe', "behavior"]="inactive+pe"
    loader.deg.loc[loader.deg["behavior"]=='feed+inactive', "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=='inactive+micromovement+twitch', "behavior"]="inactive+twitch"
    loader.deg.loc[loader.deg["behavior"]=='inactive+micromovement+turn', "behavior"]="inactive+turn"
    loader.deg.loc[loader.deg["behavior"]=='inactive+micromovement+turn+twitch', "behavior"]="inactive+turn"
    

    loader.deg.sort_values("frame_number", inplace=True)
    loader.deg=annotate_bouts(loader.deg, variable="behavior")
    loader.deg=annotate_bout_duration(loader.deg, fps=150)
    loader.deg["t"]=loader.deg["frame_number"]/150
    loader.deg["score"]=None
    assert loader.deg is not None
    
    
def load_scores(loader):
    hdf5_file=os.path.join(loader.basedir, f"motionmapper/{str(loader.identity).zfill(2)}/pose_raw/{loader.experiment}__{str(loader.identity).zfill(2)}/{loader.experiment}__{str(loader.identity).zfill(2)}.h5")
    with h5py.File(hdf5_file, "r") as file:
        scores=pd.DataFrame(file["point_scores"][0, :, :].T)
        node_names=[e.decode() for e in file["node_names"][:]]
        scores.columns=node_names

    loader.scores=scores


def load_pose(loader, chunksize, frame_numbers=None, load_distance_travelled_features=False, load_inter_bp_distance_features=True, downsample=1, filters="rle-jump"):
    """
    Populate loader.pose and loader.first_fn
    """

    pose_folder=f"motionmapper/{str(loader.identity).zfill(2)}/pose_filter_{filters}"
    hdf5_file=os.path.join(loader.basedir, f"{pose_folder}/{loader.experiment}__{str(loader.identity).zfill(2)}/{loader.experiment}__{str(loader.identity).zfill(2)}.h5")
    with h5py.File(hdf5_file, "r") as file:
        files=[e.decode() for e in file["files"][:]]
        chunks=[int(os.path.basename(file).split(".")[0]) for file in files]
        frame_number_available=np.array(list(itertools.chain(*[(np.arange(0, chunksize)+chunksize*chunk).tolist() for chunk in chunks])))
        first_fn=frame_number_available[0]
        loader.first_fn=first_fn

        if frame_numbers is None:
            pose_r=file["tracks"][0, :, :, :]
        else:
            frames=frame_numbers-first_fn
            pose_r=file["tracks"][0, :, :, frames]

        m=pose_r.shape[2]
        pose=pose_r.transpose(2, 1, 0).reshape((m, -1))
        pose=pd.DataFrame(pose)
        
        node_names=[e.decode() for e in file["node_names"][:]]
        pose.columns=list(itertools.chain(*[[bp +"_x", bp+"_y"] for bp in node_names]))
        
        if frame_numbers is None:
            frame_numbers=np.array(list(itertools.chain(*[(np.arange(0, chunksize)+chunksize*chunk).tolist() for chunk in chunks])))

    columns = list(itertools.chain(*[[bp +"_x", bp+"_y"] for bp in node_names]))
    if load_distance_travelled_features:
        logger.debug("Adding distance features %s", pose.shape)
        pose, bodyparts_distance=compute_distance(pose, bodyparts_xy, framerate=150, time_window_length=.2)
        columns+=bodyparts_distance
        logger.debug("Done %s", pose.shape)

    pose.columns=columns
    pose["frame_number"]=frame_numbers
    pose=pose.loc[pose["frame_number"]%downsample==0]
    frame_numbers=frame_numbers[frame_numbers%downsample==0]
    pose["id"]=loader.ids[0]

    if load_inter_bp_distance_features:
        pose=compute_distance_features_pairs(pose, [("head", "proboscis"),])
        pose.loc[pose["head_proboscis_distance"].isna(), "head_proboscis_distance"]=0

    pose.set_index(["id", "frame_number"], inplace=True)    
    loader.pose=pose

def select_wavelet_file(loader, filters):
    wavelets_folder=os.path.join(loader.basedir, "motionmapper", str(loader.identity).zfill(2), f"wavelets_{filters}", "FlyHostel_long_timescale_analysis", "Wavelets")
    wavelet_file=os.path.join(wavelets_folder, loader.experiment + "__" + str(loader.identity).zfill(2) + "-pcaModes-wavelets.mat")

    if not os.path.exists(wavelet_file):
        logger.error(f"{wavelet_file} not found")
        return None
        
    try:
        with h5py.File(wavelet_file, "r") as file:
            file.keys()
            logger.debug(f"{wavelet_file} OK")
            return wavelet_file
        
    except OSError:
        logger.error(f"{wavelet_file} not readable")
        return None


def load_wavelets(loader, filters, frames):
    assert loader.pose is not None
    wavelet_file=select_wavelet_file(loader, filters=filters)
    wavelets, (frequencies, freq_names)=loader.load_wavelets(matfile=wavelet_file, frames=frames)
    loader.wavelets=wavelets


def load_animal_data(loader, filters, load_scores_data=False, load_deg_data=True, load_pose_data=True, load_wavelets_data=True, chunksize=45000, downsample=5):
    print(loader.basedir)
    if load_scores_data:
        load_scores(loader)

    if load_deg_data:
        load_deg(loader)
        labeled_frames=loader.deg["frame_number"].values
        labeled_frames_wt=labeled_frames[labeled_frames%downsample==0]
        wt_frames=labeled_frames_wt // downsample


    if load_pose_data:
        load_pose(loader, chunksize, filters=filters, downsample=downsample, frame_numbers=labeled_frames)
        if load_scores_data:
            loader.scores.index=loader.pose.index
            loader.pose=pd.concat([loader.pose, loader.scores], axis=1)
    if load_wavelets_data:
        frames=wt_frames- (loader.first_fn // downsample)
        # import ipdb; ipdb.set_trace()
        load_wavelets(loader, filters=filters, frames=frames)
        loader.wavelets["frame_number"]=labeled_frames_wt

    return loader


def compile_dataset(loader, out):
    if os.path.exists(out):
        loader.data=pd.read_feather(out)
        return loader.data
    
    deg=loader.deg.set_index(["id", "frame_number"])
    loader.data=deg.merge(loader.pose, left_index=True, right_index=True, how="inner")
    wavelets=loader.wavelets.set_index("frame_number", append=True)
    loader.data=loader.data.merge(wavelets, left_index=True, right_index=True, how="inner").reset_index()
    loader.data_path=out
    loader.data.to_feather(out)
    return loader.data


def process_animal(loader):
    out=out=f"datasets/{loader.experiment}__{str(loader.identity).zfill(2)}_ml_dataset.feather"
    if os.path.exists(out):
        data=compile_dataset(loader, out)
        
    else:
        loader=load_animal_data(
            loader,
            load_deg_data=True, load_pose_data=True, load_wavelets_data=True,
            filters=filters, downsample=wavelet_downsample, chunksize=chunksize
        )
        data=compile_dataset(loader, out)

    data=data.reset_index()
    data=annotate_bouts(data, variable="behavior")
    data=annotate_bout_duration(data, fps=30)
    data.set_index(["id", "frame_number"], inplace=True)
    
    data["label"]=data["behavior"]
    data.loc[
        data["behavior"].isin(["inactive+micromovement", "inactive+turn", "inactive+twitch"]),
        "label"
    ]="micromovement"

    return data 

def load_animals(animals, n_jobs):
    loaders=[FlyHostelLoader(experiment=expid.split("__")[0], identity=int(expid.split("__")[1]), chunks=range(0, 400)) for expid in animals]
    data = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            process_animal
        )(
            loader
        )
        for loader in loaders
    )
    data=pd.concat(data, axis=0)
    return data