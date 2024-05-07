import os.path
import itertools
import logging
import h5py
import joblib
import git

logger=logging.getLogger(__name__)

import numpy as np
import pandas as pd
from flyhostel.data.pose.constants import DEG_DATA

from flyhostel.data.pose.constants import chunksize, bodyparts_xy
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.main import FlyHostelLoader
from motionmapperpy import setRunParameters

from flyhostel.data.pose.distances import compute_distance_features_pairs
from flyhostel.data.pose.ethogram.utils import annotate_bout_duration, annotate_bouts

wavelet_downsample=setRunParameters().wavelet_downsample
DEFAULT_FILTERS="rle-jump"
DISTANCE_FEATURES_PAIRS=[("head", "proboscis"),]

idx_cols=["id", "local_identity", "frame_number"]


def get_dataset_version():
    repo = git.Repo(os.path.join(DEG_DATA, ".."))
    hexhash=repo.head.object.hexsha
    return hexhash


def get_pose_model_version():
    config_file="/home/vibflysleep/opt/vsc-scripts/nextflow/pipelines/pose_estimation/nextflow.config"
    with open(config_file, "r") as handle:
        config=handle.readlines()
    model_name=[line for line in config if "sleap_model" in line][-1].strip("\n").split("/models/")[-1].rstrip('"')
    return model_name

def document_provenance():

    model_name=get_pose_model_version()
    deg_data_version=get_dataset_version()

    return {
        "SLEAP_model_name": model_name,
        "DEG_data_version": deg_data_version, 
    }


def compute_distance(pose, bodyparts_xy, time_window_length=1, framerate=FRAMERATE//wavelet_downsample, FUN="sum"):
    
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
    if loader.deg is None:
        logger.warning("%s does not have DEG data", loader)
        return
    loader.deg.loc[loader.deg["behavior"]=="groom+micromovement", "behavior"]="groom"
    loader.deg.loc[loader.deg["behavior"]=="feed+walk", "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=="feed+groom", "behavior"]="groom"
    loader.deg.loc[loader.deg["behavior"]=="feed+micromovement", "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=="pe", "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=="pe+walk", "behavior"]="feed"
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
    loader.deg.loc[loader.deg["behavior"]=='feed+inactive+turn', "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=='feed+inactive+turn+twitch', "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=='feed+inactive+twitch', "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=="feed+inactive+micromovement+twitch", "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=="feed+inactive+micromovement", "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=='inactive+micromovement+twitch', "behavior"]="inactive+twitch"
    loader.deg.loc[loader.deg["behavior"]=='feed+twitch', "behavior"]="feed"
    loader.deg.loc[loader.deg["behavior"]=='inactive+micromovement+turn', "behavior"]="inactive+turn"
    loader.deg.loc[loader.deg["behavior"]=='inactive+micromovement+turn+twitch', "behavior"]="inactive+turn"
    loader.deg.loc[loader.deg["behavior"]=='micromovement+twitch', "behavior"]="background"

    loader.deg.sort_values("frame_number", inplace=True)
    loader.deg=annotate_bouts(loader.deg, variable="behavior")
    loader.deg=annotate_bout_duration(loader.deg, fps=150)
    loader.deg["score"]=None
    loader.deg=annotate_bouts(loader.deg, variable="behavior")
    loader.deg=annotate_bout_duration(loader.deg, fps=30)
    loader.deg["label"]=loader.deg["behavior"]
    loader.deg.loc[
        loader.deg["behavior"].isin(["inactive+micromovement", "inactive+turn", "inactive+twitch"]),
        "label"
    ]="micromovement"

    
    
def load_scores(loader):
    hdf5_file=os.path.join(loader.basedir, f"motionmapper/{str(loader.identity).zfill(2)}/pose_raw/{loader.experiment}__{str(loader.identity).zfill(2)}/{loader.experiment}__{str(loader.identity).zfill(2)}.h5")
    with h5py.File(hdf5_file, "r") as file:
        scores=pd.DataFrame(file["point_scores"][0, :, :].T)
        node_names=[e.decode() for e in file["node_names"][:]]
        scores.columns=node_names

    loader.scores=scores


def compute_proboscis_visibility_timing(pose):
    """
    Annotate for how long has the proboscis been visible
    and for how long it will be.
    These features may help distinguish feed from PE    
    """
    logger.warning("compute_proboscis_visibility_timing is not implemented. Skipping")
    return pose




def load_pose(loader, chunksize, files=None, frame_numbers=None, load_distance_travelled_features=False, load_inter_bp_distance_features=True, annotate_proboscis_visibility_timings=True, downsample=1, filters="rle-jump"):
    """
    Populate loader.pose and loader.first_fn
    """

    if files is None:
        pose_folder=f"motionmapper/{str(loader.identity).zfill(2)}/pose_filter_{filters}"
        hdf5_file=os.path.join(loader.basedir, f"{pose_folder}/{loader.experiment}__{str(loader.identity).zfill(2)}/{loader.experiment}__{str(loader.identity).zfill(2)}.h5")
    else:
        hdf5_file=files[0]

    try:
        with h5py.File(hdf5_file, "r") as file:
            files=sorted([e.decode() for e in file["files"][:]], key=lambda x: os.path.basename(x))
            chunks=[int(os.path.basename(file).split(".")[0]) for file in files]
            local_identities=[int(os.path.basename(os.path.dirname(file))) for file in files]
            frame_number_available=np.array(list(itertools.chain(*[(np.arange(0, chunksize)+chunksize*chunk).tolist() for chunk in chunks])))
            first_fn=frame_number_available[0]
            loader.first_fn=first_fn

            if frame_numbers is None:
                frames=None
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

            local_identities=np.array(list(itertools.chain(*[[local_identities[i],]*chunksize for i, chunk in enumerate(chunks)])))
            if frames is not None:
                local_identities=local_identities[frames]
    
    except Exception as error:
        logger.error("Cannot open %s", hdf5_file)
        raise error


    columns = list(itertools.chain(*[[bp +"_x", bp+"_y"] for bp in node_names]))
    if load_distance_travelled_features:
        logger.debug("Adding distance features %s", pose.shape)
        pose, bodyparts_distance=compute_distance(pose, bodyparts_xy, framerate=150, time_window_length=.2)
        columns+=bodyparts_distance
        logger.debug("Done %s", pose.shape)

    pose.columns=columns
    pose["local_identity"]=local_identities
    pose["frame_number"]=frame_numbers
    pose=pose.loc[pose["frame_number"]%downsample==0]
    frame_numbers=frame_numbers[frame_numbers%downsample==0]
    pose["id"]=loader.ids[0]

    if annotate_proboscis_visibility_timings:
        pose=compute_proboscis_visibility_timing(pose)


    if load_inter_bp_distance_features:
        pose=compute_distance_features_pairs(pose, DISTANCE_FEATURES_PAIRS)
        pose.loc[pose["head_proboscis_distance"].isna(), "head_proboscis_distance"]=0

    # pose.set_index(idx_cols, inplace=True)
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


def load_wavelets(loader, filters, frames, wavelet_file=None):
    assert loader.pose is not None
    if wavelet_file is None:
        wavelet_file=select_wavelet_file(loader, filters=filters)
    wavelets, (frequencies, freq_names)=loader.load_wavelets(matfile=wavelet_file, frames=frames)
    loader.wavelets=wavelets


def load_animal_data(
        loader, filters,
        load_scores_data=False,
        load_deg_data=True,
        load_pose_data=True,
        load_wavelets_data=True,
        wavelet_file=None,
        files=None,
        chunksize=45000,
        downsample=5
    ):
    """
    Load pose, deg labels and wavelets for a single animal
    
    Populate loader.deg, loader.pose and loader.wavelets

    If load_wavelets_data is True, load_pose_data must be True
    """

    
    if load_wavelets_data:
        assert load_pose_data

    if load_scores_data:
        load_scores(loader)

    if load_deg_data:
        load_deg(loader)
    
    if load_deg_data:
        if loader.deg is None:
            return loader
        else:
            labeled_frames=loader.deg["frame_number"].values
            # wt = wavelet transform
            wt_frames=labeled_frames[labeled_frames % downsample==0] // downsample
            labeled_frames_wt=None
        
    else:
        labeled_frames=None
        wt_frames=None
        labeled_frames_wt=None

    if load_pose_data:
        load_pose(loader, chunksize, files=files, filters=filters, downsample=downsample, frame_numbers=labeled_frames)
        if load_scores_data:
            loader.scores.index=loader.pose.index
            loader.pose=pd.concat([loader.pose, loader.scores], axis=1)
    if load_wavelets_data:
        if wt_frames is None:
            frames=None
        else:
            frames=wt_frames- (loader.first_fn // downsample)

        load_wavelets(loader, filters=filters, frames=frames, wavelet_file=wavelet_file)

        if labeled_frames_wt is None:
            frame_numbers=loader.pose["frame_number"].values
        else:
            frame_numbers=labeled_frames_wt

        loader.wavelets["frame_number"]=frame_numbers

    return loader


def compile_dataset(loader, out=None):
    
    if loader.deg is None:
        loader.data=loader.pose.copy()
    elif loader.pose is None:
        loader.data=loader.deg
    else:        
        loader.data=loader.deg.merge(loader.pose, on=idx_cols, how="inner")
    
    # import ipdb; ipdb.set_trace()

    if loader.wavelets is None:
        pass
    else:
        wavelets=loader.wavelets.reset_index()
        loader.data=loader.data.merge(wavelets, on=["id", "frame_number"], how="inner")
    if out is not None:
        loader.data_path=out
        if loader.data.shape[0]>0:
            loader.data.to_feather(out)
    
    return loader.data


def process_animal(
        loader, cache=None, refresh_cache=True, filters=DEFAULT_FILTERS, downsample=wavelet_downsample, files =None, wavelet_file=None,
        load_deg_data=True, load_pose_data=True, load_wavelets_data=True, on_fail="raise"
    ):
    try:
        if cache is None:
            must_load=True
            out=None
        else:
            out=f"{cache}/{loader.experiment}__{str(loader.identity).zfill(2)}_ml_dataset.feather"
            if not refresh_cache and os.path.exists(out):
                logger.debug("Loading %s", out)
                data=pd.read_feather(out)
                loader.data=data
                must_load=False
            else:
                must_load=True

        if must_load:
            loader=load_animal_data(
                loader,
                load_deg_data=load_deg_data, load_pose_data=load_pose_data, load_wavelets_data=load_wavelets_data,
                filters=filters, downsample=downsample, chunksize=chunksize,
                wavelet_file=wavelet_file, files=files,
            )
            if load_deg_data and loader.deg is None:
                logger.warning("Data could not be loaded for %s", loader)
                return None
            data=compile_dataset(loader, out)
                
        if "t" not in data.columns:
            loader.load_store_index(cache=cache)
            loader.store_index["t"]=loader.store_index["frame_time"]+loader.meta_info["t_after_ref"]
            data=data.merge(loader.store_index[["frame_number", "t"]], on="frame_number")

        # data.set_index(["id", "frame_number"], inplace=True)
        print(f"Loading dataset of shape {data.shape} for animal {loader.experiment}__{str(loader.identity).zfill(2)}")
        return data
    
    except Exception as error:
        if on_fail=="raise":
            raise error
        elif on_fail=="ignore":
            logger.error(error)
            return None

def load_animals(animals, cache=None, refresh_cache=True, filters=DEFAULT_FILTERS, n_jobs=1, **kwargs):
    loaders=[FlyHostelLoader(experiment=expid.split("__")[0], identity=int(expid.split("__")[1]), chunks=range(0, 400)) for expid in animals]
    data = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            process_animal
        )(
            loader, filters=filters, cache=cache, refresh_cache=refresh_cache, **kwargs
        )
        for loader in loaders
    )
    data=[d for d in data if d is not None]
    if data:
        data=pd.concat(data, axis=0)
    return data