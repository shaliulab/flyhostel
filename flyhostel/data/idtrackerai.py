import os.path
import logging
import glob
import warnings
import functools
import re

import numpy as np

from flyhostel.data.trajectorytools import get_trajectory_files
from flyhostel.utils import copy_files_to_store
from flyhostel.quantification.imgstore import read_store_description, read_store_metadata
from .trajectorytools import load_trajectories
from feed_integration.idtrackerai.paths import blobs2trajectories
from trajectorytools.trajectories import import_idtrackerai_dict
logger = logging.getLogger(__name__)

def copy_idtrackerai_data(imgstore_folder, analysis_folder, interval=None, overwrite=True):
    trajectories_paths = get_trajectory_files(analysis_folder)
    if interval is not None:
        sessions = [f"session_{str(chunk).zfill(6)}" for chunk in range(*interval)]
        files = []
        for file in trajectories_paths:
            for session in sessions:
                if session in file:
                    files.append(file)
    
    else:
        files = trajectories_paths
    if not trajectories_paths:
        warnings.warn(f"No trajectory files found in {analysis_folder}")
    copy_files_to_store(imgstore_folder, files, overwrite=overwrite)

    
def read_blobs_data(imgstore_folder, pixels_per_cm, interval=None, **kwargs):
    """
    """
    
    blob_collections = sorted(
        glob.glob(os.path.join(imgstore_folder, "idtrackerai", "session_*", "preprocessing", "blobs_collection.npy"))
    )
    chunks = [int(re.search(f".*{os.path.sep}session_(.*){os.path.sep}preprocessing.*", p).group(1)) for p in blob_collections]
    if interval is not None:
        indices = (chunks.index(interval[0]), chunks.index(interval[1]-1))
        chunks = chunks[indices[0]:indices[1]+1]
        blob_collections = blob_collections[indices[0]:indices[1]+1]
        
    session_folder=os.path.dirname(os.path.dirname(blob_collections[0]))

    
    video=np.load(os.path.join(session_folder, "video_object.npy"), allow_pickle=True).item()

    traj_dict = {
        "trajectories": np.vstack([blobs2trajectories(
            blobs_path,
            video.user_defined_parameters["number_of_animals"]
        )["trajectories"] for blobs_path in blob_collections]),
        "frames_per_second": video.frames_per_second,
        "body_length": video.median_body_length_full_resolution,
    }

    tr=import_idtrackerai_dict(traj_dict)
    tr.new_length_unit(pixels_per_cm, "cm")

    return (tr, chunks)



def read_idtrackerai_data(imgstore_folder, pixels_per_cm, interval=None, **kwargs):
    """
    """    
    
    trajectories_paths = sorted(
        glob.glob(os.path.join(imgstore_folder, "*.npy"))
    )
    timestamps_paths = sorted(
        glob.glob(os.path.join(imgstore_folder, "*.npz"))
    )
  
    # Load trajectories
    status, chunks, tr = load_trajectories(trajectories_paths=trajectories_paths, timestamps_paths=timestamps_paths, interval=interval, **kwargs)
    tr.new_length_unit(pixels_per_cm, "cm")
    return (tr, chunks)



def read_csv_data(csv_file, pixels_per_cm, interval=None):
    raise NotImplementedError

def read_data(imgstore_folder, interval, interpolate_nans=False, source="trajectories"):

    store_metadata = read_store_metadata(
        imgstore_folder
    )   
    pixels_per_cm = store_metadata["pixels_per_cm"]

   
    if source=="trajectories":
        tr, chunks = read_idtrackerai_data(
            imgstore_folder,
            interval=interval,
            pixels_per_cm=pixels_per_cm,
            interpolate_nans=interpolate_nans
        )

    elif source=="blobs":

        assert "1X" in imgstore_folder

        tr, chunks = read_blobs_data(
            imgstore_folder,
            interval=interval,
            pixels_per_cm=pixels_per_cm,
            interpolate_nans=interpolate_nans
        )

    elif source=="csv":
        tr, chunks = read_csv_data(
            csv_file=None,
            interval=interval,
            pixels_per_cm=pixels_per_cm
        )

    logger.info("Computing velocity")
    velocities = np.abs(tr.v).sum(axis=2)

    # Load metadata
    chunks, chunk_metadata = read_store_description(
        imgstore_folder, chunk_numbers=chunks
    )

    store_metadata["chunks"] = chunks
    return tr, velocities, chunks, store_metadata, chunk_metadata
