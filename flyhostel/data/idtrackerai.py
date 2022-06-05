import argparse
import os.path
import logging
import glob
import warnings
import functools

import numpy as np

from flyhostel.data.trajectorytools import get_trajectory_files
from flyhostel.utils import copy_files_to_store
from flyhostel.quantification.imgstore import read_store_description, read_store_metadata
from .trajectorytools import load_trajectories

logger = logging.getLogger(__name__)

def copy_idtrackerai_data(imgstore_folder, analysis_folder, overwrite=True):
    trajectories_paths = get_trajectory_files(analysis_folder)
    if not trajectories_paths:
        warnings.warn(f"No trajectory files found in {analysis_folder}")
    copy_files_to_store(imgstore_folder, trajectories_paths, overwrite=overwrite)

    

def copy(args=None, ap=None):

    if args is None:
        if ap is None:
            ap = argparse.ArgumentParser()

        ap.add_argument(
            "--imgstore-folder", dest="imgstore_folder", required=True, type=str
        )

        ap.add_argument(
            "--analysis-folder", dest="analysis_folder", default=None, type=str
        )
        ap.add_argument(
            "--overwrite", action="store_true", default=True,
            help="Makes a new copy of the idtrackerai's trajectory file"
            " even if a copy already exists, overwriting it"
            )
        
        args = ap.parse_args()

    copy_idtrackerai_data(args.imgstore_folder, args.analysis_folder, args.overwrite)


#@functools.lru_cache(maxsize=100, typed=False)
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

def read_data(imgstore_folder, interval, interpolate_nans=False, from_idtrackerai=True):

    store_metadata = read_store_metadata(
        imgstore_folder
    )   
    pixels_per_cm = store_metadata["pixels_per_cm"]

   
    if from_idtrackerai:
        tr, chunks = read_idtrackerai_data(
            imgstore_folder,
            interval=interval,
            pixels_per_cm=pixels_per_cm,
            interpolate_nans=interpolate_nans
        )

    else:
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
