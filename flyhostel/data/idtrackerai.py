import os.path
import logging
import glob
import copy
import warnings

import numpy as np

from flyhostel.data.trajectorytools import get_trajectory_files
from flyhostel.utils import copy_files_to_store
from flyhostel.quantification.imgstore import read_store_description, read_store_metadata
from .blobtools import read_blobs_data
from .trajectorytools import load_trajectories, pad_beginning_so_always_referenced_to_record_start
from .csvtools import read_csv_data

logger = logging.getLogger(__name__)

def copy_idtrackerai_data(imgstore_folder, analysis_folder, allow_wo_gaps=True, interval=None, overwrite=True):
    trajectories_paths = get_trajectory_files(analysis_folder, allow_wo_gaps=allow_wo_gaps)
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



def read_idtrackerai_data(imgstore_folder, pixels_per_cm, interval=None, **kwargs):
    """
    """    
    
    trajectories_paths = sorted(
        glob.glob(os.path.join(imgstore_folder, "*.npy"))
    )

    # Load trajectories
    (status, chunks, tr), (timestamps, missing_timestamps) = load_trajectories(trajectories_paths=trajectories_paths, interval=interval, **kwargs)
    tr.new_length_unit(pixels_per_cm, "cm")
    return (tr, chunks), (timestamps, missing_timestamps)



def read_data(imgstore_folder, interval, interpolate_nans=False, source="trajectories", n_jobs=1):

    store_metadata = read_store_metadata(
        imgstore_folder
    )   
    pixels_per_cm = store_metadata["pixels_per_cm"]

   
    if source=="trajectories":
        (tr, chunks), (timestamps, missing_timestamps) = read_idtrackerai_data(
            imgstore_folder,
            interval=interval,
            pixels_per_cm=pixels_per_cm,
            interpolate_nans=interpolate_nans
            n_jobs=n_jobs
        )

    elif source=="blobs":

        assert "1X" in imgstore_folder

        (tr, chunks), (timestamps, missing_timestamps) = read_blobs_data(
            imgstore_folder,
            interval=interval,
            pixels_per_cm=pixels_per_cm,
            interpolate_nans=interpolate_nans,
            n_jobs=n_jobs
        )

    elif source=="csv":
        (tr, chunks), (timestamps, missing_timestamps) = read_csv_data(
            csv_file=None,
            interval=interval,
            pixels_per_cm=pixels_per_cm
        )
        
    tr_raw = copy.deepcopy(tr)
    tr = pad_beginning_so_always_referenced_to_record_start(tr, missing_timestamps)


    #TODO
    # Eventually trajectorytools should have a chunk key
    # so we know from which chunk each data point comes
    non_nan_rows = tr._s.shape[0] - np.isnan(tr._s).any(2).all(1).sum().item()
    hours_of_data = round(non_nan_rows / tr.params["frame_rate"] / 3600, 2)

    logger.info(f"flyhostel has loaded {hours_of_data} hours of data successfully")
    
    logger.info("Computing velocity")
    velocities = np.sqrt(((tr.v)**2).sum(axis=2))
    # velocities = np.abs(tr.v).sum(axis=2)

    # Load metadata
    _, chunk_metadata = read_store_description(
        imgstore_folder, chunk_numbers=list(range(0, chunks[-1]+1))
        # imgstore_folder, chunk_numbers=chunks
    )
    if not tr.s.shape[0] == len(chunk_metadata[0]):
        min_chunk = min(chunks)
        max_chunk = max(chunks)
        target = set(list(range(min_chunk, max_chunk+1)))
        missing = sorted(list(target-set(chunks)))
        raise ValueError(f"Missing data for chunks {missing}")

    store_metadata["chunks"] = chunks
    return tr, velocities, chunks, store_metadata, chunk_metadata
