import os.path
import logging
import re
import glob
import warnings

import numpy as np
from imgstore.interface import VideoCapture
from imgstore.constants import STORE_MD_FILENAME
from idtrackerai.list_of_blobs import ListOfBlobs
from idtrackerai.crossings_detection.model_area import compute_model_area_and_body_length
from feed_integration.idtrackerai.paths import blobs2trajectories
from trajectorytools.trajectories import import_idtrackerai_dict

logger = logging.getLogger(__name__)

def read_blobs_collection(blobs_path, index, number_of_animals)
   fts = index.get_chunk_metadata(chunk)["frame_number"]

   if chunk in missing_chunks:
       warnings.warn(f"Blobs for chunk {chunk} not found")
       assert number_of_animals == 1
       trajectory = np.array([[
           np.nan, np.nan
       ]] * len(fts)).reshape((-1, number_of_animals, 2))

   else:
       trajectory = blobs2trajectories(
           blobs_path,
           number_of_animals
       )["trajectories"]
   
   
   # frame_times_all.append(fts)
   missing_last_frames =len(fts) -  trajectory.shape[0]
   if missing_last_frames != 0: 
       logger.warning(f"Blobs missing at the end of chunk {chunk}")
       for _ in range(missing_last_frames):
           trajectory=np.vstack([
               trajectory,
               trajectory[-1:]
           ])
   return trajectory
       


    
def read_blobs_data(imgstore_folder, pixels_per_cm, interval=None, n_jobs=1, **kwargs):
    """
    """
    # import ipdb; ipdb.set_trace()

    if interval is None:
        blob_collections = sorted(
            glob.glob(os.path.join(imgstore_folder, "idtrackerai", "session_*", "preprocessing", "blobs_collection.npy"))
        )
        chunks = [int(re.search(f".*{os.path.sep}session_(.*){os.path.sep}preprocessing.*", p).group(1)) for p in blob_collections]
    else:
        chunks = list(range(*interval))
        blob_collections = [
            os.path.join(imgstore_folder, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
            for chunk in chunks
        ]


    missing_chunks=[]
    for chunk in range(min(chunks), max(chunks)+1):
        blob_path = os.path.join(imgstore_folder, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
        if not os.path.exists(blob_path):
            missing_chunks.append(chunk)

    session_folder=os.path.dirname(os.path.dirname(blob_collections[0]))

    video=np.load(os.path.join(session_folder, "video_object.npy"), allow_pickle=True).item()
    number_of_animals=video._user_defined_parameters["number_of_animals"]


    if missing_chunks and number_of_animals != 1:
        raise ValueError(f"Chunks missing {' '.join([str(c) for c in missing_chunks])}")    
    
    store=VideoCapture(os.path.join(imgstore_folder, STORE_MD_FILENAME), chunk=chunks[0])
    frame_times = store._index.get_timestamps(chunks)
    timestamps = np.array([row[0] for row in frame_times]) / 1000 # ms to s
    missing_frame_times = store._index.get_timestamps(list(range(chunks[0])))
    missing_timestamps = np.array([row[0] for row in missing_frame_times]) / 1000 # ms to s


    trajectories=joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(read_blobs_collection)(
            blob_collections[i], store._index, video.user_defined_parameters["number_of_animals"]
        )
        for i, chunk in enumerate(chunks)
    )

    #for i, chunk in enumerate(chunks):
    #    blobs_path = blob_collections[i]
    #    trajectory=read_blobs_collection(blobs_path, store._index, video.user_defined_parameters["number_of_animals"])
    #    trajectories.append(trajectory)

    trajectories = np.vstack(trajectories)
    # frame_times_all = np.stack(frame_times_all)
    
    try:
        median_body_length_full_resolution=video.median_body_length_full_resolution
    except:
        logger.debug("Video has not defined median_body_length_full_resolution")
        list_of_blobs = ListOfBlobs.load(blob_collections[0])
        median_body_length_full_resolution=compute_model_area_and_body_length(list_of_blobs, video.user_defined_parameters["number_of_animals"])[1]

    traj_dict = {
        "trajectories": trajectories,
        "frames_per_second": video.frames_per_second,
        "body_length": median_body_length_full_resolution,
        "chunks": chunks,
    }

    assert len(timestamps) == traj_dict["trajectories"].shape[0], f"{len(timestamps)} != {traj_dict['trajectories'].shape[0]}"
   
    tr=import_idtrackerai_dict(traj_dict, timestamps=timestamps)
    tr.new_length_unit(pixels_per_cm, "cm")
    return (tr, chunks), (timestamps, missing_timestamps)

