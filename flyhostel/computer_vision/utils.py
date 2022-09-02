from genericpath import exists
import os.path
import pickle

import numpy as np
import imgstore
import cv2
import idtrackerai.list_of_blobs
from zt_utils import adjust_to_zt0
from flyhostel.computer_vision.get_files import (
    get_collections_file,
    get_trajectories_file,
    get_video_object
)

from flyhostel.computer_vision.contour import find_contour
from confapp import conf
try:
    import local_settings
    conf += local_settings
except ImportError:
    pass


def obtain_real_frame_number(store, frame_number):
    """
    Needed to deal with bug which causes frame numbers to be skipped in the imgstore index
    when the frame-writing queue is full
    which means the frame_numbers stored by the store's index can be overestimated
    example -> collect 100 frames but only 90 are actually saved: the last frame will be reported as frame #100
    even though it is actually #90
    """

    metadata=store._index.get_all_metadata()
    real_frame_number = metadata['frame_number'].index(frame_number)
    return real_frame_number


def reproduce_example(animal, frame_number, experiment):
    # get the frame
    store = imgstore.new_for_filename(
        os.path.join(conf.VIDEO_FOLDER, experiment)
    )

    frame, (frame_number_, frame_time)  = store.get_image(frame_number)
    assert frame_number_ == frame_number

    real_fn = obtain_real_frame_number(store, frame_number)

    # get the contour and centroid
    chunk = store._chunk_n
    blobs_in_video = idtrackerai.list_of_blobs.ListOfBlobs.load(
        get_collections_file(experiment, chunk)
    ).blobs_in_video
    
    body_size=round(get_video_object(experiment, chunk).median_body_length)

    frame_index = store._get_chunk_metadata(chunk)["frame_number"].index(frame_number)
    trajectories=np.load(get_trajectories_file(experiment))
    blobs_in_frame = blobs_in_video[frame_index]
    tr = trajectories[real_fn, animal, :]            

    centroid = tuple([round(e) for e in tr])
    contour, other_contours = find_contour(blobs_in_frame, centroid)
    filepath="test.png"
    
    return frame, (frame_number, frame_time), chunk, contour, other_contours, centroid, body_size, filepath

def hours(x):
    return x*3600


def package_frame_for_labeling(frame, center, box_size):
    
    # bbox = [tl_x, tl_y, br_x, br_y]
    bbox = [
            center[0] - box_size,
            center[1] - box_size,
            center[0] + box_size,
            center[1] + box_size,
        ]

    bbox = [
        max(0, bbox[0]),
        max(0, bbox[1]),
        min(frame.shape[1], bbox[2]),
        min(frame.shape[0], bbox[3])
    ]
    
    if conf.DEBUG:
        print(f"Final box: {bbox}")
        
    frame=frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    target_height = target_width = box_size*2
    actual_height = (bbox[3]-bbox[1])
    actual_width = (bbox[2]-bbox[0])
    

    # pad with black to ensure all img have equal size
    pad_bottom = round(target_height - actual_height)
    pad_right = round(target_width - actual_width)
    
    if conf.DEBUG:
        print(f"Padding: 0x{pad_bottom}x0x{pad_right}")
    
    frame=cv2.copyMakeBorder(frame, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, 255)
    return frame, bbox


def get_example_animal(experiment, animal=3, frame_number=143363):
    prefix = experiment.replace(os.path.sep,"-")
    example_cache_file=os.path.join("cache", f"{prefix}_{animal}_{str(frame_number).zfill(10)}.pkl") 

    if os.path.exists(example_cache_file):
        with open(example_cache_file, "rb") as filehandle:
            example = pickle.load(filehandle)

    else:
        frame, (frame_number, frame_time), chunk, contour, other_contours, centroid, body_size, filepath=reproduce_example(animal, frame_number, experiment)
        example = {
            "frame": frame,
            "chunk": chunk,
            "contour": contour,
            "other_contours": other_contours,
            "centroid": centroid,
            "body_size": body_size,
            "filepath": filepath,
            "frame_time": frame_time
        }

        os.makedirs(os.path.dirname(example_cache_file), exist_ok=True)
        with open(example_cache_file, "wb") as filehandle:
            pickle.dump(example, filehandle)
        
    return example
