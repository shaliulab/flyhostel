"""
Quickly obtain an overview of the status of the pipeline for different experiments
"""

import os.path
import pickle
import glob
import os
import sqlite3

import yaml
import numpy as np
import cv2
import h5py

def validate_file_exists_and_its_non_zero(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def validate_ai_file(path):
    
    validated=os.path.exists(path)
    if not validated:
        return validated
    
    try:
        with open(path, "rb") as filehandle:
            pickle.load(filehandle)
        
        return True
    except:
        return False


def validate_idtrackerai_preprocessing(flyhostel_id, number_of_animals, date_time, chunk_start, chunk_end):

    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], f"FlyHostel{flyhostel_id}", f"{number_of_animals}X", date_time)
    validated=True

    for chunk in range(chunk_start, chunk_end+1):
        blobs_collection=os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
        validated=validated and validate_file_exists_and_its_non_zero(blobs_collection)
        if not validated:
            break
    
    return validated


def validate_idtrackerai_integration(flyhostel_id, number_of_animals, date_time, chunk_start, chunk_end):
    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], f"FlyHostel{flyhostel_id}", f"{number_of_animals}X", date_time)
    validated=True

    for chunk in range(chunk_start, chunk_end+1):
        ai_file=os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "ai.pkl")
        validated=validated and validate_ai_file(ai_file)
        if not validated:
            break
    
    return validated


def validate_idtrackerai_crossings_detection_and_fragmentation(flyhostel_id, number_of_animals, date_time, chunk_start, chunk_end):
    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], f"FlyHostel{flyhostel_id}", f"{number_of_animals}X", date_time)
    validated=True

    for chunk in range(chunk_start, chunk_end+1):
        fragments_file=os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "fragments.npy")
        validated=validated and validate_file_exists_and_its_non_zero(fragments_file)
        if not validated:
            break
    
    return validated


def validate_idtrackerai_tracking(flyhostel_id, number_of_animals, date_time, chunk_start, chunk_end):
    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], f"FlyHostel{flyhostel_id}", f"{number_of_animals}X", date_time)
    validated=True

    for chunk in range(chunk_start, chunk_end+1):
        trrajectories_file=os.path.join(basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "trrajectories", "trrajectories.npy")
        validated=validated and validate_file_exists_and_its_non_zero(trrajectories_file)
        if not validated:
            break
    
    return validated


def validate_flyhostel_export(flyhostel_id, number_of_animals, date_time, chunk_start, chunk_end):

    results_folder = glob.glob(os.path.join(os.environ["FLYHOSTEL_RESULTS"], "*"))[0]
    filename=f"FlyHostel{flyhostel_id}_{number_of_animals}X_{date_time}.db"
    dbfile = os.path.join(results_folder, f"FlyHostel{flyhostel_id}", f"{number_of_animals}X", date_time, filename)

    validated = validate_file_exists_and_its_non_zero(dbfile)
    if not validated:
        return validated

    with sqlite3.connect(dbfile, check_same_thread=False) as conn:
        cur = conn.cursor()
        
        cur.execute("SELECT value FROM METADATA WHERE field = 'chunks';")
        chunks=[int(x) for x in cur.fetchone().split(",")]

        validated = chunks[0]<=chunk_start and chunks[1]>=chunk_end
        if not validated:
            return validated

        
        try:
            cur.execute("SELECT COUNT(*) FROM BEHAVIORS LIMIT 1;")
            count=cur.fetchone()[0]
            if count > 0:
                return 2

        except sqlite3.OperationalError:
            return validated


def validate_video(path, expected_frame_count):
    
    cap = cv2.VideoCapture(path)
    pos=cap.get(1)

    ret, frame = cap.read()

    validated = pos == 0 and cap.get(1) == 1 and cap.get(7) == expected_frame_count and ret is not None and isinstance(frame, np.ndarray)
    cap.release()
    return validated


def validate_flyhostel_make_video(flyhostel_id, number_of_animals, date_time, chunk_start, chunk_end):

    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], f"FlyHostel{flyhostel_id}", f"{number_of_animals}X", date_time)
    store_folder = os.path.join(basedir, "flyhostel", "single_animal")
    metadata_file=os.path.join(store_folder, "metadata.yaml")
    
    validated = os.path.exists(metadata_file)
    with open(metadata_file, "r") as filehandle:
        metadata=yaml.load(filehandle, yaml.SafeLoader)
    
    chunksize=int(metadata["_store"]["chunksize"])

    if not validated:
        return validated

    for chunk in range(chunk_start, chunk_end+1):
        video_file = os.path.join(store_folder, f"{str(chunk).zfill(6)}.mp4")
        validated = validated and validate_video(video_file, expected_frame_count=chunksize)
        if not validated:
            return validated


def validate_deepethogram_prediction(flyhostel_id, number_of_animals, date_time, chunk_start, chunk_end):

    data_dir = os.path.join(os.environ["DEEPETHOGRAM_PROJECT_PATH"], "DATA")
    validated=True

    for chunk in range(chunk_start, chunk_end+1):
        key=f"FlyHostel{flyhostel_id}_{number_of_animals}X_{date_time}_{str(chunk).zfill(6)}"
        filename=f"{key}_00.h5" 
        output_file=os.path.join(data_dir, key, filename)
        with h5py.File(output_file) as f:
            try:
                f["resnet18"]
            except:
                validated=False
                break

    return validated