import argparse
import os.path
import glob
import re

import joblib
import h5py
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from yolov7tools.utils import load_detection


def filter_by_chunks_pandas(paths, chunks):

    chunk_of_paths = [int(re.search("_([0-9]*)-", os.path.basename(path)).group(1)) for path in paths]
    paths_df = pd.DataFrame({"chunk": chunk_of_paths, "path": paths})
    paths_df=paths_df.loc[paths_df["chunk"].isin(chunks)]
    paths=paths_df["path"].values.tolist()
    return paths

def filter_by_chunks(paths, chunks):

    output=[]
    chunk_of_paths = [int(re.search("_([0-9]*)-", os.path.basename(path)).group(1)) for path in paths]

    for i in range(len(chunk_of_paths)):
        if chunk_of_paths[i] in chunks:
            output.append(paths[i])

    return output
    

def write_angle_to_dataset(dataset, chunks=None, n_jobs=1):
    """

    Args:
        dataset (str): Path to a labels/ folder in a YOLOv7 output
    """
    label_files = glob.glob(os.path.join(dataset, "*"))

    if chunks is not None:
       label_files = filter_by_chunks_pandas(label_files, chunks) 

    angle_folder=os.path.join(os.path.dirname(dataset), "angles")

    os.makedirs(angle_folder, exist_ok=True)

    print(f"{len(label_files)} label files to be processed")

    angles=joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(write_angle)(
        label_f, angle_folder=angle_folder, top=1
        ) for label_f in label_files
    )

    return angles
    
    
def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", required=True)
    ap.add_argument("--n-jobs", default=-2, type=int)
    ap.add_argument("--chunks", type=int, nargs="+", default=None)
    return ap

        
def process_labels(label_file):
    """
    Compute angle for all detections in the label_file

    See compute_angle
    """

    try:
        with open(label_file, "r") as filehandle:
            lines = filehandle.readlines()
    except Exception as error:
        print(f"Cannot read {label_file}")
        raise error
    
    detections = []
    for line in lines:
        detections.append(load_detection(line))
    
    descriptors = [(detection.class_id, detection.conf, round(detection.angle, 2)) for detection in detections]        
    return descriptors

def write_angle_txt(label_file, angle_folder, top=None):
    """
    Compute and write down the angle between each detection and the centroid 
    The angle is written down to a separate file with the same name as the label file, in the angle_folder

    See compute_angle
    """

    descriptors = process_labels(label_file)
    
    filename = os.path.basename(label_file)
    angle_file = os.path.join(angle_folder, filename)
    with open(angle_file, "w") as filehandle:
        for id, conf, angle in descriptor:
            filehandle.write(f"{id} {conf} {angle}\n")

    descriptors = sorted(descriptors, key=lambda descriptor: descriptor[1])[::-1]

    if top is None:
        return descriptors 
    else:
        return descriptors[:top]
    

def write_angle_h5py(label_file, angle_folder, top=None):

    dataset_name = os.path.splitext(os.path.basename(label_file))[0]
    descriptor_file = os.path.join(angle_folder, "database.h5py")
    descriptors = process_labels(label_file)
    descriptors = sorted(descriptors, key=lambda descriptor: descriptor[1])[::-1]

    with h5py.File(angle_file, "w") as filehandle:
        try:
            filehandle.create_dataset(dataset_name, data=descriptors[0])
        except Exception as error:
            # TODO Capture specifically the dataset exists error
            print(error)
            pass

    if top is None:
        return descriptors 
    else:
        return descriptors[:top]
 

def write_angle(*args, **kwargs):
    return write_angle_h5py(*args, **kwargs)

def main(args=None, ap=None):

    if args is None:
        if ap is None:
            ap = get_parser()
        args = ap.parse_args()

    basedir=os.path.dirname(args.store_path)
    dataset=os.path.join(basedir, "angles", "FlyHead", "labels")
    write_angle_to_dataset(dataset, chunks=args.chunks, n_jobs=args.n_jobs)
