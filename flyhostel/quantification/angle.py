import argparse
import os.path
import glob

import joblib
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def write_angle_to_dataset(dataset, n_jobs=1):
    label_files = glob.glob(os.path.join(dataset, "*"))
    print(f"{len(label_files)} label files to be processed")
    
    Output=joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(compute_angle)(
        file
        ) for file in label_files
    )
    
    
def get_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-path", required=True)
    ap.add_argument("--n-jobs", default=-2, type=int)
    return ap


    
    
def compute_angle(detection):
    """
    Compute angle between center of image and centroid of YOLOv7 detection
    
    Returns:
    
    * angle (float): Orientation of the detection (0 points North and 90 points East)
    """
    
    centroid = (detection[1], detection[2])
    centroid_with_detection_centroid_at_00 = (centroid[0] - 0.5, centroid[1] - 0.5)
    
    
    angle = np.rad2deg(np.arctan2(*centroid_with_detection_centroid_at_00[::-1]))
    if angle > 180:
        angle -= 360

    if angle < - 180:
        angle += 360

    return angle

    
def compute_all_angles(label_file):
    """
    Compute angle for all detections in the label_file

    See compute_angle
    """

    with open(label_file, "r") as filehandle:
        lines = filehandle.readlines()
    
    detections = []
    for line in lines:
        detections.append([float(e) for e in line.strip("\n").split(" ")])
    
    # detections = sorted(detections, key=lambda detection: detection[-1])[::-1]
    angles = [compute_angle(detection) for detection in detections]        
    return angles

def write_angle(label_file):
    """
    Edit the label file so the angle of each object is appended to the line
    See compute_angle
    
    """
    angles = compute_all_angles(label_file)
    
    with open(label_file, "r") as filehandle:
        lines = filehandle.readlines()
    
    with open(label_file, "w") as filehandle:
        for i, line in enumerate(lines):
            angle=round(angles[i], 2)
            line = f"{line.strip()} {angle}"
            filehandle.write(line + "\n")
            
def main():
    ap = get_parser()
    args = ap.parse_args()

    dataset=os.path.join(os.path.dirname(args.store_path), "angles", "FlyHead")
    write_angle_to_dataset(dataset, args.n_jobs)