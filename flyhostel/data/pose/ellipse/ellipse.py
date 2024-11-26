import itertools
import joblib
import os.path
import math
import json
import sqlite3

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from imgstore.interface import VideoCapture
from idtrackerai.animals_detection.segmentation import _process_frame

def project_to_absolute(loader, bodyparts):
    features = list(itertools.chain(*[[f"{bodypart}_x", f"{bodypart}_y"] for bodypart in bodyparts]))
    
    df_pose=loader.pose[["id", "frame_number"] + features]
    xy=loader.dt[["frame_number", "x", "y"]]
    xy["x"]*=loader.roi_width
    xy["y"]*=loader.roi_width
    df_pose=df_pose.merge(xy, on="frame_number", how="inner")
    for bodypart in bodyparts:
        df_pose[f"{bodypart}_x"]+=df_pose["x"]-50
        df_pose[f"{bodypart}_y"]+=df_pose["y"]-50
    return df_pose


def compute_ellipse_parameters(df_pose):
    """
    Computes the ellipse parameters (major axis length, minor axis length, angle)
    for each fly based on the pose data, matching the desired angle convention.

    Parameters:
    - df_pose: pandas DataFrame containing the pose data with required columns.

    Returns:
    - df_pose: pandas DataFrame with added columns 'major', 'minor', 'angle', 'x', 'y'.
    """
    # Copy df_pose to avoid modifying the original DataFrame
    df_pose = df_pose.copy()

    # Since the origin is at the top-left corner and y increases downwards,
    # we do not invert the y-coordinates.

    # Step 1: Compute the Center of the Ellipse
    df_pose['center_x'] = (df_pose['head_x'] + df_pose['abdomen_x']) / 2
    df_pose['center_y'] = (df_pose['head_y'] + df_pose['abdomen_y']) / 2  # center_y is positive

    # Step 2: Compute the Major Axis
    df_pose['dx'] = df_pose['head_x'] - df_pose['abdomen_x']
    df_pose['dy'] = df_pose['head_y'] - df_pose['abdomen_y']  # Use original y-coordinates
    df_pose['length_major_axis'] = np.sqrt(df_pose['dx']**2 + df_pose['dy']**2)
    df_pose['major'] = df_pose['length_major_axis']  # Full length of the major axis

    # Compute the orientation angle in degrees
    # Adjust the angle so that:
    # - 0 degrees corresponds to the fly pointing upwards (North)
    # - 90 degrees corresponds to the fly pointing rightwards (East)
    df_pose['angle'] = (np.degrees(np.arctan2(df_pose['dx'], -df_pose['dy']))) % 360

    # Step 3: Compute the Minor Axis
    # Unit vector along major axis
    df_pose['u_x'] = df_pose['dx'] / df_pose['length_major_axis']
    df_pose['u_y'] = df_pose['dy'] / df_pose['length_major_axis']

    # Unit vector perpendicular to major axis
    df_pose['v_x'] = -df_pose['u_y']
    df_pose['v_y'] = df_pose['u_x']

    # Vectors from center to middle legs
    df_pose['dx_mLL'] = df_pose['mLL_x'] - df_pose['center_x']
    df_pose['dy_mLL'] = df_pose['mLL_y'] - df_pose['center_y']
    df_pose['dx_mRL'] = df_pose['mRL_x'] - df_pose['center_x']
    df_pose['dy_mRL'] = df_pose['mRL_y'] - df_pose['center_y']

    # Projections onto minor axis direction
    df_pose['proj_mLL'] = df_pose['dx_mLL'] * df_pose['v_x'] + df_pose['dy_mLL'] * df_pose['v_y']
    df_pose['proj_mRL'] = df_pose['dx_mRL'] * df_pose['v_x'] + df_pose['dy_mRL'] * df_pose['v_y']

    # Length of minor axis
    df_pose['length_minor_axis'] = np.abs(df_pose['proj_mLL'] - df_pose['proj_mRL'])
    df_pose['minor'] = df_pose['length_minor_axis']  # Full length of the minor axis

    # Handle missing 'minor' values by setting minor axis to half of the major axis
    df_pose.loc[df_pose["minor"].isna(), "minor"] = df_pose.loc[df_pose["minor"].isna(), "major"] / 2

    # Assign 'center_x' and 'center_y' to 'x' and 'y' for consistency
    df_pose['x'] = df_pose['center_x']
    df_pose['y'] = df_pose['center_y']

    # Drop intermediate columns if desired
    intermediate_cols = [
        'dx', 'dy', 'length_major_axis',
        'u_x', 'u_y', 'v_x', 'v_y',
        'dx_mLL', 'dy_mLL', 'dx_mRL', 'dy_mRL',
        'proj_mLL', 'proj_mRL', 'length_minor_axis'
    ]
    df_pose.drop(columns=intermediate_cols, inplace=True)

    return df_pose


def load_idtrackerai_config(basedir):
    dbfile = os.path.join(basedir, "_".join(basedir.split(os.path.sep)[-3:]) + ".db")
    with sqlite3.connect(dbfile) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM METADATA WHERE field = 'idtrackerai_conf';")
        config_str = cursor.fetchone()

    idtrackerai_config = json.loads(config_str[0].rstrip('\n'))
    return idtrackerai_config


def process_config(config):

    user_defined_parameters = {
        "number_of_animals": int(config["_number_of_animals"]["value"]),
        "min_threshold": config["_intensity"]["value"][0],
        "max_threshold": config["_intensity"]["value"][1],
        "min_area": config["_area"]["value"][0],
        "max_area": config["_area"]["value"][1],
        "check_segmentation": True,
        "tracking_interval": [0, math.inf],
        "apply_ROI": True,
        "rois": config["_roi"]["value"],
        "subtract_bkg": False,
        "bkg_model": None,
        "resolution_reduction": config["_resreduct"]["value"],
        "identity_transfer": False,
        "identification_image_size": None,
    }
    return user_defined_parameters


def process_frame(frame, config):
    """
    Generate the contours that idtrackerai would obtain using the passed config

    Arguments:

        frame (np.ndarray):
        config (dict): idtrackerai config

    Returns
        contour_list (list): List of contours. See draw_frame on how to draw them on the frame
    """
    config=process_config(config)

    roi_mask = np.zeros_like(frame)
    roi_contour = np.array(eval(config["rois"][0][0])).reshape((-1, 1, 2))
    roi_mask = cv2.drawContours(roi_mask, [roi_contour], -1, 255, -1)
    config["mask"]=roi_mask
    # cv2.imwrite("mask.png", roi_mask)
    config["resolution_reduction"]=1.0

    (
        bounding_boxes,
        miniframes,
        centroids,
        areas,
        pixels,
        contours,
        estimated_body_lengths
    ) = _process_frame(
        frame,
        config,
        0,
        "NONE",
        "NONE",
    )

    contours_list = [contour.tolist() for contour in contours]
    return contours_list

def project_to_ellipse(contours):
    ellipses=[cv2.fitEllipse(np.array(contour)) for contour in contours]
    return ellipses

def preprocess_ellipses(basedir, frame_numbers):
    records=[]
    idtrackerai_config=load_idtrackerai_config(basedir)
    cap=VideoCapture(f"{basedir}/metadata.yaml", 50)
    for i, frame_number in enumerate(tqdm(frame_numbers, desc="Preprocessing frame")):
        if i!=0 and frame_number == frame_numbers[i-1] + 1:
            _, frame=cap.read()
        else:
            frame, (frame_number, frame_timestamp) = cap.get_image(frame_number)
        
        contours=process_frame(frame, idtrackerai_config)
        ellipses=project_to_ellipse(contours)
        for ellipse in ellipses:
            (x_center, y_center), (major_axis_length, minor_axis_length), angle = ellipse
            records.append((x_center, y_center, major_axis_length, minor_axis_length, angle, frame_number))
    df=pd.DataFrame.from_records(records, columns=["x", "y", "major", "minor", "angle", "frame_number"])
    return df

def preprocess_ellipses_mp(basedir, frame_numbers, n_jobs):

    cpus=joblib.cpu_count()
    if n_jobs<0:
        n_jobs=cpus+n_jobs
    
    partition_size=len(frame_numbers)//n_jobs + 1

    partitions=[frame_numbers[i*partition_size:((i+1)*partition_size)] for i in range(n_jobs)]
    dfs=joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            preprocess_ellipses
        )(
            basedir, partitions[i]
        )
        for i in range(len(partitions))
    )
    df=pd.concat(dfs, axis=0).reset_index(drop=True)
    return df

