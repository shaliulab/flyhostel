import glob
import logging
import os.path

import matplotlib.pyplot as plt
import cv2
import yaml
import joblib
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from flyhostel.utils import get_chunksize
from .constants import (
    NUMBER_OF_COLS,
    NUMBER_OF_ROWS,
    RESOLUTION,
)

logger=logging.getLogger(__name__)

def list_video(scene_number, folder):
    pattern=f"{folder}/movies/*{str(scene_number).zfill(10)}*.mp4"
    try:
        path=glob.glob(pattern)[0]
    except IndexError as error:
        logger.error(f"{pattern} not found")
        raise error
    return path

def generate_image(scene_number, folder, number_of_rows=4, number_of_columns=4):
    video=list_video(scene_number, folder)
    scene_name=os.path.splitext(os.path.basename(video))[0]

    frames=[]
    empty_frame=np.zeros((*RESOLUTION[::-1], 3), dtype=np.uint8)
    assert os.path.exists(video)
    logger.debug("Opening %s", video)
    cap=cv2.VideoCapture(video)
    count=0
    while True:
        ret, frame=cap.read()
        if count==0:
            logger.debug("Reading %s", video)
        if ret:
            frame=cv2.resize(frame, RESOLUTION)
            frames.append(frame)
        else:
            frames.append(empty_frame.copy())
        count+=1
        if not ret and count % (number_of_rows*number_of_columns) == 0:
            break
    cap.release()

    output_folder=os.path.join(folder, "space_time", scene_name)
    os.makedirs(output_folder, exist_ok=True)
    block_size=number_of_rows*number_of_columns
    
    for block in range(0, len(frames)//block_size, 1):
        img=[]
        for i in range(number_of_rows):
            start=i*number_of_columns + block*block_size
            end=i*number_of_columns+number_of_columns + block*block_size
            
            img.append(
                np.hstack(frames[start:end])
            )
        img=np.vstack(img)
        cv2.imwrite(f"{output_folder}/{scene_name}_{str(block).zfill(6)}.png", img)


def generate_images_all(scenes, folder, n_jobs):
    logger.debug("Running generate_image in %s jobs", n_jobs)
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            generate_image
        )(
            scene_number, folder=folder,
            number_of_rows=NUMBER_OF_ROWS,
            number_of_columns=NUMBER_OF_COLS,
        )
       for scene_number in tqdm(scenes)
    )

def load_yaml_qc(path, scene):
    with open(path, "r") as handle:
        qc=yaml.load(handle, yaml.SafeLoader)
        qc=pd.DataFrame({k: [v] for k, v in qc.items()})
        qc.insert(0, "scene", scene)
    return qc

def load_qc(folder):
    yaml_files=sorted(glob.glob(f"{folder}/movies/*yaml"))
    logger.debug(f"{len(yaml_files)} yaml files found")
    qc=[]

    for path in yaml_files:
        csv_file=path.replace("_qc.yaml", ".csv")      
        scene=int(pd.read_csv(csv_file)["frame_number"])
        qc.append(load_yaml_qc(path, scene))
    qc=pd.concat(qc, axis=0)
    qc.sort_values("scene", inplace=True)
    return qc


def export_images(folder, experiment, n_jobs):

    qc=load_qc(folder)
    pass_qc=qc.loc[qc["pass"]==True]

    with open(os.path.join(folder, "qc_pass.txt"), "w") as handle:
        for scene_number in pass_qc["scene"]:
            video=list_video(scene_number=scene_number, folder=folder)
            handle.write(f"{video}\n")

    # NOTE This is where we produce the subset of frames with problem
    complex_qc=qc.loc[qc["pass"]==False]
    logger.info("Generating space time images for %s scenes", complex_qc.shape[0])
    generate_images_all(complex_qc["scene"].values.tolist(), folder=folder, n_jobs=n_jobs)