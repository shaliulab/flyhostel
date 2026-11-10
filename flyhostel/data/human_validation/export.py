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
    IMAGES_FOLDER,
    MAX_SCENE_LENGTH,
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

def write_block(frames, number_of_rows, number_of_columns, path):
    img=np.vstack([
        np.hstack(frames[i*number_of_columns:(i+1)*number_of_columns])
        for i in range(number_of_rows)
    ])
    cv2.imwrite(path, img)


def generate_image(scene_number, folder, number_of_rows=4, number_of_columns=4):
    """
    One image per block of number_of_rows x number_of_columns frames of the scene's movie, written as soon as
    the block is read (memory does not grow with the length of the scene). The last block is completed with
    empty frames, so the images (and their names) are the same as when all frames were read first.
    """
    video=list_video(scene_number, folder)
    scene_name=os.path.splitext(os.path.basename(video))[0]
    output_folder=os.path.join(folder, IMAGES_FOLDER, scene_name)
    os.makedirs(output_folder, exist_ok=True)
    block_size=number_of_rows*number_of_columns

    empty_frame=np.zeros((*RESOLUTION[::-1], 3), dtype=np.uint8)
    assert os.path.exists(video)
    logger.debug("Opening %s", video)
    cap=cv2.VideoCapture(video)
    frames=[]
    count=0
    block=0
    while True:
        ret, frame=cap.read()
        if count==0:
            logger.debug("Reading %s", video)
        frames.append(cv2.resize(frame, RESOLUTION) if ret else empty_frame.copy())
        count+=1
        if len(frames)==block_size:
            write_block(frames, number_of_rows, number_of_columns,
                        f"{output_folder}/{scene_name}_{str(block).zfill(6)}.png")
            frames=[]
            block+=1
        if not ret and count % block_size == 0:
            break
    cap.release()


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
    """
    entrypoint of export-images
    """

    qc=load_qc(folder)
    pass_qc=qc.loc[qc["pass"]==True]

    with open(os.path.join(folder, "qc_pass.txt"), "w") as handle:
        for scene_number in pass_qc["scene"]:
            video=list_video(scene_number=scene_number, folder=folder)
            handle.write(f"{video}\n")

    # NOTE This is where we produce the subset of frames with problem
    complex_qc=qc.loc[qc["pass"]==False]

    # Scenes this long are not errors to annotate frame by frame (e.g. hours where YOLOv7 found the flies
    # because idtrackerai did not): they are listed in long_scenes.txt and not exported
    long=complex_qc["length"] > MAX_SCENE_LENGTH
    with open(os.path.join(folder, "long_scenes.txt"), "w") as handle:
        handle.write("scene\tlength\tlast_frame_number\n")
        for scene, length in complex_qc.loc[long, ["scene", "length"]].values:
            handle.write(f"{int(scene)}\t{int(length)}\t{int(scene + length - 1)}\n")
    if long.any():
        logger.warning("%s scene(s) longer than %s frames not exported (see long_scenes.txt): %s",
                       long.sum(), MAX_SCENE_LENGTH, complex_qc.loc[long, "scene"].tolist())
    complex_qc=complex_qc.loc[~long]
    logger.info("Generating space time images for %s scenes", complex_qc.shape[0])
    generate_images_all(complex_qc["scene"].values.tolist(), folder=folder, n_jobs=n_jobs)