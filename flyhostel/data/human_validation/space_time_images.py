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

from flyhostel.data.pose.constants import chunksize

logger=logging.getLogger(__name__)

def list_video(scene_number, folder):
    pattern=f"{folder}/movies/*{str(scene_number).zfill(10)}*.mp4"
    try:
        path=glob.glob(pattern)[0]
    except IndexError as error:
        logger.error(f"{pattern} not found")
        raise error
    return path

def generate_space_time_image(scene_number, folder, number_of_rows=4, number_of_columns=4):
    video=list_video(scene_number, folder)
    scene_name=os.path.splitext(os.path.basename(video))[0]

    frames=[]
    empty_frame=np.zeros((500, 500, 3), dtype=np.uint8)
    assert os.path.exists(video)
    cap=cv2.VideoCapture(video)
    count=0
    while True:
        ret, frame=cap.read()
        if ret:
            frame=cv2.resize(frame, (500, 500))
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
        cv2.imwrite(f"{output_folder}/{scene_name}_{str(block).zfill(3)}.png", img)


def generate_space_time_image_all(scenes, folder, n_jobs):
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            generate_space_time_image
        )(
            scene_number, folder=folder, number_of_rows=3, number_of_columns=3
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
        scene=int(os.path.splitext(os.path.basename(path))[0].rstrip("_qc").split("_")[-1])
        qc.append(load_yaml_qc(path, scene))
    qc=pd.concat(qc, axis=0)
    qc["chunk"]=qc["scene"]//chunksize
    qc.sort_values("scene", inplace=True)
    return qc



def make_space_time_images(folder, experiment, n_jobs):

    qc=load_qc(folder)
    pass_qc=qc.loc[(qc["gap_n_frames"]==0) & (qc["gap_distance"]==0)]

    with open(os.path.join(folder, "qc_pass.txt"), "w") as handle:
        for scene_number in pass_qc["scene"]:
            video=list_video(scene_number=scene_number, folder=folder)
            handle.write(f"{video}\n")

    complex_qc=qc.loc[~((qc["gap_n_frames"]==0) & (qc["gap_distance"]==0))  & ((qc["between_chunks"]==0) | (qc["all_valid_ids"]==0))]
    
    complex_qc.loc[complex_qc["between_chunks"]==0, "broken"].mean()
    complex_qc["maintains_id"].mean()

    mp4_videos=glob.glob(os.path.join(folder, "movies", "*.mp4"))
    manifest_files=glob.glob(os.path.join(folder, "movies", "*.jsonl"))

    assert len(mp4_videos)!=0, f"No .mp4 videos found, did you run the make_videos.sh script?"
    video_keys=set([os.path.splitext(path)[0] for path in mp4_videos])
    manifest_keys=set([os.path.splitext(path)[0] for path in manifest_files])

    if len(mp4_videos)!=len(manifest_files):
        print("Videos without manifest")
        for video in list(video_keys.difference(manifest_keys)):
            print(video)
        raise Exception(f"{len(mp4_videos)}!={len(manifest_files)}")


    logger.info("Generating space time images for %s scenes", complex_qc.shape[0])

    generate_space_time_image_all(complex_qc["scene"].values.tolist(), folder=folder, n_jobs=n_jobs)


    df_bin=pd.read_feather(f"{folder}/{experiment}_machine-validation-index-0.013-s.feather")
    chunk_time=df_bin.groupby("chunk").first()["t_round"].reset_index().rename({"t_round": "t"}, axis=1)
    chunk_time["zt"]=np.round(chunk_time["t"]/3600, 2)
    chunk_time["t"]=np.round(chunk_time["t"], 2)
    chunk_time.to_csv(os.path.join(folder, "chunk_time.csv"))

    mean_length_per_chunk=complex_qc.groupby("chunk").agg({"length": np.mean}).reset_index()
    
    complex_qc=complex_qc.merge(chunk_time, on=["chunk"])
    mean_length_per_chunk=mean_length_per_chunk.merge(chunk_time, on=["chunk"])
    
    plt.bar(mean_length_per_chunk["t"]/3600, mean_length_per_chunk["length"])
    plt.savefig(os.path.join(folder, f"{experiment}_scene_len_per_chunk.png"))
    plt.clf()
    
    _ = plt.hist(complex_qc["t"]/3600, bins=24)
    plt.savefig(os.path.join(folder, f"{experiment}_scene_count.png"))
    plt.clf()

