import json
import requests
import os.path
import logging
import zipfile
import shutil
import subprocess
import shlex
from tqdm.auto import tqdm
import joblib
import numpy as np
import pandas as pd
from flyhostel.data.human_validation.cvat.utils import assign_in_frame_indices
from flyhostel.data.human_validation.cvat.contour_utils import (
    rle_to_blob,
    polygon_to_blob,
)
from flyhostel.data.human_validation.cvat.utils import (
    load_original_resolution,
)
from flyhostel.utils.utils import (
    get_chunksize,
    get_number_of_animals,
    get_experiment_identifier
)
from flyhostel.utils.cvat import (
    get_tasks_for_project,
    get_project_id_from_name,
    get_task_mtime,
)

logger=logging.getLogger(__name__)
from flyhostel.data.human_validation.cvat.constants import cvat_username, cvat_host, cvat_password


def get_zipfile_for_task(path, task_number):
    zip_file=os.path.join(path, f"{task_number}_annotations.zip")
    return zip_file


DEBUG=True
def download_task_annotations_to_zip(task_number, path = ".", redownload=False):
    """
    
    Arguments
        task (int):
        path (str): Folder where to download the zip file
    """

    unzipped_folder=f"task_{task_number}"
    zip_file = get_zipfile_for_task(path, task_number)

    if not os.path.exists(unzipped_folder) or redownload:
    
        if os.path.exists(zip_file):
            os.remove(zip_file)

        if os.path.exists(zip_file):
            shutil.rmtree(unzipped_folder)

        cmd=f"""
        /home/vibflysleep/mambaforge/envs/rapids-23.04/bin/cvat-cli
        --auth {cvat_username}:{cvat_password}
        --server-host 'http://{cvat_host}'
        --server-port 8080
        dump --format 'COCO 1.0' {task_number} {zip_file}
        """
        print(cmd)
        cmd_list=shlex.split(cmd)

        p=subprocess.Popen(
            cmd_list
        )
        p.communicate()

        assert os.path.exists(zip_file), f"{zip_file} was not downloaded"

    return zip_file

def download_annotations_from_cvat(experiment, path, tasks=None):
    if tasks is None:
        tasks=sorted(get_tasks_for_project(get_project_id_from_name(experiment, errors="raise")))
    zip_files=[]

    for task in tqdm(tasks, desc="Downloading CVAT annotations to .zip"):
        zip_files.append(download_task_annotations_to_zip(task, path = path, redownload=True))
    return zip_files


def download_task_annotations(task_number, *args, **kwargs):
    unzipped_folder=f"task_{task_number}"

    zip_file=download_task_annotations_to_zip(task_number, *args, **kwargs)
    assert os.path.exists(zip_file), f"{zip_file} not found"

    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzipped_folder)
    except Exception as error:
        logger.error(error)
        logger.error("Cannot unzip %s", zip_file)
        import ipdb; ipdb.set_trace()

    with open(f"{unzipped_folder}/annotations/instances_default.json", "r") as handle:
        cvat_annotations=json.load(handle)
   
    annotations=cvat_annotations["annotations"]
    images=pd.DataFrame(cvat_annotations["images"])
    categories=pd.DataFrame(cvat_annotations["categories"])


    mtime=get_task_mtime(task_number)

    return annotations, images, categories, mtime


def load_task_annotations(annotations, images, categories, basedir, frame_width=1000, frame_height=1000, number_of_rows=1, number_of_cols=1, chunksize=None, image_format="v1"):
    """
    Returns:

      frame_number
      x
      y
      local_identity
      contour_id
      text
      frame_idx_in_block
    """
    parsed_annot=[]
    contours=[]

    assert chunksize is not None

    original_resolution=load_original_resolution(basedir)


    print(f"Detected resolution: {original_resolution}")


    block_size=number_of_rows*number_of_cols
    seen_lids={}
    i=0

    for annotation in annotations:
        panel=annotation["image_id"]
        contour_id=annotation["id"]

        image=images.loc[
            images["id"]==panel
        ]
        image_filename=image["file_name"].item()
        # frame_number0 = first frame of the scene

        tokens=os.path.splitext(os.path.basename(image_filename))[0].split("_")

        if image_format=="v1":
            try:
                block=int(tokens[-1])
                frame_number0=int(tokens[-2])
            except Exception as error:
                print("Detected filenames dont conform to v1 format. Did you forget to pass --multisex?")
                raise error
            
        else:
            block=None
            frame_number0=int(tokens[0])

        category=categories.loc[
            categories["id"]==annotation["category_id"], "name"
        ].item()
        try:
            local_identity=int(category)
            text=None
        except:
            text=category
            local_identity=None

        is_mask=isinstance(annotation["segmentation"], dict) and "counts" in annotation["segmentation"]
        
        if is_mask:
            try:
                # if mask
                frame_idx_in_block, center, contour=rle_to_blob(
                    rle=annotation["segmentation"], shape=annotation["segmentation"]["size"],
                    frame_width=frame_width, frame_height=frame_height,
                    number_of_cols=number_of_cols,
                    original_resolution=original_resolution,
                )
            except AssertionError as error:
                logger.error("Problem with image %s", image_filename)
                raise error

        else:
            polygon=annotation["segmentation"]
            if not polygon:
                # user created a rectangle (a polygon with 4 points)\
                # which is stored in the bbox entry
                if annotation["bbox"]:
                    x, y, w, h=annotation["bbox"]
                    polygon=[(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                else:
                    raise ValueError("No ROI found")

            frame_idx_in_block, center, contour=polygon_to_blob(
                polygon=polygon,
                frame_width=frame_width, frame_height=frame_height,
                number_of_cols=number_of_cols,
                original_resolution=original_resolution,
            )


        if image_format=="v1":
            frame_number=frame_number0+frame_idx_in_block + block*block_size
        else:
            frame_number=frame_number0

        if frame_number not in seen_lids:
            seen_lids[frame_number]=[]
    
        if local_identity is not None:
            if local_identity in seen_lids[frame_number]:
                # logger.debug("local identity %s already seen in %s", local_identity, frame_number)
                continue
            else:
                seen_lids[frame_number].append(local_identity)

        parsed_annot.append((i, frame_number, *center, local_identity, contour_id, text, frame_number0, block, block_size, panel-1, frame_idx_in_block))
        contours.append(contour)
        i+=1

    annotations_df=pd.DataFrame.from_records(
        parsed_annot,
        columns=["idx", "frame_number", "x", "y", "local_identity", "contour_id", "text", "frame_number0", "block", "block_size", "panel", "frame_idx_in_block"],
    )
    
    annotations_df["frame_number"]=annotations_df["frame_number"].astype(int)
    annotations_df["chunk"]=annotations_df["frame_number"]//chunksize
    annotations_df["x"]=annotations_df["x"].astype(float)
    annotations_df["y"]=annotations_df["y"].astype(float)
    return annotations_df, contours


def join_task_data(task1, task2):
    """
    Join cvat annotations from more than 1 task but the same experiment
    """
    annotations_df, contours=task1
    annotations_df2, contours2=task2

    first_id_of_next_contour=len(contours)

    contours=contours+contours2
    annotations_df2["idx"]+=first_id_of_next_contour
    annotations_df=pd.concat([
        annotations_df, annotations_df2
    ], axis=0)
    return annotations_df, contours

def get_annotations(experiment, basedir, tasks, n_jobs=2, **kwargs):

    n_jobs=min(len(tasks), n_jobs)
    out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            get_annotation
        )(
            experiment, basedir, task_number, **kwargs
        )
        for task_number in tasks
    )

    annotations_df, contours, mtime=out[0]
    max_mtime=mtime
    
    # if more than 1 task was passed
    for annotations_df2, contours2, mtime in out[1:]:
        annotations_df, contours=join_task_data(
            (annotations_df, contours), (annotations_df2, contours2)
        )
        if mtime > max_mtime:
            max_mtime = mtime

    
    experiment=get_experiment_identifier(basedir)
    number_of_animals=get_number_of_animals(experiment)

    annotations_df=assign_in_frame_indices(annotations_df, number_of_animals, experiment=experiment)
    annotations_df["fragment"]=np.nan

    return annotations_df, contours, max_mtime


def get_annotation(experiment, basedir, task_number, number_of_cols=1, number_of_rows=1, image_format="v1", **kwargs):
    annotations, images, categories, mtime=download_task_annotations(task_number, **kwargs)
    chunksize=get_chunksize(experiment)
    
    assert len(images["width"].unique())==1
    frame_width=images["width"].unique()[0]//number_of_cols

    assert len(images["height"].unique())==1
    frame_height=images["height"].unique()[0]//number_of_rows
    
    annotations_df, contours=load_task_annotations(
        annotations, images, categories,
        basedir=basedir,
        frame_width=frame_width,
        frame_height=frame_height,
        number_of_rows=number_of_rows, number_of_cols=number_of_cols,
        chunksize=chunksize,
        image_format=image_format
    )
    annotations_df["task"]=task_number
    return annotations_df, contours, mtime



