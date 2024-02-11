import json
import os.path
import logging
import zipfile
import shutil
import subprocess
import shlex



import joblib
import numpy as np
import pandas as pd
from imgstore.interface import VideoCapture
from idtrackerai_validator_server.backend import load_idtrackerai_config, process_frame
from flyhostel.data.pose.constants import chunksize
from flyhostel.data.human_validation.cvat.contour_utils import (
    rle_to_blob, polygon_to_blob, get_contour_list_from_yolo_centroids, select_by_contour
)
from flyhostel.data.human_validation.cvat.utils import (
    load_original_resolution, annotate_crossings
)
logger=logging.getLogger(__name__)


def download_task_annotations(task_number, reload=False):

    unzipped_folder=f"task_{task_number}"
    
    if not os.path.exists(unzipped_folder) or reload:
        zip_file=f"{task_number}_annotations.zip"
        
        if os.path.exists(zip_file):
            os.remove(zip_file)
        
        if os.path.exists(zip_file):
            shutil.rmtree(unzipped_folder)
        
        cmd=f"/home/vibflysleep/mambaforge/envs/rapids-23.04/bin/cvat-cli --auth vibflysleep:flysleep1 --server-host 'http://0.0.0.0' --server-port 8080 dump --format 'COCO 1.0' {task_number} {task_number}_annotations.zip"
        cmd_list=shlex.split(cmd)
        
        p=subprocess.Popen(
            cmd_list
        )
        p.communicate()
        
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(unzipped_folder)

    with open(f"{unzipped_folder}/annotations/instances_default.json", "r") as handle:
        cvat_annotations=json.load(handle)

    annotations=cvat_annotations["annotations"]
    images=pd.DataFrame(cvat_annotations["images"])
    categories=pd.DataFrame(cvat_annotations["categories"])
    return annotations, images, categories


def load_task_annotations(annotations, images, categories, basedir, frame_width=500, frame_height=500, number_of_rows=3, number_of_cols=3):
    parsed_annot=[]
    contours=[]
    original_resolution=load_original_resolution(basedir)
    
    
    block_size=number_of_rows*number_of_cols
    for i, annotation in enumerate(annotations):
        image=images.loc[
            images["id"]==annotation["image_id"]
        ]
        image_filename=image["file_name"].item()
        frame_number=int(
            os.path.splitext(os.path.basename(image_filename))[0].split("_")[-2]
        )
        block=int(
            os.path.splitext(os.path.basename(image_filename))[0].split("_")[-1]
        )
        
        category=categories.loc[
            categories["id"]==annotation["category_id"], "name"
        ].item()
        try:
            identity=int(category)
            text=None
        except:
            text=category
            identity=None
    
        if isinstance(annotation["segmentation"], dict):
            frame_idx_in_block, center, contour=rle_to_blob(
                rle=annotation["segmentation"], shape=annotation["segmentation"]["size"],
                frame_width=frame_width, frame_height=frame_height,
                number_of_cols=number_of_cols,
                original_resolution=original_resolution,
            )
        else:
            frame_idx_in_block, center, contour=polygon_to_blob(
                polygon=annotation["segmentation"],
                frame_width=frame_width, frame_height=frame_height,
                number_of_cols=number_of_cols,
                original_resolution=original_resolution,
            )
    
        frame_number0=int(image_filename.split("_")[-2])
        frame_number=frame_number0+frame_idx_in_block + block*block_size
        parsed_annot.append((frame_number, *center, identity, i, text))
        contours.append(contour)

    annotations_df=pd.DataFrame.from_records(parsed_annot, columns=["frame_number", "x", "y", "local_identity", "contour_id", "text"])
    

    return annotations_df, contours


def join_task_data(task1, task2):
    """
    Join cvat annotations from more than 1 task but the same experiment
    """
    annotations_df, contours=task1
    annotations_df2, contours2=task2
    
    first_id_of_next_contour=len(contours)

    contours=contours+contours2
    annotations_df2["contour_id"]+=first_id_of_next_contour
    annotations_df=pd.concat([
        annotations_df, annotations_df2
    ], axis=0)
    return annotations_df, contours

def get_annotations(basedir, tasks, n_jobs=2):
    out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            get_annotation
        )(
            basedir, task_number
        )
        for task_number in tasks
    )

    annotations_df, contours=out[0]
    for annotations_df2, contours2 in out[1:]:
        annotations_df, contours=join_task_data(
            (annotations_df, contours), (annotations_df2, contours2)
        )

    return annotations_df, contours


def get_annotation(basedir, task_number):
    reload=True
    annotations, images, categories=download_task_annotations(task_number, reload=reload)
    annotations_df, contours=load_task_annotations(
        annotations, images, categories,
        basedir=basedir,
        frame_width=500, frame_height=500,
        number_of_rows=3, number_of_cols=3,
    )
    return annotations_df, contours
    
    
def cross_tracking_annotation(basedir, tracking_data, annotations_df, annotated_contours, last_machine_id):
    config=load_idtrackerai_config(basedir)
    cap=None
    score_dist=[]
    try:
        cap=VideoCapture(os.path.join(basedir, "metadata.yaml"), 50)

        roi0_corrected=[]
        identity_corrected=[]
        fragments_must_break=[]
        
        for frame_number, df in tracking_data.groupby("frame_number"):
            annotation=annotations_df.loc[annotations_df["frame_number"]==frame_number]
            cap.set(1, frame_number)
            ret, frame = cap.read()
            frame=frame[:,:,0]
            contours_list = process_frame(frame, config)

            selection_method="contour"
            
            if (df["modified"]==1).any():
                contours_list=get_contour_list_from_yolo_centroids(df[["x", "y"]].values, size=50)
                # raise ValueError("modified frames not supported")

            used_indices=[]
        
            for annot_idx_2 in range(annotation.shape[0]):

                contour=annotated_contours[annotation["contour_id"].iloc[annot_idx_2]]

                if selection_method=="contour":
                    try:
                        match_idx=select_by_contour(contour, contours_list)
                    except ValueError as error:
                        logger.warning("More than 1 yolo box overlaps in frame %s and annotation index %s", frame_number, annot_idx_2)
                        match_idx=None
                    # else:
                    #     match_idx=select_by_centroid(contour, df[["x", "y"]])
                    
                # annotation overlaps
                if match_idx is not None:
                    fragment=df["fragment"].iloc[match_idx]
                    in_frame_index=df["in_frame_index"].iloc[match_idx]
                    # overlaps with a 
                    if in_frame_index in used_indices:
                        in_frame_index=annot_idx_2+last_machine_id
                        
                    used_indices.append(in_frame_index)
                    local_identity=annotation["local_identity"].iloc[annot_idx_2]
                
                # annotation doesn't overlap with anything
                else:
                    fragment=None
                    in_frame_index=annot_idx_2+last_machine_id
                    while in_frame_index in used_indices:
                        in_frame_index+=1
                    used_indices.append(in_frame_index)

                    
                    # print(annot_idx_2, last_machine_id)
                    local_identity=annotation["local_identity"].iloc[annot_idx_2]
                    logger.debug("De novo annotation detected in frame %s with local identity %s", frame_number, local_identity)

                if np.isnan(local_identity):
                    text=annotation["text"].iloc[annot_idx_2]
                    if text=="FMB": # fragment must break
                        fragments_must_break.append((frame_number, fragment))
                        continue

                roi0_row=(frame_number, in_frame_index, annotation["x"].iloc[annot_idx_2], annotation["y"].iloc[annot_idx_2], fragment)
                ident_row=(frame_number, in_frame_index, local_identity)

                # if frame_number == 11386698:
                #     print(roi0_row, ident_row)
                        
                roi0_corrected.append(roi0_row)
                identity_corrected.append(ident_row)
        identity_corrected=pd.DataFrame.from_records(identity_corrected, columns=["frame_number", "in_frame_index", "local_identity"])
        roi0_corrected=pd.DataFrame.from_records(roi0_corrected, columns=["frame_number", "in_frame_index", "x", "y", "fragment"])
    
    finally:
        if cap is not None:
            cap.release()

    roi0_corrected["modified"]=False
    roi0_corrected["validated"]=True
    identity_corrected["validated"]=True
    roi0_corrected["chunk"]=roi0_corrected["frame_number"]//chunksize

    if fragments_must_break:
        max_fragment_identifier_per_chunk=roi0_corrected.groupby("chunk").agg({"fragment": np.max}).reset_index()

        for frame_number, fragment in fragments_must_break:
            chunk=frame_number//chunksize
            new_identifier=max_fragment_identifier_per_chunk.loc[max_fragment_identifier_per_chunk["chunk"]==chunk, "fragment"].item()+1

            logger.warning("Fragment %s after frame number %s becomes fragment %s", fragment, frame_number, new_identifier)
            roi0_corrected.loc[(roi0_corrected["frame_number"]>frame_number)&(roi0_corrected["chunk"]==chunk)&(roi0_corrected["fragment"]==fragment), "fragment"]=new_identifier
            max_fragment_identifier_per_chunk.loc[max_fragment_identifier_per_chunk["chunk"]==chunk, "fragment"]+=1
    
    identity_corrected=annotate_crossings(identity_corrected)

    fragment_counter=roi0_corrected.groupby(["frame_number", "fragment"]).size().reset_index()
    bad_fragments=fragment_counter.loc[fragment_counter[0]>1]

    # logger.warning(bad_fragments)
    
    
    for i, row in bad_fragments.iterrows():
        roi0_corrected.loc[(roi0_corrected["fragment"]==row["fragment"]) & (roi0_corrected["frame_number"]==row["frame_number"]), "fragment"]=np.nan

    return identity_corrected, roi0_corrected, score_dist
