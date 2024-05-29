import json
import os.path
import math
import logging
import zipfile
import shutil
import subprocess
import shlex
from tqdm.auto import tqdm
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
    load_original_resolution, annotate_crossings, get_dbfile
)
from flyhostel.utils.utils import get_chunksize
logger=logging.getLogger(__name__)

cvat_host="localhost"

def download_task_annotations(task_number, redownload=False):

    unzipped_folder=f"task_{task_number}"

    if not os.path.exists(unzipped_folder) or redownload:
        zip_file=f"{task_number}_annotations.zip"

        if os.path.exists(zip_file):
            os.remove(zip_file)

        if os.path.exists(zip_file):
            shutil.rmtree(unzipped_folder)

        cmd=f"/home/vibflysleep/mambaforge/envs/rapids-23.04/bin/cvat-cli --auth vibflysleep:flysleep1 --server-host 'http://{cvat_host}' --server-port 8080 dump --format 'COCO 1.0' {task_number} {task_number}_annotations.zip"
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
    original_resolution=load_original_resolution(basedir)


    block_size=number_of_rows*number_of_cols
    for i, annotation in enumerate(annotations):
        panel=annotation["image_id"]

        image=images.loc[
            images["id"]==panel
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
        parsed_annot.append((frame_number, *center, identity, i, text, frame_number0, block, block_size, panel-1, frame_idx_in_block))
        contours.append(contour)

    annotations_df=pd.DataFrame.from_records(parsed_annot, columns=["frame_number", "x", "y", "local_identity", "contour_id", "text", "frame_number0", "block", "block_size", "panel", "frame_idx_in_block"])

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

def get_annotations(basedir, tasks, n_jobs=2, **kwargs):
    out = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            get_annotation
        )(
            basedir, task_number, **kwargs
        )
        for task_number in tasks
    )

    annotations_df, contours=out[0]
    for annotations_df2, contours2 in out[1:]:
        annotations_df, contours=join_task_data(
            (annotations_df, contours), (annotations_df2, contours2)
        )

    return annotations_df, contours


def get_annotation(basedir, task_number, **kwargs):
    annotations, images, categories=download_task_annotations(task_number, **kwargs)
    annotations_df, contours=load_task_annotations(
        annotations, images, categories,
        basedir=basedir,
        frame_width=500, frame_height=500,
        number_of_rows=3, number_of_cols=3,
    )
    annotations_df["task"]=task_number
    return annotations_df, contours

def spatial_copy_annotations(annotations_df, identity_corrected, roi0_corrected, annotations_to_copy):
    """
    For every pair of fn0, fn1 in annotations_to_copy, copy the annotations in fn1 to fn0
    """
    block_size=annotations_df["block_size"].unique().tolist()
    assert len(block_size)==1
    block_size=block_size[0]

    for frame_number, ref_frame_number in annotations_to_copy:
        annotations_df, identity_corrected, roi0_corrected=copy_annotations(
            annotations_df, identity_corrected, roi0_corrected,
            frame_number, ref_frame_number, block_size=block_size
        )

    identity_corrected.sort_values(["frame_number", "local_identity"], inplace=True)
    roi0_corrected.sort_values("frame_number", inplace=True)
    return identity_corrected, roi0_corrected


def copy_annotations(annotations_df, identity_corrected, roi0_corrected, frame_number, ref_frame_number, block_size):
    """
    
    frame_number = frame_number where the annotations will be copied
    ref_frame_number = frame number from which the annotations will be taken

    """
    row=annotations_df.loc[annotations_df["frame_number"]==ref_frame_number].iloc[:1]
    if row.shape[0]==0:
        logger.warning("No annotations for frame %s", ref_frame_number)
        return annotations_df, roi0_corrected, identity_corrected


    copied_data=identity_corrected.loc[identity_corrected["frame_number"]==ref_frame_number].copy()
    copied_data["frame_number"]=frame_number
    identity_corrected=pd.concat([
        identity_corrected.reset_index(drop=True),
        copied_data.reset_index(drop=True)
    ], axis=0)

    copied_data=roi0_corrected.loc[roi0_corrected["frame_number"]==ref_frame_number].copy()
    copied_data["frame_number"]=frame_number
    roi0_corrected=pd.concat([
        roi0_corrected.reset_index(drop=True),
        copied_data.reset_index(drop=True)
    ], axis=0)


    # so that the row has data in the following iterations (when copying a copy)
    mark_annotation=pd.DataFrame({
        "frame_number": [frame_number], "x": [None], "y": [None],  "local_identity": [None],
        "contour_id": [None],  "text": ["COPY"], "frame_number0": [None],  "block": [None],  "block_size": [block_size], 
        "panel": [None],  "task": [None], "frame_idx_in_block": [None]
    })
    annotations_df=pd.concat([annotations_df.reset_index(drop=True), mark_annotation], axis=0)
    return annotations_df, roi0_corrected, identity_corrected


def copy_annotations_one_block_back(annotations_df, identity_corrected, roi0_corrected, annotations_to_copy):
    """
    Replicate the annotations annotated one block back in the present frame

    A block is the amount of frames placed in a single space-time image (9 by default)
    This is needed to make the annotation in the validation GUI more efficient.
    One block is annotated, and several following blocks can be annotated by simply making a mask
    with the copy label in any of the frames of the block. This will make all frames of the block
    get their annotations from the equivalent frame in the preceding block

    The COPY tag should not be overused, because the exact position of the animals will drift over time
    and the preceding annotations lose accuracy with every block. 
    """

    block_size=annotations_df["block_size"].unique().tolist()
    assert len(block_size)==1
    block_size=block_size[0]
    n_steps=1

    for frame_number in annotations_to_copy:
        ref_frame_number=frame_number-block_size*n_steps
        annotations_df, identity_corrected, roi0_corrected=copy_annotations(
            annotations_df, identity_corrected, roi0_corrected,
            frame_number, ref_frame_number, block_size=block_size
        )

    identity_corrected.sort_values(["frame_number", "local_identity"], inplace=True)
    roi0_corrected.sort_values("frame_number", inplace=True)
    return identity_corrected, roi0_corrected


def cross_machine_human(basedir, identity_machine, roi_0_machine, annotations_df, annotated_contours, last_machine_id, first_frame_number=0, last_frame_number=math.inf):
    """
    
    annotations_df contains the fields exported in load_task_annotations
    """
    config=load_idtrackerai_config(basedir)
    dbfile=get_dbfile(basedir)
    chunksize=get_chunksize(dbfile)

    cap=None
    score_dist=[]
    machine_data_of_modified_frames=roi_0_machine.drop("id", axis=1).merge(identity_machine.drop("id", axis=1), on=["frame_number", "in_frame_index"])
    machine_data_of_modified_frames["chunk"]=machine_data_of_modified_frames["frame_number"]//chunksize

    frame_numbers=sorted(annotations_df["frame_number"].unique())
    machine_data_of_modified_frames=machine_data_of_modified_frames.loc[
        machine_data_of_modified_frames["frame_number"].isin(frame_numbers)
    ]
    del frame_numbers


    try:
        cap=VideoCapture(os.path.join(basedir, "metadata.yaml"), 50)

        roi0_corrected=[]
        identity_corrected=[]
        fragments_must_break=[]
        annotations_to_copy=set()
        annotations_to_spatial_copy=set()
        crossings=[]
        n_frames=len(machine_data_of_modified_frames["frame_number"].unique())

        pb=tqdm(total=n_frames, desc="Crossing human annotations and machine data")

        for frame_number, df in machine_data_of_modified_frames.groupby("frame_number"):
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
                        match_idx=select_by_contour(contour, contours_list, debug=False)
                    except ValueError as error:
                        logger.warning(error)
                        # logger.debug("More than 1 yolo box overlaps in frame %s and annotation index %s", frame_number, annot_idx_2)
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
                    local_identity=annotation["local_identity"].iloc[annot_idx_2]

                if np.isnan(local_identity):
                    text=annotation["text"].iloc[annot_idx_2]
                    if text=="FMB": # fragment must break
                        fragments_must_break.append((frame_number, fragment))
                        continue
                    elif text=="COPY":
                        frame_number0, block, block_size=annotation.loc[annotation["text"]=="COPY", ["frame_number0", "block", "block_size"]].values.flatten()
                        frame_numbers=list(range(frame_number0+block*block_size, frame_number0+block_size+block*block_size))

                        for fn in frame_numbers:
                            annotations_to_copy.add(fn)
                    
                    elif text=="SPATIAL-COPY":
                        frame_number0, block, block_size=annotation.loc[annotation["text"]=="SPATIAL-COPY", ["frame_number0", "block", "block_size"]].values.flatten()
                        ref_frame=frame_number0 + block_size//2
                        frame_numbers=list(range(frame_number0+block_size*block, frame_number0+block_size*(block+1)))


                        for fn in frame_numbers:
                            annotations_to_spatial_copy.add((fn, ref_frame))

                    elif text=="CROSSING":
                        crossings.append((frame_number, fragment))

                elif match_idx is None:
                    logger.debug("De novo annotation detected in frame %s with local identity %s", frame_number, local_identity)


                roi0_row=(frame_number, in_frame_index, annotation["x"].iloc[annot_idx_2], annotation["y"].iloc[annot_idx_2], fragment)
                ident_row=(frame_number, in_frame_index, local_identity)

                roi0_corrected.append(roi0_row)
                identity_corrected.append(ident_row)
            
            pb.update(1)
        identity_corrected=pd.DataFrame.from_records(identity_corrected, columns=["frame_number", "in_frame_index", "local_identity"])
        roi0_corrected=pd.DataFrame.from_records(roi0_corrected, columns=["frame_number", "in_frame_index", "x", "y", "fragment"])

        identity_corrected, roi0_corrected=spatial_copy_annotations(
            annotations_df,
            identity_corrected,
            roi0_corrected,
            sorted(list(annotations_to_spatial_copy))           
        )

        identity_corrected, roi0_corrected=copy_annotations_one_block_back(
            annotations_df,
            identity_corrected,
            roi0_corrected,
            sorted(list(annotations_to_copy))
        )



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

    identity_corrected=annotate_crossings(identity_corrected, roi0_corrected, crossings)

    fragment_counter=roi0_corrected.groupby(["frame_number", "fragment"]).size().reset_index()
    bad_fragments=fragment_counter.loc[fragment_counter[0]>1]

    for i, row in bad_fragments.iterrows():
        roi0_corrected.loc[(roi0_corrected["fragment"]==row["fragment"]) & (roi0_corrected["frame_number"]==row["frame_number"]), "fragment"]=np.nan

    roi0_corrected=roi0_corrected.loc[(roi0_corrected["frame_number"]>=first_frame_number) & (roi0_corrected["frame_number"]<last_frame_number)]
    identity_corrected=identity_corrected.loc[(identity_corrected["frame_number"]>=first_frame_number) & (identity_corrected["frame_number"]<last_frame_number)]

    roi0_corrected["chunk"]=roi0_corrected["frame_number"]//chunksize
    identity_corrected["chunk"]=identity_corrected["frame_number"]//chunksize

    identity_corrected["annotation_id"]=[f"{row['frame_number']}_{row['in_frame_index']}" for _, row in identity_corrected.iterrows()]
    roi0_corrected["annotation_id"]=[f"{row['frame_number']}_{row['in_frame_index']}" for _, row in roi0_corrected.iterrows()]

    return identity_corrected, roi0_corrected, score_dist
