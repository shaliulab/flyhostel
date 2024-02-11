import os.path
import logging
import math

import numpy as np
import pandas as pd

from flyhostel.data.pose.constants import chunksize
from flyhostel.data.human_validation.cvat.cvat_integration import get_annotations, cross_tracking_annotation
from flyhostel.data.human_validation.cvat.utils import load_tracking_data, load_data, get_number_of_animals, get_basedir
from flyhostel.data.human_validation.cvat.fragments import make_annotation_wo_fragment_index, make_fragment_index

logger=logging.getLogger(__name__)

def qc1(roi0, ident):
    """
    Check all fragments have blobs that are always or never crossing
    """
    control=roi0.merge(
        ident.drop(["validated", "chunk"], errors="ignore", axis=1), on=["frame_number", "in_frame_index"],
        how="left",
    )
    control["frame_idx"]=control["frame_number"]%chunksize

    control_summary=control.groupby(["chunk","fragment"]).agg({"is_a_crossing": np.mean}).reset_index()
    # n_bad_fragments=control_summary.loc[~control_summary["is_a_crossing"].isin([0, 1])].shape[0]

    return control_summary

def qc2(lid_fragment_index):
    """
    Check all fragments have a unique local identity
    """

    fragment_purity_table=lid_fragment_index.groupby(["chunk", "fragment"]).apply(lambda df: len(df["local_identity"].unique())==1).reset_index()
    impure_fragments=fragment_purity_table.loc[~fragment_purity_table[0]]
    
    # fragment_purity_table[0].mean() * 100
    
    out=[]
    for i, row in impure_fragments.iterrows():
        out.append(
            lid_fragment_index.loc[
                (lid_fragment_index["chunk"]==row["chunk"]) & (lid_fragment_index["fragment"]==row["fragment"])
            ]
        )
    impure_fragments=pd.concat(out, axis=0)
    return impure_fragments



def integrate_human_annotations(experiment, folder, tasks, min_t=-math.inf, max_t=math.inf):

    basedir=get_basedir(experiment)
    number_of_animals=get_number_of_animals(experiment)
    
    annotations_df, contours=get_annotations(basedir, tasks)
    frame_numbers=sorted(annotations_df["frame_number"].unique())

    # load machine tracking data for the frames where a human annotation is found
    tracking_data=load_tracking_data(basedir, frame_numbers)
    ident, roi0, _=cross_tracking_annotation(basedir, tracking_data, annotations_df, contours, number_of_animals)
    roi0["chunk"]=roi0["frame_number"]//chunksize
    ident["chunk"]=ident["frame_number"]//chunksize
    tracking_data["chunk"]=tracking_data["frame_number"]//chunksize

    # override machine predicitons with human annotations
    identity_table, roi0_table, store_index_table=load_data(basedir, where=f"WHERE frame_number >= {frame_numbers[0]} AND frame_number < {frame_numbers[-1]+1}")
    store_index_table=store_index_table.loc[
        (store_index_table["t"] >= min_t) &
        (store_index_table["t"] < max_t)
    ]

    first_frame_number, last_frame_number=store_index_table["frame_number"].min(), store_index_table["frame_number"].max()

    identity_table=identity_table.loc[
        (identity_table["frame_number"] >= first_frame_number) &
        (identity_table["frame_number"] < last_frame_number)
    ]

    roi0_table=roi0_table.loc[
        (roi0_table["frame_number"] >= first_frame_number) &
        (roi0_table["frame_number"] < last_frame_number)
    ]
    roi0_table=roi0_table.merge(store_index_table[["frame_number", "t"]], on="frame_number")


    #1 generate index of chunk - fragment - local_identity
    lid_fragment_index=make_fragment_index(ident, roi0)

    fragment_crossing_fraction=qc1(roi0, ident)
    fragment_crossing_fraction.to_csv(
        os.path.join(folder, "fragment_crossing_fraction.csv")
    )
    impure_fragments=qc2(lid_fragment_index)
    impure_fragments.to_csv(
        os.path.join(folder, "impure_fragments.csv")
    )


    lid_fragment_index_nofragm=make_annotation_wo_fragment_index(ident, roi0)

    # 2 merge roi0_table and identity_table to bring fragment annotation
    data=identity_table.merge(roi0_table[["frame_number", "in_frame_index", "fragment", "x", "y", "t"]], on=["frame_number", "in_frame_index"])


    # 3 split data according to whether there is a match of chunk + fragment in the lid_fragment_index or not
    data_corrected=data.drop(["validated", "local_identity"], axis=1).merge(lid_fragment_index[["chunk", "fragment", "local_identity", "validated"]].drop_duplicates(), on=["chunk", "fragment"], how="inner")
    data_not_corrected=data.merge(lid_fragment_index[["chunk", "fragment"]].drop_duplicates(), on=["chunk", "fragment"], how="left", indicator=True)
    
    # Add blobs that belong to hybrid fragments that are crossings at some point but also separate flies at some other point
    # this happens when YOLOv7 classifies as a single fly a group of >1 flies
    extra_fragments=lid_fragment_index.merge(data_not_corrected[["chunk","fragment"]], on=["chunk","fragment"], how="left", indicator=True)
    extra_fragments=extra_fragments.loc[extra_fragments["_merge"]=="left_only"].drop("_merge", axis=1)
    extra_fragments=extra_fragments.merge(roi0[["in_frame_index", "x", "y", "frame_number"]], on=["frame_number", "in_frame_index"], how="left")
    data_not_corrected=data_not_corrected.loc[data_not_corrected["_merge"]=="left_only"].drop("_merge", axis=1)
    data_not_corrected=pd.concat([data_not_corrected, extra_fragments], axis=0)

    # Add blobs from crosing fragments (fragment set to na)
    data_not_corrected=pd.concat([lid_fragment_index_nofragm, data_not_corrected], axis=0).reset_index(drop=True)
    
    # First discard:
    # all duplicates come from data_not_corrected (machine annotation) with the same local identity as a human annotation
    # because the data is sorted by frame number and validated (True=human and False=machine)
    # so we will in any case keep the human annotation in lid_fragment_index_nofragm

    # PS The validated data comes first because the concat above has first the human data and then the machine data
    data_not_corrected.drop_duplicates(["frame_number", "local_identity"], inplace=True)

    # combine all data sources
    new_data=pd.concat([
        data_corrected[data_corrected.columns],
        data_not_corrected[data_corrected.columns]
    ], axis=0).sort_values(["frame_number", "validated"], ascending=[True, False]).drop_duplicates(["frame_number", "local_identity"])


    # Second discard:
    # Discard remaining predictions with local identity of 0 in frames where a human validation is also present
    # This means such predictions were ignored by the human (because they are wrong)
    validated_frames=new_data.groupby("frame_number").agg({"validated": np.sum}).reset_index()
    validated_frames=validated_frames.loc[validated_frames["validated"]>0]
    new_data["frame_validated"]=False
    new_data.loc[new_data["frame_number"].isin(validated_frames["frame_number"]), "frame_validated"]=True
    discarded_predictions=new_data.loc[(new_data["local_identity"]==0) & (new_data["frame_validated"]==True)]

    discarded_predictions.to_csv(
        os.path.join(
            folder, "discarded_predictions.csv"
        )
    )

    new_data=new_data.loc[~((new_data["local_identity"]==0) & (new_data["frame_validated"]==True))]
    new_data["frame_idx"]=new_data["frame_number"]%chunksize

    perc_validated_frames=new_data["frame_validated"].mean()*100
    perc_validated_frames=np.round(perc_validated_frames, 2)
    logger.info("Validated frame fraction %s", perc_validated_frames)



    number_of_animals_qc=new_data.groupby("frame_number").size().reset_index()
    number_of_animals_qc["chunk"]=number_of_animals_qc["frame_number"]//chunksize
    number_of_animals_qc["frame_idx"]=number_of_animals_qc["frame_number"]%chunksize
    number_of_animals_qc.loc[number_of_animals_qc[0]!=number_of_animals]

    flies_lid_0=list_flies_with_lid_0(new_data)
    flies_lid_0.to_csv(os.path.join(folder, "flies_lid_0.csv"))


    # roi0_table.loc[~roi0_table["class_name"].isna()]
    # Annotate whether the prediction is made by YOLOv7 or pixel segmentation, and if it is, then what YOLOv7 class it has
    out=new_data.merge(roi0_table[["in_frame_index", "frame_number", "class_name", "modified"]], on=["frame_number", "in_frame_index"], how="left")
    
    # Save result!
    out.reset_index(drop=True).to_feather(os.path.join(folder, f"{experiment}.feather"))



def list_flies_with_lid_0(data):
    missing_identity=data.loc[data["local_identity"]==0][["chunk", "fragment"]].drop_duplicates()
    out=[]
    for i, row in missing_identity.iterrows():
        print("##")
        out.append(data.loc[
            (data["chunk"]==row["chunk"]) & (data["fragment"]==row["fragment"])
        ])
    out=pd.concat(out, axis=0)
    return out


# experiment="FlyHostel4_6X_2023-08-31_13-00-00"
# tasks=[1210, 1211, 1212]
