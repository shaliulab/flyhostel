import os.path
import logging
import math

import numpy as np
import pandas as pd

from flyhostel.data.pose.constants import chunksize
from flyhostel.data.human_validation.cvat.cvat_integration import get_annotations, cross_machine_human
from flyhostel.data.human_validation.cvat.utils import load_tracking_data, load_machine_data, get_number_of_animals, get_basedir, get_dbfile
from flyhostel.data.human_validation.cvat.fragments import make_identity_singletons, make_identity_tracks
from flyhostel.data.human_validation.cvat.identity import annotate_identity
from flyhostel.data.human_validation.cvat.sqlite3 import write_validated_identity, write_validated_roi0, write_validated_concatenation


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
    impure_fragments=pd.DataFrame({"chunk": [], "fragment": []})

    if lid_fragment_index.shape[0]==0:
        return impure_fragments

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
    if out:
        impure_fragments=pd.concat(out, axis=0)

    return impure_fragments


def test_df(df):
    for _, df2 in df.loc[df["frame_number"].isin(list(range(10194436, 10194442)))].sort_values(["frame_number", "local_identity"]).groupby("frame_number"):
        print(df2)



REDOWNLOAD_FROM_CVAT=True
def integrate_human_annotations(
        experiment, folder, tasks,
        first_frame_number=0, last_frame_number=math.inf,
        redownload=REDOWNLOAD_FROM_CVAT,
    ):
    """
    Add human validated identity tracks to a flyhostel dbfile
    Annotations are downloaded from a locally running CVAT instance and stored in the form of new tables in the dbfile

    This function returns None. Its purpose is to add or regenerate 3 tables in the dbfile of the experiment:
        IDENTITY_VAL: Same information as IDENTITY, but containing human annotations
        ROI_0_VAL: Same information as ROI_0, but containing human annotations
        CONCATENATION_VAL: Same information as CONCATENATION, but containing human annotations

    The identity information is organized in tracks and singletons:

       * An identity track consists of a pair of a fragment identifier and a chunk.
           The track informs which local identity acquired all blobs within an idtrackerai fragment and within the same chunk
           The track has the ability that it can propagate the identity given by a human to one blob in the fragment
               to all other blobs in the same fragment
        * An identity singleton consists of an identity assigned to a blob which has no fragment or it's a crossing
            blobs without fragments are generated when the human marks a fly _de_ _novo_ i.e. idtrackerai did not segment it as a separate fly because
                1) it is missed
                2) it is segmented merged with another fly but the merged blob is not marked as a crossing (even though it is)
            crossings occur when the crossing detector of idtrackerai marks a blob as a crossing

    Throughout the code the terms machine data and human data are used:

        machine_data refers to identity assigments and segmentations produced by the machine, either AI or not (stored in a SQLite file)
        human_data or annotations refers to the identity assignemtns and segmentations produced by a human (stored in a CVAT task)

    A data point (combination of spatial and identity information) can have a value in the field "validated" out of 3 levels

    2: The data point itself is modified by a human
    1: The data point is not modified by a human directly, but some other data point in its fragment is modified by a human
    0: The data point is not modified at all (still may be validated by the human, just not modified) 


    Arguments
        * experiment (str): Identifier of the flyhostel run (FlyHostelX_YX_YYYY-MM-DD_HH-MM-SS)
        * folder (str): Where to save output csv files
        * tasks (list): Task IDs corresponding to this experiment
        * first_frame_number (int): First frame number of the validation tables
        * last_frame_numebr (int): Last frame number of the validation tables
    """

    basedir=get_basedir(experiment)
    number_of_animals=get_number_of_animals(experiment)

    annotations_df, contours=get_annotations(basedir, tasks, redownload=redownload)
    annotations_df["chunk"]=annotations_df["frame_number"]//chunksize

    # load original predictions (machine made)
    logger.info("Loading predictions from frame number %s to %s", first_frame_number, last_frame_number)
    where_statement=f"WHERE frame_number >= {first_frame_number}"
    if ~np.isinf(last_frame_number):
        where_statement+=f" AND frame_number < {last_frame_number}"

    identity_machine, roi_0_machine=load_machine_data(basedir, where=where_statement)

    identity_annotations, roi0_annotations, _=cross_machine_human(
        basedir, identity_machine, roi_0_machine, annotations_df, contours, number_of_animals,
        first_frame_number=first_frame_number, last_frame_number=last_frame_number
    )

    roi_0_machine=roi_0_machine[["frame_number", "in_frame_index", "x", "y", "fragment", "modified", "class_name"]]
    identity_machine=identity_machine[["frame_number", "in_frame_index", "local_identity"]]
    roi_0_machine["validated"]=False
    identity_machine["validated"]=False

    logger.info("Processing...")
    identity_tracks=make_identity_tracks(identity_annotations, roi0_annotations)
    identity_singletons=make_identity_singletons(identity_annotations, roi0_annotations)
    identity_singletons["validated"]=2

    # Generate machine data
    # This is a dataset of data points that combine spatial and identity information purely made by the machine
    machine_data=identity_machine.merge(
        roi_0_machine[["frame_number", "in_frame_index", "fragment", "x", "y", "modified", "class_name"]],
        on=["frame_number", "in_frame_index"]
    )
    machine_data["chunk"]=machine_data["frame_number"]//chunksize
    machine_data["validated"]=0

    # Generate human data
    # This is a dataset of data points that combine spatial and identity information purely made by the human
    human_data=roi0_annotations[[
        "annotation_id", "frame_number", "x", "y", "fragment", "chunk"
    ]].merge(
        identity_annotations[[
            "annotation_id", "local_identity", "is_a_crossing"
        ]], on="annotation_id"
    ).drop("annotation_id", axis=1)
    human_data["validated"]=2

    # Propagate human-made identities through the fragment structure found by the machine 
    machine_data_with_identity_annotations=machine_data.drop(["validated", "local_identity"], axis=1).merge(
        identity_tracks[["chunk", "fragment", "local_identity"]].drop_duplicates(),
        on=["chunk", "fragment"],
        how="inner"
    )
    machine_data_with_identity_annotations["validated"]=1

    # Combine the human data with the machine data whose ids have been modified
    # The rest of the machine data is not included here.
    # This is achieved to the how='inner', which means fragments with no human-made identity assignments are excluded
    human_data_and_propagation_via_fragments=pd.concat([
        human_data,
        machine_data_with_identity_annotations
    ], axis=0).sort_values(["frame_number","local_identity", "validated"], ascending=[True, True, False]).drop_duplicates(
        ["frame_number","local_identity", "validated"]
    )


    # Add blobs that belong to hybrid fragments that are crossings at some point but also separate flies at some other point
    # this happens when YOLOv7 classifies as a single fly a group of >1 flies
    # Capture fragments generated with the FMB (Fragment Must Break) tag
    # These are fragments present in the human annotation but not in the machine data
    # because they are generated by splitting a machine fragment in two
    de_novo_fragments=identity_tracks.merge(
        machine_data[["chunk","fragment"]].drop_duplicates(),
        on=["chunk","fragment"],
        how="left",
        indicator=True
    )
    # keep identity tracks exclusive to the human
    de_novo_fragments=de_novo_fragments.loc[de_novo_fragments["_merge"]=="left_only"].drop("_merge", axis=1)
    # get spatial information of those tracks
    de_novo_fragments=de_novo_fragments.merge(
        roi0_annotations[["in_frame_index", "x", "y", "frame_number"]],
        on=["frame_number", "in_frame_index"],
        how="left"
    )

    de_novo_fragments["validated"]=2
    identity_singletons["modified"]=0
    identity_singletons["class_name"]=None

    de_novo_fragments["modified"]=0
    de_novo_fragments["class_name"]=None
    
    
    # Add blobs from crossing fragments (fragment set to na)
    annotations_without_clean_spatial_machine_data_and_rest_of_machine_data=pd.concat([
        identity_singletons[machine_data.columns],
        de_novo_fragments[machine_data.columns],
        machine_data,
    ], axis=0).reset_index(drop=True).drop_duplicates([
        "frame_number", "local_identity"
    ])

    # First discard:
    # all duplicates come from data_not_corrected (machine annotation) with the same local identity as a human annotation
    # because the data is sorted by frame number and validated (True=human and False=machine)
    # so we will in any case keep the human annotation in lid_fragment_index_nofragm

    # PS The validated data comes first because the concat above has first the human data and then the machine data

    # TODO
    # Write test that verifies in experiment FlyHostel4_6X_2023-08-31_13-00-00  chunk 220 frame_idx 25181 (7090181) that the flies get local_identity 5 and 2
    # Also frame 10194436 from same experiment
    
    # combine all data sources
    new_data=pd.concat([
        human_data_and_propagation_via_fragments,
        annotations_without_clean_spatial_machine_data_and_rest_of_machine_data[machine_data_with_identity_annotations.columns]
    ], axis=0).sort_values(["frame_number", "validated"], ascending=[True, False]).drop_duplicates(["frame_number", "local_identity"])


    # Second discard:
    # Discard remaining predictions with local identity of 0 in frames where a human validation is also present
    # This means such predictions were ignored by the human (because they are wrong)

    validated_frames=new_data.groupby("frame_number").agg({"validated": np.sum}).reset_index()
    validated_frames=validated_frames.loc[validated_frames["validated"]>0]
    new_data["frame_validated"]=False
    new_data.loc[new_data["frame_number"].isin(validated_frames["frame_number"]), "frame_validated"]=True

    new_data=new_data.loc[~((new_data["local_identity"]==0) & (new_data["frame_validated"]==True))]
    new_data["frame_idx"]=new_data["frame_number"]%chunksize

    flies_lid_0=list_flies_with_lid_0(new_data)
    flies_lid_0.to_csv(os.path.join(folder, "flies_lid_0.csv"))

    # roi_0_table.loc[~roi_0_table["class_name"].isna()]
    # Annotate whether the prediction is made by YOLOv7 or pixel segmentation, and if it is, then what YOLOv7 class it has
    #out=new_data.merge(roi_0_machine[["in_frame_index", "frame_number", "class_name"]], on=["frame_number", "in_frame_index"], how="left")

    try:
        # Annotate identity!!!
        out=annotate_identity(new_data, number_of_animals)
    except:
        import ipdb; ipdb.set_trace()

    # Save result!
    out_file=os.path.join(folder, f"{experiment}.feather")
    logger.debug("Saving ---> %s", out_file)
    out.reset_index(drop=True).to_feather(out_file)

    # reports
    fragment_crossing_fraction=qc1(roi0_annotations, identity_annotations)
    fragment_crossing_fraction.to_csv(
        os.path.join(folder, "fragment_crossing_fraction.csv")
    )
    impure_tracks=qc2(identity_tracks)
    if impure_tracks.shape[0] > 0:
        logger.error("%s fragments have more than 1 identity", impure_tracks.shape[0])
        logger.error("Fragments:")
        for _, row in impure_tracks.iterrows():
            logger.error(row)

    impure_tracks.to_csv(
        os.path.join(folder, "impure_tracks.csv")
    )


    discarded_predictions=new_data.loc[(new_data["local_identity"]==0) & (new_data["frame_validated"]==True)]

    discarded_predictions.to_csv(
        os.path.join(
            folder, "discarded_predictions.csv"
        )
    )

    perc_validated_frames=new_data["frame_validated"].mean()*100
    perc_validated_frames=np.round(perc_validated_frames, 2)
    logger.info("Validated frame perc: %s %%", perc_validated_frames)

    number_of_animals_qc=new_data.groupby("frame_number").size().reset_index()
    number_of_animals_qc["chunk"]=number_of_animals_qc["frame_number"]//chunksize
    number_of_animals_qc["frame_idx"]=number_of_animals_qc["frame_number"]%chunksize
    number_of_animals_qc.columns=["frame_number", "number_of_animals", "chunk", "frame_idx"]
    number_of_animals_qc_fail=number_of_animals_qc.loc[number_of_animals_qc["number_of_animals"]!=number_of_animals]

    number_of_animals_qc_fail.to_csv(
        os.path.join(
            folder, "number_of_animals_qc_fail.csv"
        )
    )

    # prepare validation tables of sqlite3 file

    df_identity=out[["frame_number", "in_frame_index", "local_identity", "identity", "validated"]].reset_index(drop=True)
    
    groupby=df_identity.groupby("local_identity")
    logger.debug("Number of frames seen")
    logger.debug("\n%s", groupby.size())
    logger.debug("First frame seen")
    logger.debug("\n%s", groupby.first())
    logger.debug("Last frame seen")
    logger.debug("\n%s", groupby.last())

    if df_identity["identity"].isna().any():
        logger.error("NaN identities:")
        nan_rows=df_identity["identity"].isna()
        logger.error(df_identity.loc[nan_rows])
        df_identity=df_identity.loc[~nan_rows]


    df_identity["identity"]=df_identity["identity"].astype(np.int32)
    df_identity["local_identity"]=df_identity["local_identity"].astype(np.int32)

    df_roi0=out[["frame_number", "in_frame_index", "x", "y", "fragment", "modified", "class_name", "validated"]].reset_index(drop=True)
    df_roi0["chunk"]=df_roi0["frame_number"]//chunksize
    df_roi0["frame_idx"]=df_roi0["frame_number"]%chunksize

    df_concatenation=df_identity[["frame_number", "local_identity", "identity"]]
    df_concatenation["chunk"]=df_concatenation["frame_number"]//chunksize
    df_concatenation=df_concatenation.groupby(["chunk", "identity"]).agg({"local_identity": np.mean}).reset_index()

    for name, df in [("IDENTITY_VAL", df_identity), ("ROI_0_VAL", df_roi0), ("CONCATENATION_VAL", df_concatenation)]:
        df.reset_index(drop=True).to_feather(
            os.path.join(
                folder, f"{name}.feather"
            )
        )

    groupby=df_identity.groupby("identity")
    logger.debug("Number of frames seen")
    logger.debug(groupby.size())
    logger.debug("First frame seen")
    logger.debug(groupby.first())
    logger.debug("Last frame seen")
    logger.debug(groupby.last())



def save_human_annotations(experiment, folder):

    basedir=get_basedir(experiment)
    dbfile=get_dbfile(basedir)


    dfs={}
    for name in ["IDENTITY_VAL", "ROI_0_VAL", "CONCATENATION_VAL"]:
        dfs[name]=pd.read_feather(
            os.path.join(
                folder, f"{name}.feather"
            )
        )
    write_validated_roi0(dfs["ROI_0_VAL"], dbfile)
    write_validated_identity(dfs["IDENTITY_VAL"], dbfile)
    write_validated_concatenation(dfs["CONCATENATION_VAL"], dbfile)
        


def list_flies_with_lid_0(data):
    missing_identity=data.loc[data["local_identity"]==0][["chunk", "fragment"]].drop_duplicates()
    out=[]
    for i, row in missing_identity.iterrows():
        out.append(data.loc[
            (data["chunk"]==row["chunk"]) & (data["fragment"]==row["fragment"])
        ])
    if out:
        out=pd.concat(out, axis=0)
    else:
        out=pd.DataFrame({"chunk": [], "fragment": []})
    return out