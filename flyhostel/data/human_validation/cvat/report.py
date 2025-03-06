import os.path
import logging
logger=logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flyhostel.data.pose.constants import chunksize
from flyhostel.utils import get_number_of_animals, get_basedir

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

def make_report(out, folder, identity_tracks, roi0_annotations, identity_annotations, new_data, number_of_animals):

    fragment_crossing_fraction=qc1(roi0_annotations, identity_annotations)
    fragment_crossing_fraction.to_csv(
        os.path.join(folder, "fragment_crossing_fraction.csv")
    )
    impure_tracks=qc2(identity_tracks)

    if impure_tracks.shape[0] > 0:
        logger.error("%s fragments have more than 1 identity", impure_tracks.drop_duplicates(["chunk", "fragment"]).shape[0])
        logger.error("Fragments:")

        counts=impure_tracks.groupby(["fragment", "local_identity", "chunk"]).size().reset_index(name="count")
        loser_lids=counts.groupby(["chunk", "fragment"]).apply(lambda df: df.loc[df["count"]==df["count"].min()]).reset_index(drop=True)
        loser_lids=loser_lids.merge(impure_tracks[["chunk", "fragment", "frame_idx", "local_identity"]], on=["chunk", "fragment", "local_identity"])
        logger.error("\n%s", loser_lids)
        logger.error("\n%s", counts)

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
    jump_report(out, folder, number_of_animals)


def jump_report(out, folder, number_of_animals):
    
    MAX_DIST_IN_ONE_FRAME=20

    dist_df=[]
    for identity in range(1, number_of_animals+1):
        out_fly=out.loc[out["identity"]==identity]
        
        distance=np.sqrt((np.diff(out_fly[["x","y"]], axis=0)**2).sum(axis=1))
        df=pd.DataFrame({
            "distance": distance,
            "identity": identity,
            "local_identity": out_fly["local_identity"].iloc[:-1],
            "x": out_fly["x"].iloc[:-1],
            "y": out_fly["y"].iloc[:-1],
            "frame_number": out_fly["frame_number"].iloc[:-1]}
        )
        diff=df["frame_number"].diff()

        if not (diff==1).all():
            logger.error("Tracks missing for identity in frames %s: %s", identity, df["frame_number"].loc[diff!=1].tolist())
            logger.error("Tracks missing for identity with gaps of %s: %s frames", identity, diff[diff!=1].tolist())
        dist_df.append(df)

    dist_df=pd.concat(dist_df, axis=0)
    dist_df.sort_values(["distance", "local_identity"], ascending=[False, True], inplace=True)
    dist_df["frame_idx"]=dist_df["frame_number"]%chunksize
    dist_df["chunk"]=dist_df["frame_number"]//chunksize
    
    index=dist_df.loc[dist_df["distance"]>MAX_DIST_IN_ONE_FRAME, ["frame_number", "local_identity"]]
    indexm1=index.copy()
    indexp1=index.copy()

    indexm1["frame_number"]-=1
    indexp1["frame_number"]+=1
     
    index=pd.concat([
        indexm1, index, indexp1
    ], axis=0)\
        .drop_duplicates(["frame_number", "local_identity"])

    dist_df["distance"]=dist_df["distance"].astype(int)
    dist_df["x"]=dist_df["x"].astype(int)
    dist_df["y"]=dist_df["y"].astype(int)

    database=dist_df\
        .merge(index, on=["frame_number", "local_identity"], how="right")\
        .sort_values(["local_identity", "frame_number"])\
        .reset_index(drop=True)
    
    print(database.duplicated(["frame_number", "local_identity"]).sum())

    
    database.to_csv(
        os.path.join(
            folder, "jumps_database.csv"
        )
    )

    for identity in range(1, number_of_animals+1):
        df=dist_df.loc[dist_df["identity"]==identity].sort_values("frame_number")
        x=df.loc[df["distance"]>MAX_DIST_IN_ONE_FRAME]
        x=x.iloc[::10]
        plt.plot(x["frame_number"]/chunksize, x["distance"])
        plt.savefig(
            os.path.join(
                folder, f"identity_{str(identity).zfill(2)}_jumps.png"
            )
        )
        plt.clf()


def jump_report_from_outputs(experiment, folder):

    basedir=get_basedir(experiment)
    number_of_animals=get_number_of_animals(experiment)
    roi_0_val=pd.read_feather(f"{basedir}/flyhostel/validation/ROI_0_VAL.feather")
    identity_val=pd.read_feather(f"{basedir}/flyhostel/validation/IDENTITY_VAL.feather")
    out=roi_0_val.merge(identity_val.drop(["validated"], axis=1, errors="ignore"), on=["frame_number", "in_frame_index"], how="left")
    jump_report(out, folder, number_of_animals=number_of_animals)
