import yaml
import os.path
import logging
import glob
logger=logging.getLogger(__name__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flyhostel.utils import (
    get_pixels_per_mm,
    get_number_of_animals,
    get_basedir,
    get_chunksize
)

def qc1(roi0, ident, chunksize):
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

def make_report(folder, identity_tracks, roi0_annotations, identity_annotations, new_data, number_of_animals, chunksize):

    fragment_crossing_fraction=qc1(roi0_annotations, identity_annotations, chunksize)
    fragment_crossing_fraction.to_csv(
        os.path.join(folder, "fragment_crossing_fraction.csv")
    )
    impure_tracks=qc2(identity_tracks)
    impure_tracks.to_csv("impure_tracks.csv")

    if impure_tracks.shape[0] > 0:
        logger.error("%s fragments have more than 1 identity", impure_tracks.drop_duplicates(["chunk", "fragment"]).shape[0])
        logger.error("Fragments:")

        counts=impure_tracks.groupby(["fragment", "local_identity", "chunk"]).size().reset_index(name="count")
        loser_lids=counts.groupby(["chunk", "fragment"], group_keys=False).apply(lambda df: df.loc[df["count"]==df["count"].min()]).reset_index(drop=True)
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


def jump_report(out, folder, number_of_animals, chunksize, pixels_per_mm, images_subfolder=None):
    
    MAX_DIST_IN_ONE_FRAME_MM=1
    MAX_DIST_IN_ONE_FRAME=pixels_per_mm*MAX_DIST_IN_ONE_FRAME_MM
    max_diff=1
        

    if images_subfolder is not None:
        files=glob.glob(f"{folder}/{images_subfolder}/*png")
        tokens=[os.path.basename(file).split("_") for file in files]
        records=[(int(token[0]), "_".join(token[1:]).split(".")[0]) for token in tokens]
        frames_index=pd.DataFrame.from_records(records, columns=["frame_number", "tag"]).sort_values("frame_number")
        diff=frames_index["frame_number"].diff().tolist()[1:]
        diff.append(np.nan)
        frames_index["until_next"]=diff
        frames_index["seen"]=True  
        frames_index.loc[
            (frames_index["tag"] == "checkpoint") |
            ((frames_index["tag"].shift(-1) == "checkpoint") & (frames_index["tag"] == "start")) |
            ((frames_index["tag"].shift() == "checkpoint") & (frames_index["tag"] == "end")),
            "keep"
        ]=True


        with open(f"{folder}/{images_subfolder}/metadata.yaml", "r") as handle:
            metadata=yaml.load(handle, yaml.SafeLoader)
            max_diff=metadata["max_diff"]


    dist_df=[]
    for identity in range(1, number_of_animals+1):
        out_fly=out.loc[out["identity"]==identity]
        
        # x and y are in pixels
        distance=np.sqrt((np.diff(out_fly[["x","y"]], axis=0)**2).sum(axis=1))
        df=pd.DataFrame({
            "distance": distance,
            "identity": identity,
            "local_identity": out_fly["local_identity"].iloc[:-1],
            "x": out_fly["x"].iloc[:-1],
            "y": out_fly["y"].iloc[:-1],
            "frame_number": out_fly["frame_number"].iloc[:-1]}
        )
        if images_subfolder is not None:
            df=df.merge(frames_index, on="frame_number", how="left").sort_values("frame_number")
            df.loc[df["seen"].isna(), "seen"]=False
            df.loc[df["keep"].isna(), "keep"]=False
            df_backup=df.copy()
            df=df.loc[(df["keep"]==True)]

        diff=np.concatenate([
            df["frame_number"].diff().iloc[1:].tolist(),
            [np.nan]
        ]).astype(np.int64)


        if not (diff==max_diff).all():
            if images_subfolder is None:
                selector=(diff>max_diff)
            else:
                selector=(diff>max_diff) & (df["tag"]!="end")

            frames=df["frame_number"].loc[selector].tolist()
            gaps=diff[selector].tolist()

            logger.error("Tracks missing for identity in frames %s: %s", identity, frames)
            logger.error("Tracks missing for identity with gaps of %s: %s frames", identity, gaps)
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

    
    na_dist=dist_df["distance"].isna()
    if na_dist.any():
        logger.warning(dist_df.loc[na_dist])

    dist_df.loc[~dist_df["x"].isna(), "x"]=dist_df.loc[~dist_df["x"].isna(), "x"].astype(int)
    dist_df.loc[~dist_df["y"].isna(), "y"]=dist_df.loc[~dist_df["y"].isna(), "y"].astype(int)
    
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
    chunksize=get_chunksize(experiment)
    pixels_per_mm=get_pixels_per_mm(experiment)
    jump_report(out, folder, number_of_animals=number_of_animals, chunksize=chunksize, pixels_per_mm=pixels_per_mm)


MAX_JUMP_MM=1


def find_problem_frames(out, number_of_animals, max_jump=None):
    """
    Problems left in the validated tracking, one row per frame and problem

    A frame has a problem if
        * an identity is missing or appears more than once
        * a blob has no identity
        * an identity moves more than max_jump pixels since the previous frame

    Arguments
        * out (pd.DataFrame): validated tracking with frame_number, identity, x, y
        * number_of_animals (int)
        * max_jump (float): in pixels

    Returns
        pd.DataFrame with columns frame_number, problem
    """
    problems=[]
    out=out[["frame_number", "identity", "x", "y"]]

    no_identity=out.loc[out["identity"].isna(), ["frame_number"]].drop_duplicates()
    no_identity["problem"]="blob without identity"
    problems.append(no_identity)

    identified=out.loc[~out["identity"].isna()].copy()
    identified["identity"]=identified["identity"].astype(np.int64)
    counts=identified.groupby(["frame_number", "identity"]).size().unstack("identity", fill_value=0)
    counts=counts.reindex(columns=range(1, number_of_animals+1), fill_value=0)
    counts=counts.reindex(out["frame_number"].unique(), fill_value=0)
    for identity in counts.columns:
        for selector, problem in [(counts[identity]==0, "missing"), (counts[identity]>1, "duplicated")]:
            frames=counts.index[selector.values]
            problems.append(pd.DataFrame({"frame_number": frames, "problem": f"identity {identity} {problem}"}))

    # jumps, only between consecutive frames of identities that appear once in both
    single=identified.loc[~identified.duplicated(["frame_number", "identity"], keep=False)]
    single=single.sort_values(["identity", "frame_number"])
    same_track=(single["identity"].diff()==0) & (single["frame_number"].diff()==1)
    distance=np.sqrt(single["x"].diff()**2 + single["y"].diff()**2)

    if max_jump is not None:
        jumps=single.loc[same_track & (distance>max_jump)]

        extra_problems=pd.DataFrame({
            "frame_number": jumps["frame_number"].values,
            "problem": [f"identity {int(i)} jumps" for i in jumps["identity"]],
        })

        problems.append(extra_problems)

    problems=pd.concat(problems, axis=0)
    problems["frame_number"]=problems["frame_number"].astype(np.int64)
    return problems.sort_values(["frame_number", "problem"]).reset_index(drop=True)


def find_problem_blocks(problems, cvat_frames):
    """
    Group the frames with problems into blocks of consecutive frames of a CVAT task

    Arguments
        * problems (pd.DataFrame): output of find_problem_frames
        * cvat_frames (dict): frame_number -> [task, job, task frame],
            as in the index of idtrackerai_validator_server.cvat.build_index

    Returns
        pd.DataFrame with one row per block: task, job, task_frame (of the first frame),
        first_frame_number, last_frame_number, n_frames, problems
    """
    columns=["task", "job", "task_frame", "first_frame_number", "last_frame_number", "n_frames", "problems"]
    index=pd.DataFrame(
        [(int(fn), *location) for fn, location in cvat_frames.items()],
        columns=["frame_number", "task", "job", "task_frame"]
    )
    frames=index.merge(problems, on="frame_number", how="inner")
    if frames.shape[0]==0:
        return pd.DataFrame(columns=columns)

    frames=frames.groupby(["task", "job", "task_frame", "frame_number"])["problem"].apply(list).reset_index()
    frames=frames.sort_values(["task", "task_frame"]).reset_index(drop=True)
    new_block=(frames["task"].diff()!=0) | (frames["task_frame"].diff()!=1)
    frames["block"]=new_block.cumsum()

    blocks=[]
    for _, block in frames.groupby("block"):
        first=block.iloc[0]
        summary=pd.Series([p for problems in block["problem"] for p in problems]).value_counts(sort=False)
        blocks.append((
            first["task"], first["job"], first["task_frame"],
            first["frame_number"], block["frame_number"].iloc[-1], block.shape[0],
            ", ".join(f"{problem} ({count} frame{'' if count == 1 else 's'})" for problem, count in summary.items()),
        ))
    return pd.DataFrame.from_records(blocks, columns=columns)


def cvat_problem_report(out, folder, experiment, number_of_animals, chunksize, pixels_per_mm):
    """
    Print the CVAT link to the first frame of every block of CVAT frames
    that still has a problem after integrating the human annotations,
    and save the blocks to <folder>/cvat_problem_blocks.csv
    """
    from idtrackerai_validator_server.cvat import build_index, configured, public_base, frame_url

    if not configured():
        logger.warning("CVAT is not configured, cannot list the CVAT frames with problems")
        return None

    try:
        cvat_frames=build_index(experiment)["frames"]
    except Exception as error:
        logger.warning("Cannot list the CVAT frames of %s: %s", experiment, error)
        return None

    # only frames covered by this integration (and the frame before, to detect jumps)
    first_frame_number=out["frame_number"].min()
    last_frame_number=out["frame_number"].max()
    cvat_frames={
        fn: location for fn, location in cvat_frames.items()
        if first_frame_number <= int(fn) <= last_frame_number
    }
    frame_numbers=set(int(fn) for fn in cvat_frames)
    frame_numbers|=set(fn-1 for fn in frame_numbers)
    out=out.loc[out["frame_number"].isin(frame_numbers)]

    problems=find_problem_frames(out, number_of_animals, max_jump=pixels_per_mm*MAX_JUMP_MM)
    # CVAT frames absent from the output have no flies at all
    empty=sorted(set(int(fn) for fn in cvat_frames) - set(out["frame_number"].unique()))
    problems=pd.concat([problems, pd.DataFrame({"frame_number": empty, "problem": "no flies"})], axis=0)

    blocks=find_problem_blocks(problems, cvat_frames)
    base_url=public_base()
    blocks["url"]=[
        frame_url(base_url, row["task"], row["job"], row["task_frame"])
        for _, row in blocks.iterrows()
    ]
    blocks.to_csv(os.path.join(folder, "cvat_problem_blocks.csv"))

    print(f"{blocks.shape[0]} blocks of CVAT frames still have problems after validation")
    for _, row in blocks.iterrows():
        chunk=row["first_frame_number"]//chunksize
        print(
            f"Frames {row['first_frame_number']}-{row['last_frame_number']} "
            f"(chunk {chunk}, {row['n_frames']} frames): {row['problems']}\n    {row['url']}"
        )
    return blocks
