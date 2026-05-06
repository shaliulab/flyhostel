import h5py
import itertools
import logging
import os.path
import time
import pandas as pd
import numpy as np
import cupy as cp
import cudf
from tqdm.auto import tqdm
from flyhostel.data.pose.loaders.centroids import flyhostel_sleep_annotation_primitive as flyhostel_sleep_annotation
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.groups.group import FlyHostelGroup
from flyhostel.data.pose.constants import bodyparts as BODYPARTS
from flyhostel.data.interactions.touch import preprocess_data, check_intersections

logger=logging.getLogger(__name__)
DEBUG=True


def infer_touch(pose, mask, bodyparts, n_jobs=1):
    mask=flatten_data(mask)
    df = pose.merge(mask[["id", "frame_number", "nn"]], on=["id", "frame_number"], how="inner")
    df = preprocess_data(df, bodyparts=bodyparts)

    # Run the intersection check
    touching_pairs = list(itertools.chain(*check_intersections(df, mask=mask, n_jobs=n_jobs)))
    touch_interactions=pd.DataFrame.from_records(touching_pairs, columns=["frame_number", "id", "nn", "edge_distance"])

    print(f'Number of touching pairs: {len(touching_pairs)}')
    print('Example touching pairs:', touching_pairs[:5])
    return touch_interactions

def flatten_data(df):
    """
    Represent every pairwise interaction twice,
    once per animal involved    
    """
    if isinstance(df, cudf.DataFrame):
        xf=cudf
    else:
        xf=pd

    paired_df=df.rename({"id": "nn", "nn": "id"}, axis=1)
    if all(c in paired_df.columns for c in ["id_bodypart", "nn_bodypart"]):
        paired_df.rename({
            "id_bodypart": "nn_bodypart",
            "nn_bodypart": "id_bodypart",
        }, axis=1, inplace=True)


    df=xf.concat([
        df, paired_df
    ], axis=0).sort_values("frame_number")
    return df


def download_from_gpu(df):
    try:
        df_cpu=df.to_pandas()
        del df
    except Exception:
        df_cpu=df
    
    return df_cpu

def infer_neighbors_by_time_partitions(
        group,
        partition_size=None,
        step=1,
        store="RAM"
    ):

    """
    Call infer_neighbors for one temporal partition of the dataset at a time 

    partition_size (int): Number of seconds making each partition
    """

    neighbors_l=[]

    min_t=group.dt["t"].min()
    max_t=group.dt["t"].max()

    interval=(min_t, max_t)
    partition_size=min(partition_size, interval[1]-interval[0])
    t0s=np.arange(interval[0], interval[1], partition_size)
    t1s=t0s+partition_size

    for i, (min_time, max_time) in tqdm(enumerate(zip(t0s, t1s)), desc="Inferring interactions"):

        zt0=round(min_time/3600, 2)
        zt1=round(max_time/3600, 2)

        before=time.time()
        print(f"Partition {i} ({zt0} - {zt1})")

        centroid_dataset=group.dt.loc[
            (group.dt["t"]>=min_time)&(group.dt["t"]<max_time)
        ]
        after=time.time()

        logger.debug("%s seconds to filter partition data", after-before)
        dt_neighbors_=find_neighbors(
            group,
            centroid_dataset,
            step=step,
            chunksize=group.chunksize
        )
        if dt_neighbors_ is not None:
            dt_neighbors_cpu=download_from_gpu(dt_neighbors_)
            n_rows=dt_neighbors_cpu.shape[0]
            if store=="RAM":
                neighbors_l.append(dt_neighbors_cpu)
            elif store=="DISK":
                raise NotImplementedError()
                dt_neighbors_cpu.to_feather(f"interactions_{str(i+1).zfill(3)}.feather")
            logger.info(
                "Collected %s rows of interaction data in partition %s/%s",
                n_rows, i+1, len(t0s)
            )


            del dt_neighbors_
        else:
            logger.warning("No interactions between %s and %s", zt0, zt1)

    if store=="RAM":
        # put together all partitions
        if len(neighbors_l)==0:
            dt_neighbors=None
        else:
            dt_neighbors=pd.concat(neighbors_l, axis=0).reset_index(drop=True)
    elif store=="DISK":
        raise NotImplementedError()
    return dt_neighbors


def analyze_group(
        group, fps=50,
        useGPU=True, interval=None,
        partition_size=None,
        **kwargs
    ):
    """
    Detect interactions between animals in a group
    This is the function that runs as the first process of the interactions pipeline
    """

    # load the x y coordinates of the centroids of each animal over time
    group.dt=group.load_centroid_data(fps=fps, useGPU=useGPU)
    
    if useGPU:
        assert isinstance(group.dt, cudf.DataFrame)

    if interval is None:
        interval=(
            group.dt["frame_number"].min(),
            group.dt["frame_number"].max()+1
        )

    # interactions
    neighbors=infer_neighbors_by_time_partitions(
        group,
        step=1,
        partition_size=partition_size,
        **kwargs
    )

    group.neighbors=neighbors
    if len(neighbors)==0:
        return None

    return group


def find_neighbors(group, dt, step, chunksize):
    """
    Some preprocessing and calls the group method
    """

    dt["centroid_x"]=dt["center_x"]
    dt["centroid_y"]=dt["center_y"]

    # find frames where the centroid of at least two flies it at most dist_max_mm mm from each other
    dt_neighbors=group.find_neighbors(
        dt[["id", "frame_number", "centroid_x", "centroid_y", "t"]],
        dist_max_mm=group.dist_max_mm,
        step=step,
    )

    dt_neighbors["chunk"]=dt_neighbors["frame_number"]//chunksize
    dt_neighbors["frame_idx"]=dt_neighbors["frame_number"]%chunksize
    return dt_neighbors

def annotate_interactions(group, framerate, time_window_length=10, asleep_annotation_age=10):

    assert group.dt_sleep is not None

    legs=[bp for bp in BODYPARTS if "L" in bp]
    core=["head", "thorax", "abdomen"]

    interactions=group.interactions
    
    ids=sorted(interactions["id"].unique())
    print(f"ids: {ids}")
    
    # ids=sorted(group.dt_sleep["id"].cat.categories.tolist())
    dt_sleep=group.dt_sleep.loc[group.dt_sleep["id"].isin(ids)]

    dt_sleep["id"]=pd.Categorical(dt_sleep["id"].astype(str), categories=ids)
    interactions["id"]=pd.Categorical(interactions["id"], categories=ids)
    interactions["frame_number"]=interactions["frame_number"].astype(np.int32)
    interactions.sort_values(["frame_number", "id"], inplace=True)

    # how many seconds in the past should the sleep annotation come from
    dt_sleep["frame_number"]+=framerate*asleep_annotation_age
    # this is useful so that the sleep state at the time of the interaction can be defined based on the behavior
    # from a little bit before in time

    # annotate sleep state of the interaction
    hits=pd.merge_asof(
        interactions,
        dt_sleep, by="id",
        on="frame_number",
        direction="backward",
        tolerance=framerate*time_window_length
    )

    # remove all data until the first annotation of sleep
    first_non_na=hits.groupby("id").apply(lambda df: df.iloc[np.where(~df["asleep"].isna())[0][0]])[["frame_number"]].reset_index()
    selectors=[]
    for _, row in first_non_na.iterrows():
        selectors.append(
            ((hits["id"]==row["id"]) & (hits["frame_number"]>=row["frame_number"])).values
        )
    keep_rows=np.any(np.stack(selectors), axis=0)
    hits=hits.loc[keep_rows]
    
    # keep interactions where legs and core are involved, discard the rest
    hits=hits.loc[(hits["nn_bodypart"].isin(legs + core)) & (hits["id_bodypart"].isin(legs + core))].sort_values("distance_bodypart_mm")

    # annotate local_identity, keys and videos
    hits=hits.merge(group.pose[["local_identity", "id", "frame_number"]].to_pandas(), on=["id", "frame_number"], how="left")
    hits["keys"]=hits["animal"].str.slice(start=0, stop=33) + "_" + hits["chunk"].apply(lambda x: str(x).zfill(6)) + "_" + hits["local_identity"].apply(lambda x: str(x).zfill(3))
    hits["videos"]=hits["keys"]+"/"+ hits["chunk"].apply(lambda x: str(x).zfill(6)) + ".mp4"
    return hits


def compute_experiment_neighbors(
        group,
        fps=15,
        partition_size=None,
        interval=None,
        **kwargs
    ):
    """
    CLI entry point

    """
    analyze_group(
        group,
        fps=fps,
        partition_size=partition_size,
        interval=interval,
        **kwargs
    )
    return group



def initialize_group(experiment, identities, **kwargs):

    loaders=[
        FlyHostelLoader(
            experiment=experiment,
            identity=identity,
            chunks=range(0, 400),
            identity_table="IDENTITY_VAL",
            roi_0_table="ROI_0_VAL"
        )
        for identity in identities
    ]

    group=FlyHostelGroup.from_list(loaders, protocol="centroids", **kwargs)
    return group
