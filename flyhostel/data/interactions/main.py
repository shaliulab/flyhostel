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
from flyhostel.data.pose.constants import SQUARE_WIDTH, SQUARE_HEIGHT
from flyhostel.data.pose.constants import framerate as FRAMERATE
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.data.interactions.touch import preprocess_data, check_intersections

logger=logging.getLogger(__name__)
DEBUG=True

def compute_experiment_interactions_v1(experiment, number_of_animals, output=None, dist_max_mm=2.5, min_interaction_duration=.3):
    loaders=[
        FlyHostelLoader(
            experiment=experiment,
            identity=identity,
            chunks=range(0, 400),
            identity_table="IDENTITY_VAL",
            roi_0_table="ROI_0_VAL"
        )
        for identity in range(1, number_of_animals+1)
    ]

    # from centroid data
    ################################
    group=FlyHostelGroup.from_list(loaders, protocol="centroids", dist_max_mm=dist_max_mm, min_interaction_duration=min_interaction_duration)
    dt=group.load_centroid_data()
    # assume the thorax is where the centroid is,
    # which is the middle of the frame
    dt["thorax_x"]=SQUARE_WIDTH//2 
    dt["thorax_y"]=SQUARE_HEIGHT//2

    pose=dt[["id", "identity", "frame_number", "thorax_x", "thorax_y"]],
    dt=dt[["id", "identity", "frame_number", "x", "y"]],
    bodyparts=["thorax"]
    ################################

    # from pose data
    ################################
    group=FlyHostelGroup.from_list(loaders, protocol="full", dist_max_mm=dist_max_mm, min_interaction_duration=min_interaction_duration)
    dt=group.load_centroid_data()
    pose=group.load_pose_data("pose_boxcar")
    ################################


    # finally
    interactions, pose_absolute = group.find_interactions(
        dt,
        pose,
        bodyparts=BODYPARTS,
        framerate=FRAMERATE
    )

    if output is not None:
        interactions.to_csv(output)
    return interactions, pose_absolute


def load_animal_pair_data(animals, pose_name, **kwargs):
    experiments=[animal.split("__")[0] for animal in animals]
    identities=[int(animal.split("__")[1]) for animal in animals]

    loaders=[
        FlyHostelLoader(
            experiment=experiment,
            identity=identity,
            chunks=range(0, 400),
            identity_table="IDENTITY_VAL",
            roi_0_table="ROI_0_VAL"
        )
        for experiment, identity in zip(experiments, identities)
    ]

    failed=False
    for fly in loaders:
        animal=fly.experiment + "__" + str(fly.identity).zfill(2)

        pose_file=fly.get_pose_file_h5py(pose_name)
        assert os.path.exists(pose_file)

        try:
            with h5py.File(pose_file, "r") as f:
                keys=f.keys()
                logger.debug(f"Validated %s", pose_file)
        except Exception as error:
            logger.error("Can't read %s", pose_file)
            logger.error(error)
            failed=True
    if failed:
        raise ValueError("Corrupted files")

    group=FlyHostelGroup.from_list(loaders, protocol="centroids", **kwargs)
    return group


def infer_interactions_by_id_pairs(group, dt, pose, framerate=30, bodyparts=BODYPARTS):


    if isinstance(dt, cudf.DataFrame):
        xf=cudf
        ids=dt["id"].to_pandas().unique()
    else:
        xf=pd
        ids=dt["id"].unique()

    pose_absolute=[]
    interactions=[]

    for id1, id2 in itertools.combinations(ids, 2):
        interactions_, pose_absolute_=infer_interactions(
            group,
            dt.loc[dt["id"].isin([id1, id2])],
            pose.loc[pose["id"].isin([id1, id2])],
            bodyparts=bodyparts
        )
        if interactions_ is not None:
            interactions.append(interactions_)
        if pose_absolute_ is not None:
            pose_absolute.append(pose_absolute_)

    interactions=xf.concat(interactions, axis=0)
    pose_absolute=xf.concat(pose_absolute, axis=0)

    return interactions, pose_absolute


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


def infer_interactions_by_time_partitions(
        group, dt,
        pose_name,
        partition_size=None,
        framerate=30,
        bodyparts=BODYPARTS,
        keep_pose=False
    ):

    """
    Call infer_interactions for one temporal partition of the dataset at a time 

    partition_size (int): Number of seconds making each partition
    """
    if isinstance(dt, cudf.DataFrame):
        xf=cudf
        useGPU=True
    else:
        xf=pd
        useGPU=False

    interactions=[]
    pose_absolute=[]

    min_t=dt["t"].min()
    max_t=dt["t"].max()
    interval=(min_t, max_t)
    partition_size=min(partition_size, interval[1]-interval[0])
    t0s=np.arange(interval[0], interval[1], partition_size)
    t1s=t0s+partition_size

    for i, (min_time, max_time) in tqdm(enumerate(zip(t0s, t1s)), desc="Inferring interactions"):

        zt0=round(min_time/3600, 2)
        zt1=round(max_time/3600, 2)

        before=time.time()
        print(f"Partition {i} ({zt0} - {zt1})")
        if len(bodyparts)==1 and bodyparts[0]=="thorax":
            pose_dataset=dt[["id", "frame_number"]]
            pose_dataset["thorax_x"]=SQUARE_WIDTH//2
            pose_dataset["thorax_y"]=SQUARE_HEIGHT//2
        else:
            pose_dataset=group.load_pose_data(
                framerate=framerate,
                pose_name=pose_name,
                min_time=min_time,
                max_time=max_time,
                useGPU=useGPU
            )

        if useGPU:
            assert isinstance(pose_dataset, cudf.DataFrame)

        if pose_dataset.shape[0]==0 and DEBUG:
            logger.warning(
                "No pose data found from zt= %s to zt= %s", zt0, zt1
            )
            continue

        centroid_dataset=dt.loc[
            (dt["t"]>=min_time)&(dt["t"]<max_time)
        ]
        after=time.time()

        logger.debug("%s seconds to filter partition data", after-before)
        interactions_, pose_absolute_=infer_interactions(
            group,
            centroid_dataset,
            pose_dataset,
            framerate=framerate,
            bodyparts=bodyparts
        )
        if interactions_ is not None:
            try:
                interactions_cpu=interactions_.to_pandas()
                del interactions_
            except Exception:
                interactions_cpu=interactions_
            n_rows=interactions_cpu.shape[0]
            interactions.append(pd.DataFrame(interactions_cpu))
        else:
            logger.warning(
                "No interactions detected between %s and %s", zt0, zt1
            )
        if pose_absolute_ is not None and keep_pose:
            pose_absolute.append(xf.DataFrame(pose_absolute_))
        else:
            del pose_absolute_

        logger.info(
            "Collected %s rows of interaction data in partition %s/%s",
            n_rows, i+1, len(t0s)
        )
        del pose_dataset

    # put together all partitions
    if len(interactions)==0:
        interactions=None
    else:
        interactions=pd.concat(interactions, axis=0)

    if len(pose_absolute)==0:
        pose_absolute=None
    else:
        pose_absolute=pd.concat(pose_absolute, axis=0)

    return interactions, pose_absolute

def analyze_group(
        group, pose_name, bodyparts=None, framerate=15,
        useGPU=True, interval=None,
        partition_size=None, n_jobs=1,
        **kwargs
    ):
    """
    Detect interactions between animals in a group
    """
    if bodyparts is None:
        keep_pose=False
    else:
        # TODO
        # should be true but I havent implemented
        # a proper handling of pose data because it is very big
        # and can fill the gpu
        keep_pose=False

    # load the x y coordinates of the centroids of each animal over time
    dt=group.load_centroid_data(framerate=framerate, useGPU=useGPU)

    if useGPU:
        assert isinstance(dt, cudf.DataFrame)

    if interval is None:
        interval=(
            dt["frame_number"].min(),
            dt["frame_number"].max()+1
        )

    # interactions
    interactions, pose=infer_interactions_by_time_partitions(
        group, dt, pose_name=pose_name,
        framerate=framerate,
        partition_size=partition_size,
        keep_pose=keep_pose,
        bodyparts=bodyparts,
        **kwargs
    )

    group.interactions=interactions
    if len(interactions)==0:
        return None

    group.dt=dt
    group.pose=pose
    return group


def infer_interactions(group, dt, pose, framerate, bodyparts=BODYPARTS):
    interactions, pose_absolute = group.find_interactions(
        dt, pose,
        framerate=framerate,
        bodyparts=bodyparts,
        using_bodyparts=False,
    )
    interactions["chunk"]=interactions["frame_number"]//CHUNKSIZE
    interactions["frame_idx"]=interactions["frame_number"]%CHUNKSIZE
    return interactions, pose_absolute

def annotate_interactions(group, time_window_length=10, asleep_annotation_age=10):

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
    dt_sleep["frame_number"]+=FRAMERATE*asleep_annotation_age
    # this is useful so that the sleep state at the time of the interaction can be defined based on the behavior
    # from a little bit before in time

    # annotate sleep state of the interaction
    hits=pd.merge_asof(
        interactions,
        dt_sleep, by="id",
        on="frame_number",
        direction="backward",
        tolerance=FRAMERATE*time_window_length
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


def compute_experiment_interactions(
        group, bodyparts=BODYPARTS,
        pose_name="filter_rle-jump",
        framerate=15,
        partition_size=None,
        n_jobs=1,
        interval=None,
        **kwargs
    ):
    """
    CLI entry point

    """
    analyze_group(
        group,
        framerate=framerate,
        bodyparts=bodyparts,
        partition_size=partition_size,
        pose_name=pose_name,
        n_jobs=n_jobs,
        interval=interval,
        **kwargs
    )
    return group


def initialize_group(experiment, pose_name, number_of_animals=None, identities=None, min_time=None, max_time=None, **all_kwargs):
    if identities is None:
        identities=range(1, number_of_animals+1)

    animals=[f"{experiment}__{str(identity).zfill(2)}" for identity in identities]
    interaction_kwargs={k: all_kwargs[k] for k in ["dist_max_mm", "min_interaction_duration", "min_time_between_interactions"] if k in all_kwargs}
    group=load_animal_pair_data(animals, pose_name=pose_name, min_time=min_time, max_time=max_time, **interaction_kwargs)
    return group
