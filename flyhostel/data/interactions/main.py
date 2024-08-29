import h5py
import itertools
import logging
import os.path  
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

        pose_file=os.path.join(fly.basedir, "motionmapper", str(fly.identity).zfill(2), f"pose_{pose_name}", animal, animal + ".h5")
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
    # for key, fly in group.flies.items():
    #     fly.load_behavior_data(experiment=fly.experiment, identity=fly.identity)
    return group


def infer_interactions_by_id_pairs(group, dt, pose, framerate=30, xf=cudf):
    raise NotImplementedError()
    try:
        ids=dt["id"].to_pandas().unique()
    except:
        ids=dt["id"].unique()

    interactions_full=[]
    interactions=[]

    for id1, id2 in itertools.combinations(ids, 2):
        print(id1, id2)
        interactions_full_d, interactions_d=infer_interactions(
            group,
            dt.loc[dt["id"].isin([id1, id2])],
            pose.loc[pose["id"].isin([id1, id2])],
            framerate=framerate,
        )
        if interactions_full_d is not None:
            interactions_full.append(interactions_full_d)
        if interactions_d is not None:
            interactions.append(interactions_d)

    interactions_full=xf.concat(interactions_full, axis=0)
    interactions=xf.concat(interactions, axis=0)
    return interactions_full, interactions

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
    paired_df=df.rename({"id": "nn", "nn": "id"}, axis=1)
    if all(c in paired_df.columns for c in ["id_bodypart", "nn_bodypart"]):
        paired_df.rename({
            "id_bodypart": "nn_bodypart",
            "nn_bodypart": "id_bodypart",
        }, axis=1, inplace=True)


    df=pd.concat([
        df, paired_df
    ], axis=0).sort_values("frame_number")
    return df


def infer_interactions_by_time_partitions(group, dt, pose, partition_size=None, framerate=30, xf=cudf, keep_pose=False):

    interactions_full=[]
    interactions=[]
    pose_absolute=[]

    min_fn=pose["frame_number"].min()
    max_fn=pose["frame_number"].max()

    if partition_size is None:
        intervals=[(min_fn, max_fn+1)]
    else:
        fn0s=np.arange(min_fn, max_fn+partition_size, partition_size)
        fn1s=fn0s+partition_size
        intervals=zip(fn0s, fn1s)

    for fn0, fn1 in tqdm(intervals, desc="Infering interactions"):
        pose_dataset=pose.loc[(pose["frame_number"]>=fn0)&(pose["frame_number"]<fn1)]
        if pose_dataset.shape[0]==0 and DEBUG:
            logger.warning("No pose data found from %s to %s", fn0, fn1)
            continue

        interactions_full_d, interactions_d, pose_absolute_d=infer_interactions(
            group,
            dt.loc[(dt["frame_number"]>=fn0)&(dt["frame_number"]<fn1)],
            pose_dataset,
            framerate=framerate,
        )

        if interactions_full_d is not None:
            interactions_full.append(xf.DataFrame(interactions_full_d))
        else:
            logger.warning("No interactions detected between %s and %s", fn0, fn1)
        if interactions_d is not None:
            interactions.append(xf.DataFrame(interactions_d))
        if pose_absolute_d is not None:
            pose_absolute.append(xf.DataFrame(pose_absolute_d))

    if len(interactions_full)==0:
        interactions_full=None
    else:
        interactions_full=xf.concat(interactions_full, axis=0)

    if len(interactions)==0:
        interactions=None
    else:
        interactions=xf.concat(interactions, axis=0)

    if keep_pose:
        if len(pose_absolute)==0:
            pose_absolute=None
        else:
            pose_absolute=xf.concat(pose_absolute, axis=0)
    else:
        del pose_absolute
        pose_absolute=None

    return interactions_full, interactions, pose_absolute


def analyze_group(group, pose_name, bodyparts, framerate=30, useGPU=True, interval=None, partition_size=None, n_jobs=1, **kwargs):
    if useGPU:
        xf=cudf
        nx=cp
    else:
        xf=pd
        nx=np

    dt=group.load_centroid_data(framerate=framerate, useGPU=useGPU)
    behavior_data=group.load_behavior_data(framerate=framerate, useGPU=useGPU)


    all_interactions=[]
    all_interactions_full=[]

    if interval is None:
        interval=(
            dt["frame_number"].min(),
            dt["frame_number"].max()+1
        )
    
    if partition_size is None:
        partitions=[None]
        fn0s=[interval[0]]
        fn1s=[interval[-1]]
        
    
    else:
        partition_size=min(partition_size, interval[1]-interval[0])
        fn0s=np.arange(interval[0], interval[1], partition_size)
        fn1s=fn0s+partition_size
        partitions=[slice(fn0, fn1) for fn0, fn1 in zip(fn0s, fn1s)]

    for i, partition in enumerate(partitions):
        print(f"Partition {i}: {fn0s[i]}-{fn1s[i]}")
        pose=group.load_pose_data(
            framerate=framerate,
            # bodyparts=bodyparts,
            pose_name=pose_name, partition=partition, useGPU=useGPU
        )
        

        if partition_size is None:
            interactions_full, interactions, pose_absolute=infer_interactions_by_id_pairs(group, dt, pose, framerate=framerate, xf=xf)
        else:
            interactions_full, interactions, pose_absolute=infer_interactions_by_time_partitions(group, dt, pose, framerate=framerate, xf=xf, keep_pose=True)
        
        if interactions_full is None:
            n_rows=0
        else:
            n_rows=interactions_full.shape[0]
            # interactions_full=xf.concat([
            #     interactions_full, interactions_full.rename({"id": "nn", "nn": "id"}, axis=1)
            # ], axis=0).sort_values("frame_number")

            try:
                pose_cpu=pose_absolute.to_pandas()
            except:
                pose_cpu=pose_absolute
            
            # assert touch
            if isinstance(interactions_full, cudf.DataFrame):
                interactions_full_cpu=interactions_full.to_pandas()
            else:
                interactions_full_cpu=interactions_full
    
            # we interpolate time because otherwise it takes too long
            # it does not matter because the frame numbers are still accurate

            # index=dt.loc[dt["frame_number"]%FRAMERATE==0].to_pandas()
            # interactions_full_cpu=interactions_full_cpu.merge(index[["id", "frame_number", "t"]], on=["id", "frame_number"], how="left")
            # interactions_full_cpu["t"].interpolate(method="linear", limit_direction="both", inplace=True)
            # use all t

            touch_interactions=None
            # touch_interactions=infer_touch(pose_cpu, mask=interactions_full_cpu[["id", "nn", "frame_number", "distance_bodypart_mm"]], bodyparts=bodyparts, n_jobs=n_jobs)
            
            if touch_interactions is not None and touch_interactions.shape[0]>0:
                logger.debug("%s touch events detected", touch_interactions.shape[0])
                interactions_full_cpu=interactions_full_cpu.merge(
                    touch_interactions,
                    on=["id", "nn", "frame_number"],
                    # NOTE
                    # how left is needed because touch_interactions is "flat data"
                    # (represents the same pair wise interaction twice per frame)
                    # but interactions_full_cpu is not flat
                    # (each interaction is represented once per frame)
                    # outer would duplicate the dataset and set cells to NaN
                    # in the rows of the other interaction pair
                    how="left"
                ).sort_values(["id", "frame_number"])

            else:
                interactions_full_cpu["edge_distance"]=np.inf

        logger.debug("Collected %s rows of interaction data in partition %s/%s", n_rows, i+1, len(partitions))
        interactions_full_cpu=flatten_data(interactions_full_cpu)

        if interactions is not None:
            all_interactions.append(interactions)
        if interactions_full_cpu is not None:
            all_interactions_full.append(xf.DataFrame(interactions_full_cpu))
    
    if len(all_interactions_full)==0:
        group.dt=None
        group.dt_sleep=None
        group.interactions_full=None
        group.interactions=None
        group.behavior=None
        
        return group
        
    interactions_full=xf.concat(all_interactions_full, axis=0)
    interactions=xf.concat(all_interactions, axis=0)
    

    # try:
    #     ids=dt["id"].to_pandas().unique()
    # except:
    #     ids=dt["id"].unique()

    # interactions_full=[]
    # interactions=[]

    # for id1, id2 in itertools.combinations(ids, 2):
    #     print(id1, id2)
    #     interactions_full_d, interactions_d=infer_interactions(
    #         group,
    #         dt.loc[dt["id"].isin([id1, id2])],
    #         pose.loc[pose["id"].isin([id1, id2])],
    #         framerate=framerate,
    #     )
    #     if interactions_full_d is not None:
    #         interactions_full.append(interactions_full_d)
    #     if interactions_d is not None:
    #         interactions.append(interactions_d)


    dt_sleep=annotate_sleep(dt, **kwargs)
    dt_sleep["frame_number"]=dt_sleep["frame_number"].astype(np.int32)
    dt_sleep.sort_values(["frame_number", "id"], inplace=True)
    
    group.dt=dt
    group.pose=pose
    group.dt_sleep=dt_sleep
    group.interactions=interactions
    group.interactions_full=interactions_full
    group.behavior=behavior_data
    
    return group


def annotate_sleep(dt, FUN=flyhostel_sleep_annotation, time_window_length=10, min_time_immobile=300, velocity_correction_coef=0.0015, mask=None, **kwargs):
    try:
        dt_cpu=dt.to_pandas()
    except:
        dt_cpu=dt
    

    dt_sleep=[]
    for id, df in dt_cpu.groupby("id"):
        
        if mask is None:
            animal_mask=None
        else:
            animal_mask=mask.loc[mask["id"]==id].drop("id", axis=1)
        x=FUN(
            df, min_time_immobile=min_time_immobile,
            time_window_length=time_window_length,
            velocity_correction_coef=velocity_correction_coef,
            mask=animal_mask,
            **kwargs
        )
        x["id"]=id

        dt_sleep.append(
            x
        )
    dt_sleep=pd.concat(dt_sleep, axis=0)
    index=dt_cpu[["t", "frame_number"]]
    index["t"]=time_window_length*(index["t"]//time_window_length)
    index=index.groupby("t").first().reset_index()
    if "frame_number" not in dt_sleep.columns:
        dt_sleep=dt_sleep.merge(index, on="t")
    return dt_sleep


def infer_interactions(group, dt, pose, bodyparts=BODYPARTS, framerate=30):
    interactions_full, pose_absolute = group.find_interactions(
        dt, pose,
        bodyparts=bodyparts,
        framerate=framerate,
        useGPU=True
    )
    if interactions_full is None or interactions_full.shape[0]==0:
        return None, None
    interactions_full["chunk"]=interactions_full["frame_number"]//CHUNKSIZE
    interactions_full["frame_idx"]=interactions_full["frame_number"]%CHUNKSIZE
    
    # interactions_full has one row per pairwise interaction and frame
    pairwise_interactions=group.interactions_by_closest_point(interactions_full)
    
    # pairwise_interactions has one row per pairwise interaction t the closest point
    interactions=group.flatten_interactions(pairwise_interactions)
    
    # interactions has two rows per pairwise interaction (one for each member)
    interactions["animal"]=[group.animals[group.ids.index(id)] for id in interactions["id"]]
    return interactions_full, interactions, pose_absolute

def annotate_interactions(group, time_window_length=10, asleep_annotation_age=10):

    legs=[bp for bp in BODYPARTS if "L" in bp]
    core=["head", "thorax", "abdomen"]

    try:
        interactions=group.interactions.to_pandas()
    except:
        interactions=group.interactions.copy()
    
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
    hits=pd.merge_asof(interactions, dt_sleep, by="id", on="frame_number", direction="backward", tolerance=FRAMERATE*time_window_length)

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


def compute_experiment_interactions(group, bodyparts, pose_name="filter_rle-jump", framerate=15, partition_size=None, n_jobs=1, interval=None, **all_kwargs):

    print("Sleep kwargs")
    sleep_kwargs={k: all_kwargs[k] for k in ["min_time_immobile", "time_window_length", "velocity_correction_coef"] if k in all_kwargs}
    for k, v in sleep_kwargs.items():
        print(f"{k}: {v}")

    group=analyze_group(
        group,
        framerate=framerate,
        bodyparts=bodyparts,
        partition_size=partition_size,
        pose_name=pose_name,
        n_jobs=n_jobs,
        interval=interval,
        **sleep_kwargs
        )
    
    annotation_kwargs={k: all_kwargs[k] for k in ["asleep_annotation_age", "time_window_length"] if k in all_kwargs}
    if group.interactions is not None:
        hits=annotate_interactions(group, **annotation_kwargs)
    else:
        hits=None
    return group, hits


def initialize_group(experiment, pose_name, number_of_animals=None, identities=None, **all_kwargs):
    if identities is None:
        identities=range(1, number_of_animals+1)

    animals=[f"{experiment}__{str(identity).zfill(2)}" for identity in identities]
    print("Interaction kwargs")
    interaction_kwargs={k: all_kwargs[k] for k in ["dist_max_mm", "min_interaction_duration", "min_time_between_interactions"] if k in all_kwargs}
    for k, v in interaction_kwargs.items():
        print(f"{k}: {v}")

    group=load_animal_pair_data(animals, pose_name=pose_name, **interaction_kwargs)
    return group
