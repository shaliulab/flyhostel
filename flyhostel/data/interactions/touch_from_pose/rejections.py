import itertools
import os
import numpy as np
import cv2
import pandas as pd
from tqdm.auto import tqdm
pd.set_option("display.max_colwidth", 80)

from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.utils import (
    get_framerate,
    get_chunksize,
)

from flyhostel.data.pose.constants import legs as all_legs
legs=[leg for leg in all_legs if "J" not in leg]
legs

import numpy as np
import movement.kinematics as kin
from flyhostel.data.interactions.touch_from_pose.loaders import (
    load_sleep_data_all,
    load_experiment_features
)

from flyhostel.data.interactions.touch_from_pose.rejection_utils import (
    add_animal_state,
    distance_per_second,
)
from flyhostel.data.pose.ethogram.utils import annotate_bouts, annotate_bout_duration
from .constants import LONG_IMMOBILE_SECONDS
from flyhostel.data.pose.loaders.interactions import CONTACT_THRESHOLD
from flyhostel.utils import (
    annotate_local_identity,
    build_interaction_video_key,
    get_basedir,
    get_dbfile,
    get_local_identities,
    get_single_animal_video,
)
os.environ["DEEPETHOGRAM_PROJECT_PATH"]="/flyhostel_data/fiftyone/FlyBehaviors/DEG/FlyHostel_deepethogram_47fps/"
TOUCH_MASK_DURATION=10

def make_index_for_videos(experiment, interactions, centroids):
    """
    
    interactions (DataFrame): Has columns
      interaction_duration
      animal
      chunk
      frame_number
      id
      nn
    centroids (DataFrame): Has columns
      id
      frame_number
      center_x
      center_y
    """
    
    chunksize=get_chunksize(experiment)
    basedir=get_basedir(experiment)
    fps=get_framerate(experiment)
    
    interactions["framerate"]=fps
    interactions["nframes"]=np.ceil((interactions["interaction_duration"]*interactions["framerate"]))
    interactions["nframes"]=interactions["nframes"].astype(int)
    interactions["identity"]=interactions["animal"].str.slice(start=-2).astype(int)
    interactions=annotate_local_identity(interactions, experiment)
    interactions["first_frame"]=interactions["frame_number"].copy()
    interactions["last_frame_number"]=interactions["frame_number"]+(interactions["interaction_duration"]*fps).astype(int)
    interactions["key"]=[build_interaction_video_key(experiment, row) for _, row in interactions.iterrows()]
    index=pd.DataFrame.from_records(
        list(itertools.chain(*[
            zip(
                [row["first_frame"],]*row["nframes"],
                [row["last_frame_number"],]*row["nframes"],
                [row["framerate"],]*row["nframes"],
                [row["identity"],]*row["nframes"],
                [row["key"],]*row["nframes"],
                [row["id"],]*row["nframes"],
                [row["nn"],]*row["nframes"],
                np.arange(row["frame_number"], row["frame_number"]+row["nframes"]),
            ) for i, row in interactions.iterrows()
        ])),
        columns=["first_frame", "last_frame", "framerate","identity", "key", "id", "nn", "frame_number"]
    )
    index["chunk"]=(index["frame_number"]//chunksize).astype(np.int64)
    index["frame_idx"]=index["frame_number"]%chunksize
    
    index=annotate_local_identity(index, experiment)
    index_by_chunk=index.groupby(["chunk", "identity"]).first().reset_index()
    dbfile=get_dbfile(basedir)
    index_by_chunk["frame_number"]=chunksize*index_by_chunk["chunk"]
    table = get_local_identities(dbfile, frame_numbers=index_by_chunk["frame_number"]).reset_index(drop=True)
    
    index_by_chunk["video"]=[
        get_single_animal_video(basedir, row["frame_number"], table, row["identity"], chunksize)
        for i, row in tqdm(index_by_chunk[["frame_number", "identity"]].iterrows(), total=index_by_chunk.shape[0])
    ]
    
    index=index.merge(index_by_chunk[["chunk", "local_identity", "video"]], on=["chunk", "local_identity"], how="left")
    index=index.merge(centroids, on=["id", "frame_number"], how="left")

    index=index.merge(index[["id", "frame_number", "center_x", "center_y"]].rename({
        "id": "nn",
        "center_x": "center_x_nn",
        "center_y": "center_y_nn",
    }, axis=1), on=["nn", "frame_number"], how="left")
    return index
                        

def compute_leg_features(df):

    data={}
    for side in ["L", "R"]:
        other_side = {"L": "R", "R": "L"}[side]
        this_leg=f"m{side}L"
        other_leg=f"m{other_side}L"
        target = df.query(f"bodypart == @this_leg")["displacement"].sum()
        other_legs = df.query(f"bodypart.isin(['rLL', 'rRL', 'fLL', 'fRL', @other_leg])")["displacement"].sum()
        ratio = target / (target + other_legs)

        side_total=df.query(f"bodypart.isin(['r{side}L', 'f{side}L', 'm{side}L'])")["displacement"].sum()
        data.update({f"{side}_displacement": target, f"{side}_ratio": ratio, f"{side}_total": side_total})
    
    return pd.Series(data)
    

def expand_touch(df, expand):
    arr=(df["touch"].values*255).astype(np.uint8).reshape((1, -1))
    arr=cv2.dilate(arr, np.ones((1, expand)))==255
    df["touch"]=arr.flatten()
    return df


def annotate_based_on_fn_in_column(df, column, variable):
    
    # 1. Build a lookup Series: frame_number -> metric
    metric_at_frame = df.set_index("frame_number")[variable]

    # 2. Use last_event to look up the metric value at that frame
    df[variable] = df[column].map(metric_at_frame)
    return df


def select_positives(df, database, groupby=["id", "nn", "interaction"], min_ratio=0.5, min_displacement=50):

    df=df.merge(
            database.query("touch_bool==True").groupby(groupby).apply(
                lambda x: x.query(f"L_ratio > {min_ratio} & L_displacement > {min_displacement}").shape[0]
            ).reset_index(name="L_duration"),
            on=groupby
        ).merge(
            database.query("touch_bool==True").groupby(groupby).apply(
                lambda x: x.query(f"R_ratio > {min_ratio} & R_displacement > {min_displacement}").shape[0]
            ).reset_index(name="R_duration"),
            on=groupby
        )
    return df


def detect_putative_rejections(experiment, number_of_animals, touch_mask_duration=TOUCH_MASK_DURATION):

    fps=get_framerate(experiment)
    chunksize=get_chunksize(experiment)
    loaders=[]
    centroids_1s=[]
    for identity in range(1, number_of_animals+1):
        loader=FlyHostelLoader(experiment, identity)
        loader.load_centroid_data()

        loader.dt["t_round"]=1*(loader.dt["t"]//1).astype(int)
        dt_1s=loader.dt.groupby("t_round").first().reset_index()
        dt_1s["distance"]=np.sqrt((dt_1s[["center_x", "center_y"]].diff()**2).sum(axis=1))
        dt_1s["distance"]/=loader.pixels_per_mm # mm
        centroids_1s.append(dt_1s)
        loaders.append(loader)
    pixels_per_mm=loaders[0].pixels_per_mm
    centroids_1s=pd.concat(centroids_1s, axis=0).reset_index(drop=True)

    touch_database=pd.read_feather("touch_database.feather")
    touch_database["frame_number"]=np.array(touch_database["frame_number"].values, np.int64)

    touch_database["t_round"]=(1*(touch_database["t"]//1)).astype(int)
    touch_database.rename({"app_dist_best": "distance"}, axis=1, inplace=True)
    touch_database["distance"]/=pixels_per_mm


    sleep_df=load_sleep_data_all(loaders, number_of_animals, bout_annotation="asleep")
    inactive_df=load_sleep_data_all(loaders, number_of_animals, bout_annotation="inactive")
    
    touch_database=pd.merge_asof(
        touch_database.sort_values("frame_number"),
        sleep_df[[
            "id", "animal", "frame_number",
            "asleep", "inactive",
            "bout_in", "bout_out",
        ]]\
            .rename({
                "bout_in": "bout_in_asleep",
                "bout_out": "bout_out_asleep",
            }, axis=1)\
            .sort_values("frame_number"),
        on="frame_number", by="id", direction="backward", tolerance=np.int64(np.ceil(fps*1.1))
    )


    touch_database["inactive_"]=touch_database["inactive"].copy()
    touch_database=touch_database.groupby(["id", "animal"]).apply(lambda df: annotate_based_on_fn_in_column(df, "last_isolated", "inactive")).reset_index(drop=True)
    touch_database=touch_database.groupby(["id", "animal"]).apply(lambda df: annotate_bout_duration(annotate_bouts(df, "inactive"), fps)).reset_index(drop=True)

    touch_database_1s=touch_database.groupby(["t_round", "id", "animal", "nn"]).agg({
        "touch_raw": np.sum,
        "touch": np.sum,
        "asleep": np.all,
        "inactive": np.all,
        "distance": np.min,
        "last_isolated": np.min,
        "bout_in_asleep": np.min,
        "bout_out_asleep": np.min,
    }).reset_index().rename({"t_round": "t"}, axis=1)
    touch_database_1s["touch_bool"]=touch_database_1s["touch"]>0

    touch_database_1s=touch_database_1s.groupby(["id", "nn", "animal"]).apply(lambda df: annotate_bout_duration(annotate_bouts(df, "touch_bool"), 1))\
        .reset_index(drop=True)\
        .rename({
            "bout_in": "bout_in_touch",
            "bout_out": "bout_out_touch",
        }, axis=1)
    touch_database_1s=touch_database_1s.groupby(["id", "animal"]).apply(lambda df: annotate_bout_duration(annotate_bouts(df, "inactive"), 1)).reset_index(drop=True)
    touch_database_1s["pre_longImmobile"]=(touch_database_1s["inactive"])&(touch_database_1s["bout_in"]>=LONG_IMMOBILE_SECONDS)
    touch_database_1s["post_longImmobile"]=(touch_database_1s["inactive"])&(touch_database_1s["bout_out"]>=LONG_IMMOBILE_SECONDS)

    features=load_experiment_features(experiment)

    position=features.sel(keypoints=legs).position.copy()
    del position["frame_number"]

    # compute distance travelled by every bodypart, per second
    # this metric tells me whether the animal is "reacting" to the
    # intruder or not
    forward_displacement = kin.compute_forward_displacement(position)
    dist_per_sec=distance_per_second(forward_displacement)

    # annotate sleep and immobility state
    state_columns=[
        "touch", "distance", "asleep", "inactive",
        "pre_longImmobile", "post_longImmobile",
        "bout_in", "bout_out",
        "bout_in_asleep", "bout_out_asleep",
        "bout_in_touch","bout_out_touch",
        "id", "nn"
    ]
    for state in state_columns:
        dist_per_sec=add_animal_state(dist_per_sec, touch_database_1s, state_col=state)
    
    records=[]
    for individual_i in range(number_of_animals):
        individual=dist_per_sec.individuals[individual_i].values.item()
        # distance between any two bodyparts of the pair of flies needs to be < CONTACT_THRESHOLD mm
        mask=dist_per_sec.sel(individuals=individual)["distance"] < CONTACT_THRESHOLD
        hits=np.where(mask)[0]
        for frame_i in tqdm(hits):
            t=mask.sec[frame_i].values.item()
            idx=np.where(features.time.values>=t)[0][0]
            frame_number=features.frame_number.data[idx].item()
            dss=dist_per_sec.sel(individuals=individual, sec=t)
            for bodypart in dist_per_sec.keypoints.data:
                displacement=dss.sel(keypoints=bodypart).values.item()
                states=[]
                for state in state_columns:
                    states.append(getattr(dss, state).values.item())
                records.append((t, frame_number, bodypart, individual, displacement, *states))
    
    result=pd.DataFrame.from_records(records, columns=["t", "frame_number", "bodypart", "animal", "displacement"]+state_columns)
    result["chunk"]=result["frame_number"]//chunksize
    result["frame_idx"]=result["frame_number"]%chunksize
    result.to_feather("pose_database.feather")

    # analyze rejections
    rejection_database=result.groupby(["id", "animal", "nn", "frame_number", "t"]).apply(compute_leg_features).reset_index()
    rejection_database["total_displacement"]=rejection_database["L_total"]+rejection_database["R_total"]

    rejection_database=rejection_database.merge(
        result.groupby(["id", "nn", "frame_number"]).first().reset_index()[[
            "id", "nn", "frame_number", "chunk", "frame_idx", "touch",
            "distance", "asleep", "inactive",
            "bout_in", "bout_out",
            "bout_in_touch", "bout_out_touch",
            "pre_longImmobile", "post_longImmobile"
        ]],
        on=["id", "nn", "frame_number"]
    )

    # join touch bouts separated by < mask_duration seconds
    rejection_database["touch_bool"]=rejection_database["touch"]>0
    rejection_database["touch_mask"]=rejection_database["touch_bool"].astype(int).copy()
    rejection_database.loc[
        (rejection_database["touch_mask"]==0) & ((rejection_database["bout_in_touch"] + rejection_database["bout_out_touch"]) < touch_mask_duration),
        "touch_mask"
    ]=1

    # partition touch into interactions
    rejection_database=rejection_database.merge(
        rejection_database.groupby(["id", "nn"]).apply(lambda x:
            pd.DataFrame({"frame_number": x["frame_number"], "interaction": (x["touch_mask"].diff()==1).cumsum()})
        )\
            .reset_index(),
            # .drop("level_1", axis=1, errors="ignore"),
        on=["id", "nn", "frame_number"],
        how="left"
    )

    for loader in loaders:
        loader.dt["animal"]=loader.datasetnames[0]
    
    index_frame_numbers=pd.concat([loader.dt[["animal", "t", "frame_number"]] for loader in loaders], axis=0).reset_index(drop=True)
    index_frame_numbers["t"]=(1*(index_frame_numbers["t"]//1)).astype(np.int64)
    index=result[["animal", "t"]].drop_duplicates()
    index=index.merge(index_frame_numbers, on=["t", "animal"], how="outer")
    index=index.groupby("t").first().reset_index()[["t", "frame_number"]]
    index.to_csv("frame_index.csv")

    groupby=["id", "nn", "interaction"]


    rejection_database["time_inactive_300s_before"]=np.nan
    rejection_database["time_inactive_300s_after"]=np.nan
    
    rejection_database["asleep_before"]=np.nan
    rejection_database["bout_in_asleep_before"]=np.nan
    rejection_database["asleep_after"]=np.nan
    rejection_database["bout_out_asleep_after"]=np.nan

    rejection_database["inactive_before"]=np.nan
    rejection_database["bout_in_inactive_before"]=np.nan
    rejection_database["inactive_after"]=np.nan
    rejection_database["bout_out_inactive_after"]=np.nan
    
    durations=rejection_database.query("touch_mask==True")\
        .groupby(groupby)\
        .agg({"touch_mask": np.sum})\
        .reset_index()\
        .rename({"touch_mask": "interaction_duration"}, axis=1)

    rejection_database=rejection_database.merge(
        durations,
        on=groupby,
        how="left"
    )

    for i, row in tqdm(
        rejection_database.iterrows(),
        total=rejection_database.shape[0],
        desc="Annotating time inactive before and after interactions"
    ):
        try:
            animal=row["animal"]
            
            min_t=sleep_df.query("(animal == @animal)")["t"].min()
            max_t=sleep_df.query("(animal == @animal)")["t"].max()

            t=row["t"]
            tm1=t-1
            t_before=t-300
            t_end=row["t"]+row["interaction_duration"]
            t_endp1=t_end+1
            t_after=t_end+300

            if t_before < min_t or t_after > max_t or np.isnan(row["interaction_duration"]):
                continue
            inactive_before=sleep_df.query("(animal == @animal) & (t >= @t_before) & (t < @t)")["inactive"].sum()
            inactive_after=sleep_df.query("(animal == @animal) & (t >= @t_end) & (t < @t_after)")["inactive"].sum()
            rejection_database["time_inactive_300s_before"].iloc[i]=inactive_before
            rejection_database["time_inactive_300s_after"].iloc[i]=inactive_after
            asleep, bout_in=sleep_df.query("(animal == @animal) & t == @tm1")[["asleep", "bout_in"]].iloc[0]
            rejection_database["asleep_before"].iloc[i]=asleep
            rejection_database["bout_in_asleep_before"].iloc[i]=bout_in
            asleep, bout_out=sleep_df.query("(animal == @animal) & t == @t_endp1")[["asleep", "bout_out"]].iloc[0]
            rejection_database["asleep_after"].iloc[i]=asleep
            rejection_database["bout_out_asleep_after"].iloc[i]=bout_out

            inactive, bout_in=inactive_df.query("(animal == @animal) & t == @tm1")[["inactive", "bout_in"]].iloc[0]
            rejection_database["inactive_before"].iloc[i]=inactive
            rejection_database["bout_in_inactive_before"].iloc[i]=bout_in
            inactive, bout_out=inactive_df.query("(animal == @animal) & t == @t_endp1")[["inactive", "bout_out"]].iloc[0]
            rejection_database["inactive_after"].iloc[i]=inactive
            rejection_database["bout_out_inactive_after"].iloc[i]=bout_out
        except Exception as error:
            print(error)
            import ipdb; ipdb.set_trace()

    rejection_database.to_feather("rejection_database.feather")


    index=rejection_database.query("touch_mask==True").groupby(groupby).first().reset_index()
    
    # [[
    #     "id", "nn", "animal", "frame_number", "chunk", "frame_idx", "t", "interaction", "inactive", "bout_in", "bout_out"
    # ]]
    index=select_positives(index, rejection_database, groupby, min_ratio=0.5, min_displacement=50)

    # annotate distance travelled
    
    index=index\
        .merge(
            rejection_database.query("touch_mask==True").groupby(groupby).agg({"total_displacement": np.sum}).reset_index(),
            on=groupby
        ).merge(
            rejection_database.query("touch_mask==True").groupby(groupby).agg({"touch_bool": np.sum}).reset_index().rename({"touch_bool": "touch_duration"}, axis=1),
            on=groupby
        )

    index["distance_traveled_mm"]=np.nan
    for i, row in tqdm(index.iterrows()):
        t_start=row["t"]
        t_end=row["t"]+row["interaction_duration"]
        id=row["id"]
        index["distance_traveled_mm"].loc[i]=centroids_1s.query("t >= @t_start & t < @t_end & id == @id")["distance"].sum().item()

    assert not index["distance_traveled_mm"].isna().any()

    dt_centroids=pd.concat([
        loader.dt[["id", "frame_number", "center_x", "center_y", "distance"]].copy()
        for loader in loaders
    ], axis=0).reset_index(drop=True)
    video_index=make_index_for_videos(experiment, index, dt_centroids)
    video_index.to_csv("video_index.csv")

    index=index.merge(video_index[["id", "frame_number", "nn", "key"]], on=["id", "frame_number", "nn"], how="left")
    index.to_csv("interactions_v2.csv")

    return result, index