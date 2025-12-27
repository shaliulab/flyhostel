import os
import glob
import numpy as np
import pandas as pd
pd.set_option("display.max_colwidth", 80)
PROJECT_PATH="/flyhostel_data/fiftyone/FlyBehaviors/DEG-REJECTIONS/rejections_deepethogram/"

from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.utils import (
    get_number_of_animals,
)
from flyhostel.data.interactions.touch_from_pose.loaders import (
    load_sleep_data,
)


from flyhostel.data.pose.constants import legs
legs=[leg for leg in legs if "J" not in leg]
import numpy as np
from tqdm.auto import tqdm
legs

def has_labels(key):
    return len(glob.glob(f"{PROJECT_PATH}/DATA/{key}/{key}_labels.csv"))==1

def analysis(experiment):

    number_of_animals=get_number_of_animals(experiment)
    identities=range(1, number_of_animals+1)
    interactions=[]
    sleep=[]
    interactions_database=pd.read_csv("interactions_v2.csv", index_col=0)

    for identity in identities:

        loader=FlyHostelLoader(experiment, identity)
        loader.interactions_index=interactions_database\
            .query("id in @loader.ids")
        loader.interactions_index["experiment"]=loader.experiment
        loader.interactions_index["has_labels"]=[
            has_labels(key)
            for key in loader.interactions_index["key"]
        ]
        loader.load_centroid_data(cache="/flyhostel_data/cache")
        loader.sleep=load_sleep_data(loader, bout_annotation="asleep")
        sleep.append(loader.sleep)
        interactions.append(loader.interactions_index)
    sleep=pd.concat(sleep, axis=0)

    last_interaction=sleep.query("asleep==False").merge(
    interactions_database[["nn", "t", "interaction", "touch_bool", "id"]].rename({"nn": "id", "id": "nn"}, axis=1),
        on=["id", "t"], how="inner")\
        .groupby(["id", "bout_count"]).last().reset_index()[["id", "nn", "interaction"]]
    last_interaction["last_interaction"]=True

    interactions=pd.concat(interactions, axis=0).reset_index(drop=True)
    interactions=interactions.merge(last_interaction.rename({"nn": "id", "id": "nn"}, axis=1), on=["id", "nn", "interaction"], how="left")
    interactions.loc[interactions["last_interaction"].isna(), "last_interaction"]=False
    latency_dataset=annotate_latency(interactions)
    latency_dataset.to_csv("latency_dataset.csv")
    fly_may_be_falling_asleep=interactions.query("bout_in_inactive_before>=240 & inactive_before==True")
    fly_may_be_falling_asleep.to_csv("fly_may_be_falling_asleep.csv")

def annotate_latency(interactions):
    interactions["sleep_latency"]=interactions["bout_out_asleep_after"].copy()/60
    latency_dataset=interactions\
        .merge(
            interactions[["nn", "sleep_latency", "bout_in_asleep_before", "asleep_before", "id", "frame_number"]]\
                .rename({
                    "id": "nn", "nn": "id",
                    "sleep_latency": "sleep_latency_nn",
                    "asleep_before": "asleep_before_nn",
                    "bout_in_asleep_before": "bout_in_asleep_before_nn",                    
                }, axis=1),
            on=["id", "nn", "frame_number"]
        )
    return latency_dataset