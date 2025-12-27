import numpy as np

from .inference import infer
from .loaders import load_features
from .utils import load_params
from flyhostel.utils import get_number_of_animals


def annotate_last_isolated(df):
    df["last_isolated"]=df["frame_number"].copy()
    df.loc[df["touch"], "last_isolated"]=np.nan
    df["last_isolated"]=df["last_isolated"].ffill().bfill()
    return df



def detect_touch(experiment, number_of_animals, model):

    assert number_of_animals == get_number_of_animals(experiment)
    
    columns=[
        "frame_number",
        "t",
        "touch_raw",
        "touch",
        "id",
        "nn",
        "metric_cross",
        "app_dist_best",
        "last_isolated",
    ]
    features=load_features([experiment])
    trainable_params=load_params(model)

    df_machine=infer(features, **trainable_params)
    df_machine[columns].to_feather("touch_database.feather")
    return df_machine
