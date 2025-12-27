import logging

import numpy as np

from flyhostel.data.pose.ethogram.utils import (
    annotate_bout_info,
    generate_windows,
)

logger=logging.getLogger(__name__)

def apply_bg_limit(df, replace, idx):
    rows=(df["prediction"].isin(["background"])) & (df["duration_pred"]<5)
    df.loc[rows, "rule"]=idx
    df.loc[rows, "prediction2"]=replace
    return df
    
    
def apply_groom_limit(df, replace, idx):
    """
    does not modify prediction
    """
    rows=(df["prediction"].isin(["groom"])) & (df["duration_pred"]<5)
    df.loc[rows, "rule"]=idx
    df.loc[rows, "prediction2"]=replace
    return df

def apply_inactive_micromovement_limit(df, micromovement_behavior, idx):
    """
    does not modify prediction
    """
    rows=(df["prediction2"].isin([micromovement_behavior])) & (df["duration_pred"]>5)
    df.loc[rows, "rule"]=idx
    df.loc[rows, "prediction2"]=df.loc[rows, "prediction"]
    return df


def apply_inactive_pe_limit(df, replace, idx):
    """
    does not modify prediction
    """
    rows=(df["prediction"].isin(["inactive+pe"])) & (df["duration_pred"]>5)
    df.loc[rows, "rule"]=idx
    df.loc[rows, "prediction2"]=replace
    return df


def apply_inactive2feed(df, interval, replace, idx):
    """
    does not modify prediction
    """
    rows=(df["prediction"].isin(["inactive"])) & (df["rule"]==0) & (df["proboscis"]>=0.6) & (df["food_distance"] > interval[0])&(df["food_distance"] < interval[1])
    df.loc[rows, "rule"]=idx
    df.loc[rows, "prediction2"]=replace
    return df

def apply_proboscis_requirement(df, replace):
    """
    does modify prediction
    """
    rows=(df["prediction"]=="inactive+pe")&((df["head_proboscis_distance"]==0)|(df["proboscis"]==0))
    # df.loc[rows, "rule"]=idx
    df.loc[rows, "prediction"]=replace
    rows=(df["prediction"]=="feed")&((df["head_proboscis_distance"]==0)|(df["proboscis"]==0))
    # df.loc[rows, "rule"]=idx
    df.loc[rows, "prediction"]=replace
    return df


def apply_food_distance_rule(df, idx, interval):
    rows=((df["food_distance"] < interval[0])|(df["food_distance"] > interval[1])) & (df["prediction2"]=="feed")
    df.loc[rows, "prediction2"]="inactive+pe"
    df.loc[rows, "rule"]=idx
    rows=(df["food_distance"] > interval[0])&(df["food_distance"] < interval[1]) & (df["prediction2"]=="inactive+pe")
    df.loc[rows, "prediction2"]="feed"
    df.loc[rows, "rule"]=idx
    return df


    

def apply_p2inactive(df, replace, idx):
    rows=np.bitwise_or(
        (df["prediction"].isin(["feed"])) & (df["proboscis"]<0.5),
        (df["prediction"].isin(["inactive+pe"])) & (df["proboscis"]<0.5),
    )
    df.loc[rows, "rule"]=idx
    df.loc[rows, "prediction2"]=replace
    return df


def apply_sequence_rules(df, micromovement_behavior):
    df["rule"]=0
    df=apply_groom_limit(df, micromovement_behavior, 1)
    df=apply_bg_limit(df, micromovement_behavior, 1.5)
    df=apply_inactive_pe_limit(df, "feed", 3)
    df=apply_inactive2feed(df, (-0.03, 0.05), "feed", 4)
    df=apply_food_distance_rule(df, 6, interval=(-0.03, 0.05))
    
    return df


def main(predictions, fps, micromovement_behavior="inactive+micromovement"):
    logger.debug("apply_inactive_pe_requirement")
    predictions=apply_proboscis_requirement(predictions, "inactive")
    logger.debug("annotate_bout_info")
    predictions=predictions.groupby("id").apply(lambda df: annotate_bout_info(df, fps=fps, prediction="prediction")).reset_index(drop=True)
    predictions["prediction2"]=predictions["prediction"].copy()
    logger.debug("sequence rules")
    predictions=apply_sequence_rules(predictions, micromovement_behavior)
    logger.debug("annotate_bout_info")
    predictions=predictions.groupby("id").apply(lambda df: annotate_bout_info(df, fps=fps, prediction="prediction2")).reset_index(drop=True)
    logger.debug("apply_inactive_micromovement_limit")
    predictions=apply_inactive_pe_limit(predictions, "feed", 3)
    predictions=apply_inactive_micromovement_limit(predictions, micromovement_behavior, idx=1.6)
    return predictions