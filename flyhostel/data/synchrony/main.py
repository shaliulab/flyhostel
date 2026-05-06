import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from flyhostel.utils.utils import sort_ids
from .correlation import (
    annotator,
    cross_correlationv2,
    agreement,
    psi,
    group_psi_unnorm,
    group_psi,
)

# coupling parameters
COUPLING_FUNS={
    cross_correlationv2: "cross_correlationv2",
    agreement: "agreement",
    psi: "psi"
}

GROUP_FUNS={
    group_psi: "group_psi",
    group_psi_unnorm: "group_psi_unnorm",
}


DEFAULT_LAGS=[-50, -40, -30, -20,-10, 0, 10, 20, 30, 40, 50]


def compute_synchrony(dt_sleep, lags, interval=None, feature="asleep", FUN=annotator, FUN_metric=cross_correlationv2):
    """
    Compute synchrony in the behavior of a group of flies using the pearson correlation
    The average pairwise correlation between every pair of flies in the dataset is used as proxy of synchrony


    Args:
        dt_sleep (pd.DataFrame): Contains columns id, t, and whatever feature is provided in the args
        lags (list): Contains a series of windows counts which one of the timeseries will be shifted to compared to the other one in the pair
            The obtained correlation is an estimate of the background synchrony
            This is not the same as the number of seconds shifted. The number of seconds is the window count * time_window_length
        interval (tuple): If passed, two integers that refer to the ZT values used to filter the data
    
    Returns:
        all_corrs_df (pd.DataFrame): Contains lag, corr
    """
    all_corrs={"corr": [], "lag": []}
    raw_corr={"corr": [], "lag": [], "id1": [], "id2":[]}


    if interval is None:
        df=dt_sleep
    else:
        t0, t1=interval
        df=dt_sleep.loc[(dt_sleep["t"] >= t0) & ((dt_sleep["t"] < t1))]

    corrs, pairs=FUN(df, lags=lags, feature=feature, FUN=FUN_metric)
    for lag in tqdm(lags, desc="Computing corr at lag"):
        all_corrs["lag"].append(lag)
        all_corrs["corr"].append(np.mean(corrs[lag]))
        raw_corr["corr"].extend(corrs[lag])
        raw_corr["lag"].extend([lag, ]*len(corrs[lag]))
        raw_corr["id1"].extend([id1 for id1, id2 in pairs])
        raw_corr["id2"].extend([id2 for id1, id2 in pairs])

    all_corrs_df=pd.DataFrame(all_corrs)
    raw_corr_df=pd.DataFrame(raw_corr)

    return all_corrs_df, raw_corr_df

logger=logging.getLogger(__name__)
LAG_MIN=-1800*3
LAG_MAX=+1800*3

def coupling_analysis(
        dt_bin, number_of_animals, metadata, n_mins, figure_name, coupling_FUNs=COUPLING_FUNS, group_FUNs=GROUP_FUNS, bin_size=300, summary_FUN="mean", lag_min=LAG_MIN, lag_max=LAG_MAX
    ):
    """
    Quantify coupling of sleep rhythms in groups of animals

    Animals coinhabiting the same space may
    have some degree of coupling in their sleep rhythms.

    The coupling can be quantified in terms of:
        synchrony: fraction of time where all animals in the group are in the same sleep/wake state
        cross-correlation: correlation between the sleep/wake trajectory of 2 animals in a pair
    
    All quantifications are performed in pairs (pairwise), even if a group has 6 animals.
    In that case, 5+4+3+2+1=15 pairs are analyzed
    The sleep/wake trajectory is smoothed from the original frequency (1 Hz)
    to 1/300 (1 point every 5 mins) using the mean, for every animal independently

    Args:

        dt_bin (pd.DataFrame)

    Produce:

        coupling_df: for every pair between any 2 animals in the dataset, provide:
            id1: id of the first animal
            id2: id of the second animal
            value of the metric used to quantify their coupling
            lag at which this metric has been quantified
            comparison:
                1X if both animals were isolated
                real if both animals were in the same group
                virtual if both animals were in a group but not the same one
                NONE if one of the animals was isolated and the other wasnt

        summary_dataset:
            contains mean and sem of every group of pairs with the same comparison and lag

    """

    plt.clf()
    windowed_datasets={}
    logger.info("Quantifying sync in sleep bouts >= %s mins", n_mins)

    # prepare output folder

    ids=metadata["id"].unique().tolist()
    ids=sort_ids(ids)


    # compute pairwise coupling between animals using a coupling function
    logger.info("Quantify sleep coupling")
    n_flies=len(dt_bin["id"].unique())
    n_groups=len(dt_bin["experiment"].unique())
    print(f"n = {n_flies}, N = {n_groups}")

    lags=np.arange(lag_min//bin_size, lag_max//bin_size, 1) # units of bin size

    coupling_df=annotator(
        dt_bin, lags=lags,
        feature="inactive_rule",
        summary_FUN=summary_FUN,
        FUNs=coupling_FUNs,
        group_FUNs=group_FUNs,
        auto=False,
        nan=0,
        # nan=np.nan
    )
    coupling_df["n_mins"]=n_mins

    windowed_datasets[n_mins]=dt_bin
    plt.clf()

    # logger.info("Compute and save summary df")
    # summary_dataset=coupling_df.groupby(["comparison", "lag", "n_mins"]).agg({
    #     k: [np.mean, sem] for k in coupling_FUNs.values()
    # }).reset_index()
    # summary_dataset.columns=["comparison", "lag", "n_mins"] + list(itertools.chain(*[[
    #         f"{k}_mean", f"{k}_sem"
    #     ] for k in coupling_FUNs.values()
    # ]))
    return coupling_df #, summary_dataset
