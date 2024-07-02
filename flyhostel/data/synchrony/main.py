import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .correlation import annotator, cross_correlation
DEFAULT_LAGS=[-50, -40, -30, -20,-10, 0, 10, 20, 30, 40, 50]


def compute_synchrony(dt_sleep, lags, interval=None, feature="asleep", FUN=annotator, FUN_metric=cross_correlation):
    """
    Compute synchrony in the behavior of a group of flies using the pearson correlation

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
        # import ipdb; ipdb.set_trace()
        all_corrs["lag"].append(lag)
        all_corrs["corr"].append(np.mean(corrs[lag]))
        raw_corr["corr"].extend(corrs[lag])
        raw_corr["lag"].extend([lag, ]*len(corrs[lag]))
        raw_corr["id1"].extend([id1 for id1, id2 in pairs])
        raw_corr["id2"].extend([id2 for id1, id2 in pairs])

    all_corrs_df=pd.DataFrame(all_corrs)
    raw_corr_df=pd.DataFrame(raw_corr)

    return all_corrs_df, raw_corr_df