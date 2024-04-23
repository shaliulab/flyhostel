import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from .correlation import compute_corrs
DEFAULT_LAGS=[-50, -40, -30, -20,-10, 0, 10, 20, 30, 40, 50]


def correlation_synchrony(dt_sleep, lags, interval=None):
    all_corrs={"corr": [], "lag": []}

    if interval is None:
        df=dt_sleep
    else:
        t0, t1=interval
        df=dt_sleep.loc[(dt_sleep["t"] >= t0) & ((dt_sleep["t"] < t1))]

    corrs=compute_corrs(df, lags=lags)
    for lag in tqdm(lags, desc="Computing corr at lag"):
        all_corrs["lag"].append(lag)
        all_corrs["corr"].append(np.mean(corrs[lag]))


    all_corrs_df=pd.DataFrame(all_corrs)
    return all_corrs_df