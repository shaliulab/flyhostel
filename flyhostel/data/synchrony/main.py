import numpy as np
import pandas as pd
from .correlation import compute_corrs
DEFAULT_LAGS=[-50, -40, -30, -20,-10, 0, 10, 20, 30, 40, 50]


def correlation_synchrony(dt_sleep, lags):
    all_corrs={"corr": [], "lag": []}
    corrs=compute_corrs(dt_sleep.loc[(dt_sleep["t"] >= 14*3600) & ((dt_sleep["t"] < 22*3600))], lags=lags)
    for lag in lags:
        all_corrs["lag"].append(lag)
        all_corrs["corr"].append(np.mean(corrs[lag]))


    all_corrs_df=pd.DataFrame(all_corrs)
    return all_corrs_df