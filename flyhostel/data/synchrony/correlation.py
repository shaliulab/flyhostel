import itertools
import numpy as np
from scipy.stats import kendalltau

def phi_corr(series1, series2):
    assert all([e in [0, 1] for e in series1])
    assert all([e in [0, 1] for e in series2])
    
    a = np.sum((series1 == 1) & (series2 == 1))
    b = np.sum((series1 == 1) & (series2 == 0))
    c = np.sum((series1 == 0) & (series2 == 1))
    d = np.sum((series1 == 0) & (series2 == 0))
    
    denominator = np.sqrt((a+b) * (a+c) * (b+d) * (c+d))
    
    # Protect against division by zero
    if denominator == 0:
        return 0

    return (a * d - b * c) / denominator


def cross_correlation(df, col1, col2, lag=0):
    """
    Compute cross-correlation between two columns of a DataFrame with a given lag.

    Parameters:
    - df: pandas DataFrame.
    - col1: The name of the first column.
    - col2: The name of the second column.
    - lag: The lag introduced. Positive values will lag col2. Lag unit is not time, but data point

    Returns:
    - Cross-correlation value.
    """
    if lag > 0:
        series1 = df[col1].iloc[lag:]
        series2 = df[col2].iloc[:-lag]
    elif lag == 0:
        series1=df[col1]
        series2=df[col2]
    else:
        series1 = df[col1].iloc[:lag]
        series2 = df[col2].iloc[-lag:]
    # return phi_corr(series1, series2)
    # return kendalltau(series1, series2).correlation
    selected=np.bitwise_and(
        np.bitwise_not(series1.isna()),
        np.bitwise_not(series2.isna()),
    )
    return np.corrcoef(series1[selected], series2[selected])[0, 1]


def annotator(df, lags, feature="asleep", FUN=cross_correlation):
    corrs={}
    wide_table=df.reset_index().pivot_table(index=['t'], columns='id', values=feature)
    ids=wide_table.columns.tolist()
    pairs=list(itertools.combinations(ids, 2))

    for lag in lags:
        corrs[lag]=[
            FUN(wide_table, id1, id2, lag=lag)
            for id1, id2 in pairs
        ]
        
    return corrs, pairs