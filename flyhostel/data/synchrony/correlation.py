import itertools
import random
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.stats import kendalltau
from .correlation_utils import (
    real_groups_iter,
    virtual_combinations_iter
)

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

def rmse(df, col1, col2, lag=0):
    """

    From fig2 https://doi.org/10.1016/j.tree.2024.07.011

    Compute RMSE between two columns of a DataFrame with a given lag.

    Parameters:
    - df: pandas DataFrame.
    - col1: The name of the first column.
    - col2: The name of the second column.
    - lag: The lag introduced. Positive values will lag col2. Lag unit is not time, but data point

    Returns:
    - Cross-correlation value.
    """
    series1, series2, n_points=preprocess(df, col1, col2, lag)
    selected=np.bitwise_and(
        np.bitwise_not(series1.isna()),
        np.bitwise_not(series2.isna()),
    )

    M=0
    m=1

    coef=1 - (1/(M-m)) * np.sqrt(np.mean(
        (series1[selected]-series2[selected])**2
    ))
    return coef

def preprocess(df, col1, col2, lag):
    """
    
    """
    
    if lag == 0:
        series1=df[col1]
        series2=df[col2]
    else:
        series1 = pd.Series(df[col1].tolist()[lag:] + df[col1].tolist()[:lag])
        series2 = pd.Series(df[col2].tolist()[(-lag):] + df[col2].tolist()[:(-lag)])

    selected=np.where(np.bitwise_and(
        np.bitwise_not(series1.isna()),
        np.bitwise_not(series2.isna()),
    ))[0]

    n_points=selected.sum().item()

    series1=series1.iloc[selected].values.flatten()
    series2=series2.iloc[selected].values.flatten()

    return series1, series2, n_points

def euclidean_distance(df, col1, col2, lag=0, nan=0):
    """
    Compute euclidean distance between two columns of a DataFrame with a given lag.

    Parameters:
    - df: pandas DataFrame.
    - col1: The name of the first column.
    - col2: The name of the second column.
    - lag: The lag introduced. Positive values will lag col2. Lag unit is not time, but data point

    Returns:
    - Cross-correlation value.
    """
    series1, series2, n_points=preprocess(df, col1, col2, lag)
    distance=np.sqrt(((series1-series2)**2).sum())
    if isinstance(distance, pd.Series):
        if len(distance) > 1:
            print(col1, col2)
            # print(series2)
            print(df[col2])

        distance=distance.item()

    if np.isnan(distance):
        distance=nan
    return distance, n_points

def psi(df, col1, col2, lag, nan=0):
    """
    Population Synchrony Index
    https://www.science.org/doi/10.1126/science.adr3339
    """

    series1, series2, n_points=preprocess(df, col1, col2, lag)
    mean_across_animals=np.stack([series1, series2]).mean(axis=0)

    mean_across_animals=mean_across_animals[~np.isnan(mean_across_animals)]
    coefficient_of_variation=np.std(mean_across_animals) / np.mean(mean_across_animals)
    return coefficient_of_variation, n_points

def group_psi(df, lag, nan=0):
    """
    Population Synchrony Index
    https://www.science.org/doi/10.1126/science.adr3339
    """
    col1, col2=df.columns[:2]
    series1, series2, n_points=preprocess(df, col1, col2, lag)
    series=[series1, series2]
  
    for id in df.columns[2:]:
        _, series_n, n_points=preprocess(df, col1, id, lag)
        series.append(series_n)
   
    mean_across_animals=np.stack(series).mean(axis=0)
    mean_across_animals=mean_across_animals[~np.isnan(mean_across_animals)]
    coefficient_of_variation=np.std(mean_across_animals) / np.mean(mean_across_animals)
    return coefficient_of_variation, n_points


def cross_correlationv2(df, col1, col2, lag=0, nan=0):
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
    series1, series2, n_points=preprocess(df, col1, col2, lag)
    pearson=np.corrcoef(series1, series2)[0, 1]
    if np.isnan(pearson):
        pearson=nan
    return pearson, n_points


def agreement(df, col1, col2, lag=0, nan=0):
    """
    Compute level of agreement between two columns of a DataFrame with a given lag.

    Parameters:
    - df: pandas DataFrame.
    - col1: The name of the first column.
    - col2: The name of the second column.
    - lag: The lag introduced. Positive values will lag col2. Lag unit is not time, but data point

    Returns:
    - Agreement value.
    """
    series1, series2, n_points=preprocess(df, col1, col2, lag)
    score=(series1==series2).mean()

    if np.isnan(score):
        score=nan
    return score, n_points


def mean_squared_difference(df, col1, col2, lag=0, nan=0):
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
    series1, series2, n_points=preprocess(df, col1, col2, lag)
    msq_diff=((series1-series2)**2).mean()

    if np.isnan(msq_diff):
        msq_diff=nan
    return msq_diff, n_points


def annotator(df, lags, feature="asleep", FUNs={}, auto=False, summary_FUN="mean", **kwargs):

    assert df.shape[0]>0
    wide_table=df.reset_index().pivot_table(index=['t'], columns='id', values=feature)
    bool_table=(wide_table==1).astype(int)
    diff_table=bool_table.diff(axis=0).iloc[1:]

    tables={
        "mean": wide_table,
        "P_doze": (diff_table==+1).astype(int),
        "P_wake": (diff_table==-1).astype(int),
    }
    
    X=tables[summary_FUN]
    experiment_index=df[["id",  "experiment"]].drop_duplicates().reset_index(drop=True)


    ids=X.columns.tolist()
    pairs=list(itertools.combinations(ids, 2))
    if auto:
        for id in ids:
            pairs.append((id, id))

    records=[]
    for lag in tqdm(lags, desc="Analyzing lags"):
        for id1, id2 in pairs:

            if id1[:26]==id2[:26]:
                experiment=experiment_index.loc[experiment_index["id"]==id1, "experiment"].iloc[0]
            else:
                experiment="virtual"
            for FUN, FUN_name in FUNs.items():
                val, N=FUN(X, id1, id2, lag=lag, **kwargs)
                records.append((
                    id1, id2, lag, val, FUN_name, N, experiment
                ))

      
        for experiment, ids_per_groups in real_groups_iter(ids):
            for FUN, FUN_name in group_FUNs.items():
                val, N=FUN(X[ids_per_groups], lag=lag, **kwargs)
                for id1, id2 in itertools.combinations(ids_per_groups, 2):
                    records.append((
                        id1, id2, lag, val, FUN_name, N, experiment
                    ))
        
        virtual_combos=random.sample(list(virtual_combinations_iter(ids)), 100)
        counter=0
        for experiment, ids_per_groups in virtual_combos:
            for FUN, FUN_name in group_FUNs.items():
                val, N=FUN(X[ids_per_groups], lag=lag, **kwargs)
                for id1, id2 in itertools.combinations(ids_per_groups, 2):
                    records.append((
                        id1, id2, lag, val, FUN_name, N, f"{experiment}_{counter}"
                    ))
            counter+=1
    
    df=pd.DataFrame.from_records(
        records,
        columns=["id1", "id2", "lag", "value", "metric", "N", "experiment"]
    )
    df=df.pivot(index=["id1", "id2", "lag", "N", "experiment"], columns="metric", values="value").reset_index()
    return df
