import numpy as np
import pandas as pd


def annotate_interaction_bouts(dt, max_distance_mm, min_bout=3):
    """
    Split agent pairwise distance timeseries into interaction bouts with annotated length and duration

    Arguments

        dt (pd.DataFrame): Dataset of pairwise agent distances with columns id1, id2, t, distance_mm
            t must contain timestamp
            distance_mm must contain distance
            id1 and id2 contains the identity of the pair of agents whose distance is stored
             
        max_distance_mm (int): Maximum distance between the pair of agents for the pair to be considered interacting
        bout_separation (int): Minimum time during which the two agents need to be within the maxa_distance_mm for the pair to be considered interacting
         
    """
    dt=dt.loc[dt["distance_mm"] <= max_distance_mm]
    dt["pair"] = list(zip(dt["id1"], dt["id2"]))
    dt.sort_values("t", inplace=True)

    all_pairs_bouts=[]
    for _, dt_single_pair in dt.groupby("pair"):
        
        dt_single_pair["new"]=np.concatenate([
            np.array([True]), np.diff(dt_single_pair["t"]) >= min_bout
        ])
        dt_single_pair["count"] = dt_single_pair["new"].cumsum()
        del dt_single_pair["new"]
        single_pair_bouts=[]
        for _, dt_bout in dt_single_pair.groupby("count"):
            dt_bout["length"]=dt_bout.shape[0]
            dt_bout=dt_bout.iloc[:1]
            single_pair_bouts.append(dt_bout)
        single_pair_bouts=pd.concat(single_pair_bouts)
        all_pairs_bouts.append(single_pair_bouts)

    all_pairs_bouts=pd.concat(all_pairs_bouts, axis=0)
    all_pairs_bouts.sort_values("distance", inplace=True)
    return all_pairs_bouts