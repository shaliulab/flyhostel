import numpy as np
import pandas as pd
import zeitgeber
DEFAULT_STRIDE=1


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

    if all_pairs_bouts:
        all_pairs_bouts=pd.concat(all_pairs_bouts, axis=0)
        all_pairs_bouts.sort_values("distance", inplace=True)
        return all_pairs_bouts
    else:
        return None
    

def compute_bouts_(pos_events_df):

    pos_events_df["df"]=np.concatenate([[np.inf], np.diff(pos_events_df["frame_number"])])
    stride=np.unique(pos_events_df["df"], return_counts=True)[0][0]


    encoding = zeitgeber.rle.encode([str(e)[0] for e in pos_events_df["df"]<=stride])
    encoding_df=pd.DataFrame.from_records(encoding, columns=["status", "length"])
    if encoding_df.shape[0]%2==1:
        encoding_df=encoding_df.iloc[:-1]
        
    encoding_df["bout"]=np.repeat(np.arange(encoding_df.shape[0]//2), 2)
    encoding_df["row"]=encoding_df["length"].cumsum()
    encoding_df["frame_number"]=pos_events_df["frame_number"].iloc[(encoding_df["row"]-1)].values

    encoding_df=encoding_df.groupby("bout").apply(lambda df: [df["frame_number"].iloc[0], df["length"].iloc[1]+1]).reset_index()
    encoding_df["frame_number"]=[e[0]for e in encoding_df[0]]

    encoding_df["length"]=[e[1]for e in encoding_df[0]]
    del encoding_df[0]

    return encoding_df, stride