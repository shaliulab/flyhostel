import numpy as np
import pandas as pd
from tqdm.auto import tqdm
interval_time=30

def quantify_activity_in_context(loaders, interval_times):
    """
    Quantify features of id and nn animals before and after the interactoin 
    """
    
    for loader1 in loaders:
        dfs=[]
        all_features=[]

        for loader2 in tqdm(loaders):
            if loader1==loader2:
                continue
            df1=loader1.interaction_ellipse2.loc[
                    (loader1.interaction_ellipse2["nn"]==loader2.ids[0])
            ]

            assert loader1.framerate==loader2.framerate
            framerate=loader1.framerate
            index=df1.groupby(["id", "nn", "interaction"]).first().reset_index()
            index["id_distance_pre"]=np.nan
            index["nn_distance_pre"]=np.nan
            index["id_distance_post"]=np.nan
            index["nn_distance_post"]=np.nan
            for i, row in index.iterrows():
                features=[]
                for interval_time in interval_times:
                    i0=row["first_frame"]-interval_time*framerate
                    i1=row["first_frame"]
                    
                    id_distance=loader1.dt.loc[
                        (loader1.dt["frame_number"]>=i0) & (loader1.dt["frame_number"]<i1),
                        "distance"
                    ].sum()
                    nn_distance=loader2.dt.loc[
                        (loader2.dt["frame_number"]>=i0) & (loader2.dt["frame_number"]<i1),
                        "distance"
                    ].sum()
                    index.loc[i, f"id_distance_pre_{interval_time}"]=id_distance
                    index.loc[i, f"nn_distance_pre_{interval_time}"]=nn_distance
                    
                    i0=row["last_frame_number"]
                    i1=row["last_frame_number"]+interval_time*framerate
                    
                    id_distance=loader1.dt.loc[
                        (loader1.dt["frame_number"]>=i0) & (loader1.dt["frame_number"]<i1),
                        "distance"
                    ].sum()
                    nn_distance=loader2.dt.loc[
                        (loader2.dt["frame_number"]>=i0) & (loader2.dt["frame_number"]<i1),
                        "distance"
                    ].sum()
                    index.loc[i, f"id_distance_post_{interval_time}"]=id_distance
                    index.loc[i, f"nn_distance_post_{interval_time}"]=nn_distance

                    features.extend([f"id_distance_pre_{interval_time}", f"nn_distance_pre_{interval_time}", f"id_distance_post_{interval_time}", f"nn_distance_post_{interval_time}"])
                df3=df1.merge(index[["id", "nn", "interaction"]+features], on=["id", "nn", "interaction"])
                dfs.append(df3)
                all_features=features
        loader1.interaction_ellipse3=pd.concat(dfs, axis=0).reset_index(drop=True)
    return all_features
    