import os.path
import yaml
import joblib
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import logging
logger=logging.getLogger(__name__)

# Update local_identity
def update_identity(df, field="identity", n_jobs=1):
    """
    Replaces all instances of local identity = 0 by 
    negative local identities so that no two flies have the same local identity=0
    """

    frame_number_with_missing_flies=df.loc[df["local_identity"]==0, "frame_number"].unique()
    shape=df.shape

    df_ok=df.loc[~df["frame_number"].isin(frame_number_with_missing_flies)].copy()
    df_not_ok=df.loc[df["frame_number"].isin(frame_number_with_missing_flies)].copy()
    del df

    if df_not_ok.shape[0] > 0:
        df_not_ok.sort_values(["frame_number", "fragment"], inplace=True)
        diff=np.diff(df_not_ok["frame_number"])
        new_scene=np.concatenate([
            [True],
            diff>1
        ])
        df_not_ok["scene"]=np.cumsum(new_scene)

        dfs=update_identity_in_all_scenes(df_not_ok, n_jobs=n_jobs, field=field)

        df_not_ok=pd.concat(dfs, axis=0)
        df=pd.concat([
            df_ok,
            df_not_ok
        ], axis=0)
    else:
        df=df_ok

    df.sort_values(["frame_number", "fragment"], inplace=True)

    assert shape[0]==df.shape[0], f"{shape[0]}!={df.shape[0]}"

    return df


def update_identity_in_all_scenes(df, n_jobs, field="identity"):
    dfs=joblib.Parallel(
        n_jobs=n_jobs
    )(
        joblib.delayed(
            update_identity_in_scene
        )(
            df_scene.copy(), field=field, scene_id=scene_id
        )
        for scene_id, df_scene in df.groupby("scene")
    )
    return dfs


def update_identity_in_scene(df, field="identity", scene_id=None):

    # Track the last used identity for animals with identity == 0
    counter = 0
    # Dictionary to store the fragment and its updated identity
    fragment_identity = {}

    out_df=df.copy()

    for index, row in tqdm(df.iterrows()):

        fragment_identifier=row['fragment'].item()
        if row[field] == 0:
            if fragment_identifier in fragment_identity:
                # If fragment already encountered, use the stored identity
                out_df.at[index, field] = fragment_identity[fragment_identifier]
            else:
                # Assign new identity and update the counter and dictionary
                counter-=1
                out_df.at[index, field] = counter
                fragment_identity[fragment_identifier] = counter

    if scene_id is not None:
        logfile=os.path.join(
            "logs",
            str(scene_id).zfill(6) + "_" + str(df["frame_number"].iloc[0]) + "_status.txt"
        )

        metadata={
            "scene_size": df.shape[0],
            "min_counter": counter,
            "fragment_identity": fragment_identity
        }
        logger.debug(metadata)
        with open(logfile, "w") as handle:
            yaml.dump(metadata, handle, yaml.SafeDumper)

    return out_df