
"""

"""

import itertools
import numpy as np
import pandas as pd
import joblib


def make_absolute_pose_coordinates(dt, bodyparts, roi_width):
    # convert to absolute coordinates
    # dt2 contaions the coordinates of the centroid in units relative to the roi_width e.g. 1,1 means bottom right corner
    # and the coordinates of the body parts relative to the top left corner of the inset of the animal (50 pixels up and right from centroid)
    # to obtain the centroid in absolute 
    dt_absolute=dt.copy()


    dt_absolute["centroid_x"] = dt_absolute["x"] *roi_width
    dt_absolute["tl_x"] = dt_absolute["centroid_x"]-50 #dt_absolute["thorax_x"]
    dt_absolute["centroid_y"] = dt_absolute["y"] *roi_width
    dt_absolute["tl_y"] = dt_absolute["centroid_y"]-50 #dt_absolute["thorax_y"]
    dt_absolute["thorax_x_original"] = dt_absolute["thorax_x"]
    dt_absolute["thorax_y_original"] = dt_absolute["thorax_y"]
    for bp in bodyparts:
        for coord in ["x", "y"]:
            dt_absolute[bp + "_" + coord] = dt_absolute["tl_" + coord] + dt[bp + "_" + coord]
    dt_absolute["identity"] = [int(e.split("|")[1]) for e in dt_absolute.index]
    for coord in ["x", "y"]:
        dt_absolute[coord] = dt[coord]

    return dt_absolute



def find_closest_bps_all(iterable, max_distance=None, parts=[]):
    out=[]
    for df, key, i in iterable:
        out.append(find_closest_bps(df, key, i, bodyparts=parts))
    
    out=pd.concat(out, axis=0)
    out.reset_index(inplace=True)
    del out["index"]
    
    out["frame_number"]=[e[0] for e in out["interaction"]]
    
    out_filtered = out.copy()
    # if parts is not None:
    #     out_filtered = filter_parts(out_filtered, parts)
    
    if max_distance is not None:
        out_filtered=out_filtered.loc[out_filtered["distance"]<=max_distance] 
        
    return out, out_filtered
        
def find_closest_bps(df, key, i, bodyparts=[]):
    """
    For any two flies contained in df, return the bodypart of each which are closest in space
    """
    
    assert df.shape[0]==2
    assert len(bodyparts) > 0
    all_bps=[]
    df.sort_values("id1", inplace=True)
    for bp in bodyparts:

        x0=np.repeat(df[bp+"_x"].values, len(bodyparts)) # bp repeated n times for one animal and then again for second animal
        y0=np.repeat(df[bp+"_y"].values, len(bodyparts))

        x1=np.concatenate(
            [df.iloc[[1]][bp + "_x"].values for bp in bodyparts] +
            [df.iloc[[0]][bp + "_x"].values for bp in bodyparts]
        ) # all bodyparts of one animal, and then of second animal

        y1=np.concatenate(
            [df.iloc[[1]][bp + "_y"].values for bp in bodyparts] +
            [df.iloc[[0]][bp + "_y"].values for bp in bodyparts]
        ) # all bodyparts of one animal, and then of second animal

        x = pd.DataFrame({
            "bp1": bp,
            "bp2": itertools.chain(*[bodyparts, ]*2),
            "id1": np.repeat(df["id1"], len(bodyparts)),
            "id2": np.repeat(df["id2"], len(bodyparts)),
            "distance": np.sqrt((x1-x0)**2 + (y1-y0)**2),
            "interaction": [key,] * len(bodyparts)*2,
            "t": [df["t"].unique()[0],] * len(bodyparts)*2,
        })

        all_bps.append(x)

    all_bps=pd.concat(all_bps)
    all_bps=all_bps.iloc[[all_bps["distance"].argmin()]]
    return all_bps

def find_closest_bps_parallel(interactions, n_jobs=-2, chunksize=100, **kwargs):
    """
    For all instances where two animals are very close, return the bodyparts of each
    that are closest in space
    """
    
    out=[]
    for i, (key, df) in enumerate(interactions.groupby("interaction")):
        if i % chunksize == 0:
            out.append([])
        out[i//chunksize].append((df, key, i))
    out2 = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(find_closest_bps_all)(
            iterable, **kwargs
        )
        for  iterable in out
    )
    out=pd.concat([e[0] for e in out2], axis=0)
    out_filtered=pd.concat([e[1] for e in out2], axis=0)
        
    return out, out_filtered


def filter_parts(out, parts):
    out=out.loc[
        (out["bp1"].isin(parts)) &
        (out["bp2"].isin(parts))
    ]
    return out


bodyparts = [
    'thorax', 'abdomen', 'foreLeft_Leg', 'foreRightLeg', 'head', 'leftWing',
    'midLeftLeg', 'midRightLeg', 'proboscis', 'rearLeftLeg',
    'rearRightLeg', 'rightWing'
]
legs = [bp for bp in bodyparts if "leg" in bp.lower()]
wings = [bp for bp in bodyparts if "wing" in bp.lower()]
core = ["thorax", "abdomen", "head", "proboscis"]