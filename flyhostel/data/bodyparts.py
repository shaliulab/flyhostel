
"""

"""

import itertools
import numpy as np
import pandas as pd
import joblib
ROI_WIDTH=ROI_HEIGHT=100
anchor_bp="thorax"


def parse_identity(id):
    return int(id.split("|")[1])


def make_absolute_pose_coordinates(dt, bodyparts, roi_width):
    """
    Convert to absolute coordinates
    
    dt contaions the coordinates of the centroid in units relative to the roi_width e.g. 1,1 means bottom right corner
    and the coordinates of the body parts relative to the top left corner of the inset of the animal (50 pixels up and right from centroid)
    to obtain all bodyparts in absolute:

        1. Obtain the absolute coordinates of the centroid, by multiplying by the roi width and height
        2. Obtain the absolute coordinates of the other parts, by adding to the coordinate of the top left corner
            the relative coordinate of the body part
    """

    dt["centroid_x"] = dt["x"] *roi_width
    dt["tl_x"] = dt["centroid_x"]-ROI_WIDTH//2
    dt["centroid_y"] = dt["y"] *roi_width
    dt["tl_y"] = dt["centroid_y"]-ROI_HEIGHT//2
    dt[f"{anchor_bp}_x_original"] = dt[f"{anchor_bp}_x"]
    dt[f"{anchor_bp}_y_original"] = dt[f"{anchor_bp}_y"]
    for bp in bodyparts:
        for coord in ["x", "y"]:
            dt[bp + "_" + coord] = dt["tl_" + coord] + dt[bp + "_" + coord]
    dt["identity"] = [parse_identity(identity) for identity in dt["id"]]
    del dt["tl_y"]
    del dt["tl_x"]
    return dt



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

def find_closest_bps_parallel(interactions, n_jobs=-2, partition_size=100, **kwargs):
    """
    For all instances where two animals are very close, return the bodyparts of each
    that are closest in space

    Arguments:

    * interactions (pd.DataFrame): Every
    """
    
    out=[]
    for i, (key, df) in enumerate(interactions.groupby("interaction")):
        if i % partition_size == 0:
            out.append([])
        out[i//partition_size].append((df, key, i))
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


from flyhostel.data.pose.constants import bodyparts, legs, labels