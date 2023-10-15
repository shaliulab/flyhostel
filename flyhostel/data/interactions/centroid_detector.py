import pandas as pd
import numpy as np

import zeitgeber.rle


from flyhostel.data.interactions.utils import get_metadata_prop

def centroid_interaction_detector(animal0, animal1, dt0, dt1, dt_index, distance_threshold=0.15):
    """
    Arguments:

        distance_threshold (float): Max distance in cm between animals during interaction
    """

    pixels_per_cm0 = int(float(get_metadata_prop(animal0, "pixels_per_cm")))
    pixels_per_cm1 = int(float(get_metadata_prop(animal1, "pixels_per_cm")))
    chunksize = int(float(get_metadata_prop(animal0, "chunksize")))

    xy0=dt0.loc[:, pd.IndexSlice[:, "centroid", ["x", "y"]]].values / pixels_per_cm0
    xy1=dt1.loc[:, pd.IndexSlice[:, "centroid", ["x", "y"]]].values / pixels_per_cm1
    diff=np.diff(np.stack([xy0, xy1], axis=2), axis=2)[:, :, 0]
    distance=np.sqrt(np.sum(diff**2, axis=1))


    interactions=zeitgeber.rle.encode([str(e)[0] for e in (distance < distance_threshold).tolist()])
    interactions=pd.DataFrame.from_records(interactions, columns=["status", "length"])
    
    # save the index or position of the bout of interaction / non interaction
    interactions["position"]=[0] + interactions["length"].cumsum().tolist()[:-1]

    # convert this position into the experiment frame number
    # use the frame number index of the first animal
    interactions["frame_number"]=dt_index["frame_number"].values[interactions["position"].values]

    # annotate chunk and frame idx from frame number
    interactions["chunk"]=interactions["frame_number"]//chunksize
    interactions["frame_idx"]=interactions["frame_number"]%chunksize


    # keep positive bouts
    interactions=interactions.loc[(interactions["status"]=="T")] #  & (interactions["length"]<=20)
    return interactions
