import logging

import pandas as pd
import joblib
import numpy as np

from motionmapperpy import setRunParameters
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.loaders.centroids import to_behavpy
wavelet_downsample=setRunParameters().wavelet_downsample

logger=logging.getLogger(__name__)


def load_hostel(metadata, n_jobs=1):
    """
    Load centroid and pose data for every fly provided in the metadata    
    """
    metadata=metadata.loc[
        (metadata["flyhostel_number"] != "NONE") &
        (metadata["flyhostel_date"] != "NONE") &
        (metadata["flyhostel_time"] != "NONE") &
        (metadata["identity"] != "NONE") &
        (metadata["number_of_animals"] != "NONE")
    ]

    out=joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            load_single_hostel_from_metadata
        )(
            *args
        )
        for args in zip(
            metadata["flyhostel_number"],
            metadata["number_of_animals"],
            metadata["flyhostel_date"],
            metadata["flyhostel_time"],
            metadata["identity"]
        )
    )
    data=[e[0] for e in out]
    meta=[e[1] for e in out]
    
    data = pd.concat(data, axis=0)
    meta = pd.concat(meta, axis=0)
    data=to_behavpy(data=data, meta=meta)
    return data
        

def load_single_hostel_from_metadata(flyhostel_number, number_of_animals, flyhostel_date, flyhostel_time, identity):
    experiment = f"FlyHostel{flyhostel_number}_{number_of_animals}X_{flyhostel_date}_{flyhostel_time}"
    dt=load_single_hostel(experiment=experiment, identity=identity)
    return dt


def load_single_hostel(experiment, identity):

    loader=FlyHostelLoader(experiment=experiment, identity=int(identity), chunks=range(0, 400))
    loader.load_and_process_data(
        stride=1,
        cache="/flyhostel_data/cache",
        filters=None,
        useGPU=0,  
    )

    data=loader.dt.copy()
    meta=data.meta.copy()
    centroid_columns=data.columns.tolist()
    data=data.loc[data["frame_number"] % wavelet_downsample == 0]

    if loader.behavior is None:
        logger.warning("Behavior not computed for %s", loader)
        data["behavior"]=np.nan
        data["score"]=np.nan
        data["bout_in"]=np.nan
        data["bout_out"]=np.nan
        data["duration"]=np.nan
    else:
        data=data.merge(loader.behavior, on=["id", "frame_number"], how="left")


    fields = centroid_columns + ["behavior",  "score", "bout_in", "bout_out", "duration"]
    data=data[fields]
    return data, meta

