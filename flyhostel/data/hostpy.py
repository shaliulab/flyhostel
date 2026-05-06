import logging

import pandas as pd
import joblib
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.loaders.centroids import to_behavpy

logger=logging.getLogger(__name__)


def load_hostel(metadata, n_jobs=1, value="dataframe"):
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

    if value=="dataframe":
        out=joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(
                FlyHostelLoader.load_single_hostel_from_metadata
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
        data=[d for d in out]
        meta=[d.meta for d in out]
        
        data = pd.concat(data, axis=0)
        meta = pd.concat(meta, axis=0)
        data=to_behavpy(data=data, meta=meta)
        return data
    
    elif value=="object":
        loaders={}
        for _, row in metadata.iterrows():
            loader=FlyHostelLoader.from_metadata(
                row["flyhostel_number"],
                row["number_of_animals"],
                row["flyhostel_date"],
                row["flyhostel_time"],
                row["identity"]
            )
            loaders[loader.datasetnames[0]]=loader
        return loaders

