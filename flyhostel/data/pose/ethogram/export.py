import os.path
import logging

import h5py
import numpy as np
from tqdm.auto import tqdm
logger=logging.getLogger(__name__)

from flyhostel.utils import (
    get_chunksize,
    get_basedir,
    get_dbfile,
    get_wavelet_downsample,
)

def reverse_behaviors(dataset, column="behavior"):

    for behavior in ["inactive", "pe"]:
        if behavior not in dataset.columns:
            dataset[behavior]=0
        dataset.loc[
            dataset[column]=="inactive+pe",
            behavior
        ]=dataset.loc[
            dataset[column]=="inactive+pe",
            "inactive+pe"
        ]

    for behavior in ["inactive", "twitch", "turn", "micromovement"]:
        if behavior not in dataset.columns:
            dataset[behavior]=0

        dataset.loc[
            dataset[column]=="inactive+micromovement",
            behavior
        ]=dataset.loc[
            dataset[column]=="inactive+micromovement",
            "inactive+micromovement"
        ]
    

    for behavior in ["inactive", "twitch", "rejection"]:
        if behavior not in dataset.columns:
            dataset[behavior]=0

        dataset.loc[
            dataset[column]=="inactive+rejection",
            behavior
        ]=dataset.loc[
            dataset[column]=="inactive+rejection",
            "inactive+rejection"
        ]
    
    for behavior in ["interactor", "interactee", "touch"]:
        if behavior not in dataset.columns:
            dataset[behavior]=0
    
    behaviors=["background", "walk", "groom", "feed", "pe", "inactive", "micromovement", "twitch", "turn", "rejection", "touch", "interactor", "interactee"]

    assert np.isnan(dataset[behaviors].values).sum()==0


    return dataset, behaviors

def save_deg_prediction_file(experiment, dataset, features, group_name="motionmapper", column="prediction2"):
    """
    Save the prediction in the same format as Deepethogram, so that it can be rendered in the GUI
    """
    
    chunk_lids=dataset[["chunk", "local_identity"]].drop_duplicates().sort_values("chunk").values.tolist()
    
    chunksize=get_chunksize(experiment)
    wavelet_downsample=get_wavelet_downsample(experiment)


    dataset, behaviors=reverse_behaviors(dataset, column=column)

    for chunk, local_identity in tqdm(chunk_lids, desc="Saving prediction files"):

        output_folder=f"{experiment}_{str(chunk).zfill(6)}_{str(local_identity).zfill(3)}"
        filename=f"{str(chunk).zfill(6)}_outputs.h5"
        os.makedirs(output_folder, exist_ok=True)
        output_path=os.path.join(output_folder, filename)

        dataset_this_chunk=dataset.loc[(dataset["chunk"]==chunk) & (dataset["local_identity"]==local_identity)]

        P=dataset_this_chunk[behaviors].values
        P = np.repeat(P, get_wavelet_downsample, axis=0)
        assert P.shape[0] == chunksize, f"{P.shape[0] != chunksize}"

        features_data=dataset_this_chunk[features].values
        features_data = np.repeat(features_data, get_wavelet_downsample, axis=0)

        logger.debug("Writing %s", output_path)
        with h5py.File(output_path, "w") as f:
            group=f.create_group(group_name)
            P_d=group.create_dataset("P", P.shape, dtype=np.float32)
            P_d[:]=P

            class_names=group.create_dataset("class_names", (len(behaviors), ), dtype="|S24")
            class_names[:]=[e.encode() for e in behaviors]

            thresholds=group.create_dataset("thresholds", (len(behaviors), ), dtype=np.float32)
            thresholds[:]=np.array([1, ] * len(behaviors))

            features_group=group.create_dataset("features", features_data.shape, dtype=np.float32)
            features_group[:]=features_data

            features_names=group.create_dataset("features_names", (len(features), ), dtype="|S100")
            features_names[:]=[e.encode() for e in features]
