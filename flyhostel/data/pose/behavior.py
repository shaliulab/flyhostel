import os.path
import logging
import pickle

import numpy as np
import plotly.express as px
from tqdm.auto import tqdm
import pandas as pd

from flyhostel.data.pose.main import FlyHostelLoader
STRIDE=5


logger=logging.getLogger(__name__)


def load_model(model_path):
    try:
        with open(model_path, "rb") as handle:
            model=pickle.load(handle)
        return model
    except TypeError as error:
        logger.error("%s cannot be loaded from this Python due to pickle issues")
        raise error       


def predict_behavior(experiment, model_path, identity=None, wavelets=None, output=None, **kwargs):
    """
    Project pose dataset using pretrained model

    The cached model must provide an object with a transform() method
    which takes the pose features and returns an array with the same number of rows
    and at least 2 columns representing the high dimensional space
    
    predict_experiment("FlyHostelN_MX_YYYY-MM-DD_HH-MM-SS", "knn.pkl", 1)
    """
    raise DeprecationWarning()

    loader = FlyHostelLoader(experiment, identity=identity, chunks=range(0, 400))
    loader.load_and_process_data(
        stride=STRIDE,
        cache="/flyhostel_data/cache",
        filters=None,
        useGPU=0,
        **kwargs,
    )

    if loader.identity is not None:
        loader.load_behavior_data(loader.experiment, loader.identity, loader.pose_boxcar)
    
    basedir=loader.basedir

    
    labeled_dataset, unknown_dataset, (_, freq_names)=loader.load_dataset(wavelets=wavelets)
    labeled_dataset["ground_truth"]=labeled_dataset["behavior"]
    labeled_dataset.drop("behavior", axis=1, inplace=True)
    unknown_dataset["ground_truth"]=np.nan
    dataset=pd.concat([labeled_dataset, unknown_dataset])
    del unknown_dataset
    del labeled_dataset
    del loader

    if identity is not None:
        ids=sorted(dataset["id"].unique())
        id=ids[int(identity)-1]
        dataset=dataset.loc[dataset["id"]==id]
    del identity

    model = load_model(model_path)

    dataset.sort_values(["id", "frame_number"], inplace=True)
    
    # yes, I want to downsample again
    # If you change this, you have to change the stride of the wavelets as well
    # dataset=dataset.iloc[::STRIDE]


    d_input=dataset[freq_names].values
    logger.debug("Transforming dataset of shape %s with umap %s", d_input.shape, model_path)
    projection=model.transform(d_input)

    assert projection.shape[1]>=2

    features=[]
    id_cols=["id", "frame_number", "chunk", "frame_idx", "t"]
    for i in range(projection.shape[1]):
        feature=f"C_{i+1}"
        features.append(feature)
        dataset[feature]=projection[:,i]
    
    for id, df in tqdm(dataset.groupby("id"), desc="Exporting"):

        identity=str(id.split("|")[1]).zfill(2)
        
        if output is None:
            output_folder=os.path.join(basedir, "motionmapper", identity)
        else:
            output_folder=output
        
        os.makedirs(output_folder, exist_ok=True)
        px_path=os.path.join(output_folder, f"{experiment}__{identity}.html")
        csv_path=os.path.join(output_folder, f"{experiment}__{identity}.csv")

        df[id_cols + features + ["behavior"]].to_csv(csv_path)
        fig=px.scatter(
            df, x="C_1", y="C_2",
            hover_data=id_cols + ["behavior"],
        )
        logger.debug("Saving to ---> %s", px_path)
        fig.write_html(px_path)