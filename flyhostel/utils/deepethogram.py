import os.path
import h5py
import logging

import numpy as np
import pandas as pd
from deepethogram.postprocessing import get_postprocessor_from_cfg
from deepethogram import projects

logger=logging.getLogger(__name__)

def load_predictions(h5_file, dataset):
    """
    Given an h5_file and a prediction dataset inside it,
    load them and apply postprocessing just as made in the deepethogram GUI
    """
    assert os.path.exists(h5_file)

    project_path=os.path.dirname(
        os.path.dirname(os.path.dirname(h5_file))
    )
    cfg = projects.get_config_from_path(project_path)
    try:
        with h5py.File(h5_file) as f:
            # key=os.path.basename(h5_file).replace("_outputs.h5", "")
            thresholds=f[dataset]["thresholds"][:]
            behaviors=[e.decode() for e in f[dataset]["class_names"][:]]
            thresholds = f[dataset]["thresholds"][:]
            postprocessor = get_postprocessor_from_cfg(cfg, thresholds)

            p=f[dataset]["P"][:]
    except Exception as error:
        logger.error(error)
        return None, None

    predictions = postprocessor(p)
    return predictions, behaviors

def load_predictions_to_df(h5_file, *args, **kwargs):
    predictions, behaviors=load_predictions(h5_file, *args, **kwargs)
    tokens=os.path.basename(h5_file).split("_")
    fn0, fn1, local_identity, identity=[int(e) for e in tokens[5:9]]

    if predictions is None:
        data=None
    else:
        data=pd.DataFrame(predictions, columns=behaviors)
        data["frame_number"]=fn0+np.arange(data.shape[0])
        data["local_identity"]=local_identity
        data["identity"]=identity
    return data
