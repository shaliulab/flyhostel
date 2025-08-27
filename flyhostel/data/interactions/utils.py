import os.path
import glob
import datetime

import pandas as pd
import numpy as np
from flyhostel.data.pose.constants import chunksize as CHUNKSIZE
from flyhostel.data.deg import parse_chunk, parse_experiment, parse_local_identity, read_label_file_raw
from .constants import DATA_DIR, METADATA_PIPELINE_FILE

def read_label_file_rejections(labels_file):
    data_entry=os.path.basename(os.path.dirname(os.path.realpath(labels_file)))
    
    local_identity=parse_local_identity(data_entry)
    identity=int(data_entry.split("_")[-1])
    
    chunk=parse_chunk(data_entry)
    experiment=parse_experiment(data_entry)
    
    first_frame_number=int(data_entry.split("_")[5])
    last_frame_number=int(data_entry.split("_")[6])

    assert chunk == first_frame_number//CHUNKSIZE
    
    labels=read_label_file_raw(labels_file, chunk=chunk, local_identity=local_identity)
    labels["frame_number"]=first_frame_number+np.arange(labels.shape[0])
    labels["first_frame"]=first_frame_number
    labels["last_frame_number"]=last_frame_number
    labels["frame_idx"]=labels["frame_number"]%CHUNKSIZE
    labels["experiment"]=experiment
    labels["identity"]=identity
    labels["data_entry"]=data_entry
    
    return labels


def load_metadata():
    metadata_pipeline=pd.read_csv(METADATA_PIPELINE_FILE, header=None)
    
    entries=[os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{DATA_DIR}/*/*_labels.csv")]
    metadata=pd.DataFrame.from_records([("_".join(entry.split("_")[:4]), int(entry.split("_")[-1])) for entry in entries], columns=["experiment", "identity"])
    metadata.drop_duplicates(inplace=True)
    metadata_pipeline.columns=["basedir", "experiment", "number_of_animals", "complete", "select"]
    metadata_pipeline=metadata_pipeline.loc[metadata_pipeline["select"]=="SELECT"]
    metadata=metadata.merge(
        metadata_pipeline[["experiment"]],
        on=["experiment"],
        how="left"
    )
    return metadata


def verify_labels(file):
    timestamp_v2=datetime.datetime.strptime("2025-08-06 16:52:00", "%Y-%m-%d %H:%M:%S")
    return os.path.exists(file) and os.path.getmtime(file) > timestamp_v2.timestamp()