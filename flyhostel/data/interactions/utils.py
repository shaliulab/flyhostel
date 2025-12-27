from tqdm.auto import tqdm
import itertools
import os.path
import glob
import datetime

import pandas as pd
import numpy as np

from flyhostel.data.deg import parse_chunk, parse_experiment, parse_local_identity, read_label_file_raw
from flyhostel.utils import (
    get_chunksize,
)
from .constants import DATA_DIR, METADATA_PIPELINE_FILE

def read_label_file_rejections(labels_file, chunksize):
    data_entry=os.path.basename(os.path.dirname(os.path.realpath(labels_file)))
    chunk=parse_chunk(data_entry, chunksize=chunksize)
    experiment=parse_experiment(data_entry, chunksize=chunksize)

    local_identity=parse_local_identity(data_entry, chunksize=chunksize)
    identity=int(data_entry.split("_")[-1])
    
    first_frame_number=int(data_entry.split("_")[5])
    last_frame_number=int(data_entry.split("_")[6])

    assert chunk == first_frame_number//chunksize
    
    labels=read_label_file_raw(labels_file, chunk=chunk, local_identity=local_identity)
    labels["frame_number"]=first_frame_number+np.arange(labels.shape[0])
    labels["first_frame"]=first_frame_number
    labels["last_frame_number"]=last_frame_number
    labels["frame_idx"]=labels["frame_number"]%chunksize
    labels["experiment"]=experiment
    labels["identity"]=identity
    labels["data_entry"]=data_entry
    labels["frame"]=labels["frame_number"]-first_frame_number
    labels.insert(0, "id", experiment[:26] + "|" + str(identity).zfill(2))
    return labels


def load_metadata(only_videos=True):
    metadata_pipeline=pd.read_csv(METADATA_PIPELINE_FILE, header=None)
    
    entries=[os.path.basename(os.path.dirname(f)) for f in glob.glob(f"{DATA_DIR}/*/*.mp4")]
    metadata=pd.DataFrame.from_records([("_".join(entry.split("_")[:4]), int(entry.split("_")[-1])) for entry in entries], columns=["experiment", "identity"])
    metadata.drop_duplicates(inplace=True)
    metadata_pipeline.columns=["basedir", "experiment", "number_of_animals", "complete", "select"]
    metadata_pipeline=metadata_pipeline.loc[metadata_pipeline["select"]=="SELECT"]
    
    if only_videos:
        how="left"
    else:
        how="outer"
    metadata=metadata.merge(
        metadata_pipeline[["experiment", "number_of_animals"]],
        on=["experiment"],
        how=how
    )

    metadata=metadata.loc[~metadata["identity"].isna()]

    count=metadata.groupby(["experiment", "number_of_animals"]).size().reset_index(name="animals_found")
    assert (count["animals_found"]==count["number_of_animals"]).all()
    metadata["identity"]=metadata["identity"].astype(int)

    return metadata


def verify_labels(file):
    timestamp_v2=datetime.datetime.strptime("2025-08-06 16:52:00", "%Y-%m-%d %H:%M:%S")
    return os.path.exists(file) and os.path.getmtime(file) > timestamp_v2.timestamp()


def annotate_pre_post_of_nn(df):
    df=df.merge(
        df[[
            "id", "nn", "frame_number",
            "pre_asleep", "pre_asleep_duration", "pre_asleep_bout_in", "pre_asleep_bout_out",
            "post_asleep", "post_asleep_duration", "post_asleep_bout_in", "post_asleep_bout_out",
            "pre_longImmobile", "pre_longImmobile_duration",
            # "latency_next_event"
        ]].rename({
            "id": "nn",
            "nn": "id",
            "pre_asleep": "pre_asleep_nn",
            "pre_asleep_duration": "pre_asleep_duration_nn",
            "post_asleep": "post_asleep_nn",
            "post_asleep_duration": "post_asleep_duration_nn",
            # "latency_next_event": "latency_next_event_nn",
            "pre_asleep_bout_in": "pre_asleep_bout_in_nn",
            "pre_asleep_bout_out": "pre_asleep_bout_out_nn", 
            "post_asleep_bout_in": "post_asleep_bout_in_nn",
            "post_asleep_bout_out": "post_asleep_bout_out_nn",
            "pre_longImmobile": "pre_longImmobile_nn",
            "pre_longImmobile_duration": "pre_longImmobile_duration_nn",
        }, axis=1),
        on=["id", "nn", "frame_number"],
        how="left"
    )
    return df


    # Load manual annotations
def load_manual_annotations(experiments, time_index):
    raise NotImplementedError()
    files=sorted(list(itertools.chain(*[glob.glob(f"{DATA_DIR}/{experiment}*/{experiment}*_labels.csv") for experiment in experiments])))
    labels=[]
    for file in tqdm(files, desc="Reading DEG-REJECTIONS labels database"):
        data=read_label_file_rejections(file)
        data["has_labels"]=verify_labels(file)
        labels.append(data)

    print("Concantenating labels into single dataframe")
    if labels:
        data=pd.concat(labels, axis=0).reset_index(drop=True)

        # TODO Change this to use real timestamps
        data["duration"]=(data["last_frame_number"]-data["first_frame"])/FRAMERATE
        
        # Annotate t_round in human made labels
        # runs in ~ 20 seconds
        data["frame_number_round"]=5*(data["frame_number"]//5)
        data=data.merge(time_index.rename({"frame_number": "frame_number_round"}, axis=1), on=["frame_number_round", "id"], how="left")
        data.drop("frame_number_round", axis=1, inplace=True)
    else:
        data=None

    return data
