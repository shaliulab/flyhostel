import argparse
import datetime
import os
import sys
import glob
import sqlite3
import h5py
import pickle
import logging


import seaborn as sns
import numpy as np
import pandas as pd
# logging.getLogger("bsoid_app.cli.video_creator.video_creator_chain").setLevel(logging.INFO)

sys.path.append("/home/vibflysleep/opt/B-SOID/")
from bsoid_app.cli import data_preprocess
from bsoid_app.cli.utils import (
    interpolate_between_contiguous_predictions,
    filter_data,
    compute_speed,
    get_video_files,
)

from flyhostel.data.pose.constants import *


logging.basicConfig(level=logging.INFO)


# experiment="FlyHostel3/6X/2023-06-27_14-00-00"
# number_of_animals=6
# chunks=np.arange(100, 300, 1)


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--number-of-animals", type=int, required=True)
    ap.add_argument("--n-jobs", type=int, default=None, required=False)
    ap.add_argument("--chunks", nargs="+", type=int, required=True)
    return ap


def extend_index(indices, animals):

    ZT0_HOUR=11

    # add local identity and file to index
    columns=["chunk", "local_identity"]
    for animal_id, animal in enumerate(animals):
        experiment, identity = animal.split("__")
        identity = int(identity)   
        tokens = experiment.split("_")
        basedir = os.environ["FLYHOSTEL_VIDEOS"] + "/"+ "/".join([tokens[0], tokens[1], "_".join(tokens[2:4])])
        flyhostel_db = glob.glob(f"{basedir}/FlyHostel*.db")[0]
    
        with sqlite3.connect(flyhostel_db) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT {','.join(columns)} FROM CONCATENATION where identity = ?", (identity, ))
            rows=cur.fetchall()

            cur.execute(f"SELECT value FROM METADATA WHERE field = 'date_time';")
            start_time = int(float(cur.fetchall()[0][0]))

            cur.execute(f"SELECT value FROM METADATA WHERE field = 'framerate';")
            framerate = float(cur.fetchall()[0][0])


            dt = datetime.datetime.fromtimestamp(start_time)
            hour_start  = dt.hour + dt.minute / 60 + dt.second / 3600
            offset = hour_start - ZT0_HOUR

        
        indices[animal_id]["frame_time"] = indices[animal_id]["frame_number"] / framerate
        indices[animal_id]["zt"] = (indices[animal_id]["frame_time"] / 3600) + offset

        if pd.isna(indices[animal_id]["zt"]).any():
            import ipdb; ipdb.set_trace()
            
        concatenation_index = pd.DataFrame.from_records(rows, columns=columns)
        indices[animal_id]=pd.merge(indices[animal_id], concatenation_index, on=["chunk"])
        indices[animal_id]["identity"]=identity
        prediction_files = get_video_files(experiment, animal_id)
        prediction_files=pd.DataFrame.from_records(
            [(int(os.path.basename(file).split(".")[0]), file) for file in prediction_files],
            columns=["chunk", "file"]
        )
        
        nrows=indices[animal_id].shape[0]
        indices[animal_id]=pd.merge(indices[animal_id], prediction_files, on=["chunk"], how="left")
        missing = indices[animal_id].loc[
            pd.isna(indices[animal_id]["file"])
        ]
        if missing.shape[0] != 0:
            raise Exception(f"Prediction for identity {identity} chunks {np.unique(missing['chunk'].values.tolist())} is missing")
        assert nrows==indices[animal_id].shape[0]

    return indices



def main():

    ap = get_parser()
    args = ap.parse_args()

    experiment = args.experiment
    number_of_animals = args.number_of_animals
    chunks = np.array(args.chunks)

    experiment_prefix=experiment.replace('/','_')


    frame_numbers_from_first_chunk=slice(
        chunksize*(chunks[0]-first_chunk),
        chunksize*len(chunks)+chunksize*(chunks[0]-first_chunk),
    )
    frame_numbers=np.array(range(
        chunksize*(chunks[0]),
        chunksize*(chunks[-1]+1),
    ))
    bsoid_individual_step=bsoid_chunksize*len(chunks) - 1

                    
    data_directories=os.listdir(root_path)
    data_directories=[directory for directory in data_directories if directory.startswith(experiment.replace("/","_"))]
    data_directories=sorted(data_directories)
    pattern=os.path.join(os.environ["BSOID_DATA"],  data_directories[0], "*h5")

    animals=data_directories
    assert len(animals) == number_of_animals

    with h5py.File(glob.glob(pattern)[0], "r") as file:
        node_names=[e.decode() for e in file["node_names"][:]]

    if body_parts_chosen is None:
        pose_chosen=np.arange(len(node_names))
    else:
        pose_chosen=[node_names.index(node) for node in body_parts_chosen]
    
    params = {
        "body_parts_chosen": body_parts_chosen,
        "skeleton": skeleton,
        "score_filter": score_filter,
        "labels": labels,
        "criteria": criteria,
        "thorax_pos": thorax_pos,
        "chunksize": chunksize,
        "framerate": framerate,
        "bsoid_framerate": bsoid_framerate,
        "chunks": chunks,
        "frame_numbers": frame_numbers,
        "animals": animals,
        "first_chunk": first_chunk,
        
    }
    if args.n_jobs is None:
        n_jobs=number_of_animals
    else:
        n_jobs=args.n_jobs

    preprocessor=data_preprocess.preprocess(prefix=prefix, software="SLEAP", ftype="h5", pose_chosen=pose_chosen, subfolders=data_directories)
    _, _, processed_input_data, _ = preprocessor.compile_data(
        n_jobs=n_jobs, score_filter=score_filter, criteria=criteria,
        frame_numbers=frame_numbers_from_first_chunk
    )

    # interpolate frames where the pose estimation software did not run by copying the last observed value
    # this is needed for body parts where the criteria is not interpolate
    # for body parts with interpolate as criteria, this step is already done
    processed_input_data_filled=interpolate_between_contiguous_predictions(processed_input_data, body_parts_chosen, ["proboscis"], stride)
    mydata=filter_data(processed_input_data_filled, [i for i in range(len(animals))], chunks, chunksize, first_chunk=chunks[0])
    

    # sqlite3_file = f"{os.environ['FLYHOSTEL_RESULTS']}/6ac15e0aedea4a08aa1c964a01da3a4e/{experiment}/{experiment_prefix}.db"
    # assert os.path.exists(sqlite3_file), f"{sqlite3_file} does not exist"  
    centroid_data=compute_speed(animals, chunks, centroid_chunksize, centroid_framerate, framerate)
    
    col_types=["x", "y", "likelihood", "is_interpolated"]
    h5s_pandas=[]
    for animal_id, animal in enumerate(animals):
        print(animal_id)
        score=preprocessor.scores[animal_id]
        interpolated=preprocessor.interpolated[animal_id]
        positions=mydata[animal_id]

        speed=centroid_data.loc[centroid_data["animal"]==animal]["speed"].values
        assert framerate * speed.shape[0] / centroid_framerate == positions.shape[0], f"{framerate * speed.shape[0] / centroid_framerate} != {positions.shape[0]}"
        assert framerate * speed.shape[0] / centroid_framerate == score.shape[1], f"{framerate * speed.shape[0] / centroid_framerate} != {score.shape[1]}"

        all_columns = []
        for i, bp in enumerate(body_parts_chosen):
            #x
            all_columns.append(positions[:, (i*2)])
            #y
            all_columns.append(positions[:, (i*2+1)])
            #score
            all_columns.append(score[i,:])
            #interpolated
            all_columns.append(interpolated[i,:])
            

        data={}
        for col_id in range(len(all_columns)):
            col_type = col_types[col_id % 4]
            body_part = body_parts_chosen[col_id // 4]    
            col_key = ("SLEAP", body_part, col_type)
            data[col_key]=all_columns[col_id]
        
        df = pd.DataFrame(data)
        df.set_index(frame_numbers, inplace=True)
        df.index.set_names("frame_number", inplace=True)
        df = df.rename_axis(["scorer", "bodyparts", "coords"], axis=1)
        h5s_pandas.append(df)

    index=pd.DataFrame({"frame_number": frame_numbers, "chunk": frame_numbers//chunksize, "frame_idx": frame_numbers % chunksize})
  
    # downsample to the bsoid_framerate, which is also the sleap framerate
    keep_frames=slice(0, index.shape[0], framerate//bsoid_framerate)

    indices=[]
    for animal in animals:
        indices.append(
            index.iloc[keep_frames, :].copy()
        )
    
    del i
    indices = extend_index(indices, animals)

    for animal_id, animal in enumerate(animals):
        df=h5s_pandas[animal_id].iloc[keep_frames, :]
        df_index=indices[animal_id]

        df.to_hdf(f"{DATASETS}/{animal}_positions.h5", key="pose")
        df_index.to_hdf(f"{DATASETS}/{animal}_positions.h5", key="index")
        # index.iloc[keep_frames, :].to_hdf(f"{DATASETS}/{animal}_positions.h5", key="index")

    with open(f"{DATASETS}/{experiment_prefix}.pkl", "wb") as handle:
        pickle.dump(params, handle)


if __name__ == "__main__":
    main()
