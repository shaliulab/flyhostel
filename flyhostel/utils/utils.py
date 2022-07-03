import json
import logging
import os.path
import yaml
import re
import pickle
import shutil
import sqlite3

from tqdm import tqdm
import numpy as np

from flyhostel.constants import CONFIG_FILE, DEFAULT_CONFIG
from flyhostel.quantification.constants import TRAJECTORIES_SOURCE
ANALYSIS_FOLDER="idtrackerai"
logger = logging.getLogger(__name__)

def add_suffix(filename, suffix=""):

    if suffix != "":
        basename, ext = os.path.splitext(filename)
        filename = basename + "_" + suffix + ext
    
    return filename


def load_config(path=CONFIG_FILE):

    if not os.path.exists(path):
        config = DEFAULT_CONFIG
        os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
        os.makedirs(config["videos"]["folder"], exist_ok=True)

        try:
            with open(path, "w") as fh:
                json.dump(config, fh)
                logger.warning(
                    f"Saving configuration below to {path}"\
                    f" {config}"
                )
        except Exception as error:
            logger.error(
                "Cannot save configuration to {path}"\
                " Please make sure the file exists and it's writable"
            )
    
    else:
        with open(path, "r") as fh:
            config = json.load(fh)
        
    return config


def copy_files_to_store(imgstore_folder, files, overwrite=False):

    trajectories_source_path = os.path.join(imgstore_folder, f"{TRAJECTORIES_SOURCE}.pkl")
    trajectories_source={}

    if os.path.exists(trajectories_source_path):
        with open(trajectories_source_path, "rb") as filehandle:
            trajectories_source.update(pickle.load(filehandle))

    for file in tqdm(files):

        # NOTE
        # some_folder/session_N/trajectories/trajectories.npy
        session = file.split("/")[::-1][2]
        chunk = int(session.replace("session_", ""))
    
        dest_filename = str(chunk).zfill(6) + ".npy"
        dest_path = os.path.join(imgstore_folder, dest_filename)

        file_exists = os.path.exists(dest_path)

        if file_exists and not overwrite:
            logger.warning("f{file} exists. Not overwriting")
        else:
            clean_copy(file, dest_path)
            trajectories_source[file]=os.path.basename(dest_path)

    with open(trajectories_source_path, "wb") as filehandle:
        pickle.dump(trajectories_source, filehandle)

    with open(trajectories_source_path.replace(".pkl", ".yml"), "w") as filehandle:
        yaml.dump(trajectories_source, filehandle)



def raw_copy(file, dest_path):
    shutil.copy(file, dest_path)

def find_chunk_from_filename(file):

    match=int(re.search("session_(\d{6})", file).group(1))
    return match

def find_start_and_end_of_chunk(session_folder, chunk):

    with sqlite3.connect(os.path.join(session_folder, "..", "..", "index.db")) as con:
            cur = con.cursor()
            cur.execute(f"SELECT frame_number FROM frames WHERE chunk={chunk};")
            start= cur.fetchone()[0]
            end = cur.fetchall()[-1][0]

    return start, end



def clean_copy(file, dest_path):

    data=np.load(file, allow_pickle=True).item()
    

    if "chunk" in data:
        chunk = data["chunk"]
    else:
        logger.warning(f"Trajectories file {file} does not carry the source chunk")
        chunk = find_chunk_from_filename(file)
        data["chunk"] = chunk

    session_folder = os.path.dirname(os.path.dirname(file))
    start, end = find_start_and_end_of_chunk(session_folder, chunk)

    data["trajectories"]=data["trajectories"][start:(end)+1]
    data["id_probabilities"]=data["id_probabilities"][start:(end)+1]
    data["areas"]=data["areas"][start:(end)+1]

    #logger.info(f"Copying {file} --> {dest_path}")
    print(f"Copying {file} --> {dest_path}")
    np.save(dest_path, data, allow_pickle=True)
