import json
import logging
import os.path
import yaml
import pickle
import shutil

from tqdm import tqdm

from flyhostel.constants import CONFIG_FILE, DEFAULT_CONFIG
from flyhostel.quantification.constants import TRAJECTORIES_SOURCE

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
            shutil.copy(file, dest_path)
            trajectories_source[file]=os.path.basename(dest_path)

    with open(trajectories_source_path, "wb") as filehandle:
        pickle.dump(trajectories_source, filehandle)

    with open(trajectories_source_path.replace(".pkl", ".yml"), "w") as filehandle:
        yaml.dump(trajectories_source, filehandle)

