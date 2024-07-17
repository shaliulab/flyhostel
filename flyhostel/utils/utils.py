import time
import json
import logging
import os.path
import yaml
import re
import shutil
import glob
import sys
import sqlite3
import joblib
from confapp import conf
import numpy as np
import pandas as pd


logger=logging.getLogger(__name__)

if sys.version_info > (3, 8):
    import pickle
else:
    try:
        logger.warning("Python version < 3.8 detected. Loading pickle5 instead of pickle")
        import pickle5 as pickle
    except ModuleNotFoundError as error:
        logger.warning(error)


from flyhostel.constants import CONFIG_FILE, DEFAULT_CONFIG, ANALYSIS_FOLDER
from flyhostel.quantification.constants import TRAJECTORIES_SOURCE
logger = logging.getLogger(__name__)

def get_dbfile(basedir):
    dbfile=os.path.join(
        basedir,
        "_".join(basedir.rstrip(os.path.sep).split(os.path.sep)[-3:]) + ".db"
    )
    assert os.path.exists(dbfile), f"{dbfile} not found"
    return dbfile

def get_basedir(experiment):
    tokens = experiment.split("_")
    basedir=f"/flyhostel_data/videos/{tokens[0]}/{tokens[1]}/{'_'.join(tokens[2:4])}"
    return basedir

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


def copy_files_to_store(imgstore_folder, files, overwrite=False, n_jobs=None):

    trajectories_source_path = os.path.join(imgstore_folder, f"{TRAJECTORIES_SOURCE}.pkl")

    trajectories_source={}

    if os.path.exists(trajectories_source_path):
        with open(trajectories_source_path, "rb") as filehandle:
            trajectories_source.update(pickle.load(filehandle))


    if n_jobs is None:
        n_jobs = conf.NUMBER_OF_JOBS_FOR_COPYING_TRAJECTORIES

    output = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(copy_file_to_store)(
            file, imgstore_folder, overwrite, trajectories_source_path
        ) for file in files
    )

    trajectories_source.update({k: v for k, v in output})

    with open(trajectories_source_path, "wb") as filehandle:
        pickle.dump(trajectories_source, filehandle)

    with open(trajectories_source_path.replace(".pkl", ".yml"), "w") as filehandle:
        yaml.dump(trajectories_source, filehandle)


def copy_file_to_store(file, imgstore_folder, overwrite, trajectories_source_path):

    # NOTE
    # some_folder/session_N/trajectories/trajectories.npy
    session = file.split("/")[::-1][2]
    chunk = int(session.replace("session_", ""))

    dest_filename = str(chunk).zfill(6) + ".npy"
    dest_path = os.path.join(imgstore_folder, dest_filename)

    file_exists = os.path.exists(dest_path)

    if not file_exists:
        clean_copy(file, dest_path)
    elif os.path.getmtime(dest_path) >= os.path.getmtime(file):
        logger.debug(f"{file} is updated")
    else:
        if overwrite:
            clean_copy(file, dest_path)
        else:
            logger.debug(f"{dest_path} exists. Not overwriting")
        
    return file, os.path.basename(dest_path)



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

    data=np.load(file, allow_pickle=True).item()._trajectories

    if "chunk" in data:
        chunk = data["chunk"]
    else:
        logger.warning(f"Trajectories file {file} does not carry the source chunk")
        chunk = find_chunk_from_filename(file)
        data["chunk"] = chunk

    #logger.info(f"Copying {file} --> {dest_path}")
    print(f"Copying {file} --> {dest_path}")
    np.save(dest_path, data, allow_pickle=True)



def get_sqlite_file(animal):

    tokens = animal.split("_")[:4]
    sqlite_files = glob.glob(f"{os.environ['FLYHOSTEL_VIDEOS']}/{tokens[0]}/{tokens[1]}/{tokens[2]}_{tokens[3]}/{'_'.join(tokens)}.db")
    assert len(sqlite_files) == 1
    sqlite_file=sqlite_files[0]

    assert os.path.exists(sqlite_file)
    return sqlite_file

def load_metadata_prop(prop, animal=None, dbfile=None):

    if dbfile is None:
        dbfile = get_sqlite_file(animal)

    try:
        with sqlite3.connect(dbfile) as connection:
            cursor = connection.cursor()
            cursor.execute(f"SELECT value FROM METADATA WHERE field = '{prop}';")
            prop = cursor.fetchone()[0]
    except sqlite3.OperationalError as error:
        logger.error("%s METADATA table cannot be read". dbfile)
        raise error
    else:
        return prop

def load_roi_width(dbfile):
    try:
        with sqlite3.connect(dbfile) as conn:
            cursor=conn.cursor()

            cursor.execute(
                """
            SELECT w FROM ROI_MAP;
            """
            )
            [(roi_width,)] = cursor.fetchall()
            cursor.execute(
                """
            SELECT h FROM ROI_MAP;
            """
            )
            [(roi_height,)] = cursor.fetchall()

        roi_width=int(roi_width)
        roi_height=int(roi_height)
        roi_width=max(roi_width, roi_height)
    except sqlite3.OperationalError as error:
        logger.error("%s ROI_MAP table cannot be read", dbfile)
        raise error
    
    else:
        return roi_width

def parse_identity(id):
    return int(id.split("|")[1])


def get_local_identities_from_experiment(experiment, frame_number):

    tokens = experiment.split("_")
    experiment_path=os.path.sep.join([tokens[0], tokens[1], "_".join(tokens[2:4])])
    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], experiment_path)
    if not os.path.exists(basedir):
        basedirs=glob.glob(basedir+"*")
        assert len(basedirs) == 1, f"{basedir} not found"
        basedir=basedirs[0]
        experiment = "_".join(basedir.split(os.path.sep))


    dbfile = os.path.join(basedir, experiment + ".db")
    table=get_local_identities(dbfile, [frame_number])
    return table


def get_local_identities_v1(dbfile, frame_numbers, identity_table="IDENTITY"):

    with sqlite3.connect(dbfile) as conn:
        cursor = conn.cursor()
        query = "SELECT frame_number, identity, local_identity FROM {} WHERE frame_number IN ({})".format(
            identity_table,
            ','.join(['?'] * len(frame_numbers))
        )
        cursor.execute(query, frame_numbers)
        
        table = cursor.fetchall()
    
    table=pd.DataFrame.from_records(table, columns=["frame_number", "identity", "local_identity"])
    return table

def get_local_identities_v2(dbfile, frame_numbers, identity_table=None):
    chunksize=get_chunksize(dbfile)
    chunks=(np.array(frame_numbers)//chunksize).tolist()

    with sqlite3.connect(dbfile) as conn:
        cursor = conn.cursor()
        query = "SELECT * FROM CONCATENATION_VAL;"
        cursor.execute(query)
        table = cursor.fetchall()
    
    table=pd.DataFrame.from_records(table, columns=["id", "chunk", "identity", "local_identity"])
    table["local_identity"]=table["local_identity"].astype(np.int32)
    table["identity"]=table["identity"].astype(np.int32)
    
    table=table.loc[table["chunk"].isin(chunks)]
    table["frame_number"]=table["chunk"]*chunksize

    return table


def get_local_identities(dbfile, *args, **kwargs):
    try:
        return get_local_identities_v2(dbfile, *args, **kwargs)
    except sqlite3.OperationalError as error:
        logger.debug("%s is not human validated", dbfile)
        logger.debug(error)
        return get_local_identities_v1(dbfile, *args, **kwargs)



def get_chunksize(dbfile):
    with sqlite3.connect(dbfile) as conn:
        cursor=conn.cursor()
        cursor.execute(f"SELECT value FROM METADATA WHERE field = 'chunksize';",)
        chunksize = int(float(cursor.fetchone()[0]))
    return chunksize

def get_single_animal_video(dbfile, frame_number, table, identity, chunksize):
    chunk = frame_number // chunksize
    table_current_frame = table.loc[(table["frame_number"] == frame_number)]

    local_identity = table_current_frame.loc[table_current_frame["identity"] == identity, "local_identity"]
    if local_identity.shape[0] == 0:
        single_animal_video=None
    else:
        local_identity=local_identity.item()
        single_animal_video = os.path.join(os.path.dirname(dbfile), "flyhostel", "single_animal", str(local_identity).zfill(3), str(chunk).zfill(6) + ".mp4")
    
    if single_animal_video is None:
        logger.warning("identity %s not found in chunk %s", identity, chunk)
    
    return single_animal_video


def restore_cache(path):
    if os.path.exists(path):
        logger.debug("Loading ---> %s", path)
        before=time.time()
        try:
            with open(path, "rb") as handle:
                out=pickle.load(handle)
        except:
            return False, None
        after=time.time()
        logger.debug("Loading %s took %s seconds", path, after-before)
        

        return True, out
    else:
        return False, None

def save_cache(path, data):
    logger.debug(f"Caching ---> {path}")
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=4)


def annotate_time_in_dataset(dataset, index, t_column="t", t_after_ref=None):
    before=time.time()
    assert index is not None
    assert "frame_number" in dataset.columns

    if t_column in dataset.columns:
        dataset_without_t=dataset.drop(t_column, axis=1)
    else:
        dataset_without_t=dataset
    dataset=dataset_without_t.merge(index[["frame_number", t_column]], on=["frame_number"])
    if t_after_ref is not None and t_column == "frame_time":
        dataset["t"]=dataset[t_column]+t_after_ref
    after=time.time()
    logger.debug("annotate_time_in_dataset took %s seconds", after-before)
    return dataset
