import logging
import itertools
import os.path
import re
import glob
import sqlite3
import numpy as np
import yaml
from flyhostel.constants import INDEX_FORMAT
from imgstore.constants import STORE_MD_KEY, STORE_MD_FILENAME

logger = logging.getLogger(__name__)



def get_chunk_metadata(chunk_filename, source="sqlite"):
    index = {"frame_time": [], "frame_number": []}
    if source=="npz":
        data = np.load(chunk_filename)
        index["frame_time"] = data["frame_time"]
        index["frame_number"] = data["frame_number"]
    if source=="sqlite":
        sqlite_file = os.path.join(os.path.dirname(chunk_filename), "index.db")
        chunk = int(re.search(".*/([0-9][0-9][0-9][0-9][0-9][0-9]).npz*", chunk_filename).group(1))
        with sqlite3.connect(sqlite_file) as conn:
            cur = conn.cursor()
            cur.execute(f"SELECT frame_time, frame_number FROM frames WHERE chunk = {chunk};")
            fetch = cur.fetchall()
            for row in fetch:
                index["frame_time"].append(row[0])
                index["frame_number"].append(row[1])

    return index

def read_store_metadata(imgstore_folder):
    metadata_filename = os.path.join(imgstore_folder, STORE_MD_FILENAME)
    if os.path.exists(metadata_filename):
        with open(metadata_filename, "r") as filehandle:
            store_metadata = yaml.load(filehandle, Loader=yaml.SafeLoader)["__store"]
    
    else:
        raise Exception(f"{imgstore_folder} does not contain a {STORE_MD_FILENAME} file. Are you sure you sure you are in the right folder?")
        
    return store_metadata


def read_store_description(imgstore_folder, chunk_numbers=None):

    if chunk_numbers is None:
        index_files = sorted(
            glob.glob(
                os.path.join(
                    imgstore_folder,
                    f"*{INDEX_FORMAT}"
                )
            )
        )
        chunks = [
            int(os.path.basename(e.replace(INDEX_FORMAT, "")))
            for e in index_files
        ]   
    else:
        chunks = chunk_numbers
        index_files = [
            os.path.join(
                imgstore_folder,
                f"{str(chunk_index).zfill(6)}{INDEX_FORMAT}"
            )
            for chunk_index in chunks
        ]

    chunk_metadata = {
        chunk: get_chunk_metadata(chunk) for chunk in index_files
    }

    frame_number = list(
        itertools.chain(*[m["frame_number"] for m in chunk_metadata.values()])
    )
    frame_time = list(
        itertools.chain(*[m["frame_time"] for m in chunk_metadata.values()])
    )
    chunk_metadata = (frame_number, frame_time)
    return chunks, chunk_metadata
