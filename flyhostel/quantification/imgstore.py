import logging
import itertools
import os.path
import glob
import numpy as np
import yaml
from flyhostel.constants import INDEX_FORMAT
from imgstore.constants import STORE_MD_KEY, STORE_MD_FILENAME

logger = logging.getLogger(__name__)



def get_chunk_metadata(chunk_filename):

    data = np.load(chunk_filename)
    index = {}
    index["frame_time"] = data["frame_time"]
    index["frame_number"] = data["frame_number"]
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
