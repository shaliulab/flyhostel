import tempfile
import logging
import os.path
import datetime
import sqlite3
import glob
import warnings
from abc import ABC

import numpy as np
import pandas as pd

from .constants import (
    METADATA_FILE,
    RAISE_EXCEPTION_IF_METADATA_NOT_FOUND,

)
from .utils import parse_experiment_properties
from flyhostel.utils.experiment import get_experiment_identifier, parse_experiment_name
from imgstore.constants import STORE_MD_FILENAME
from imgstore.stores.utils.mixins.extract import _extract_store_metadata

logger=logging.getLogger(__name__)

def metadata_not_found(message):

    if RAISE_EXCEPTION_IF_METADATA_NOT_FOUND:
        raise FileNotFoundError(message)
    else:
        warnings.warn(message)

class MetadataExporter(ABC):
    """Generate the METADATA table of a FlyHostel SQLite dataset
    """

    _basedir = None
    data_framerate = None

    def __init__(self, *args, **kwargs):
        self._index_dbfile = os.path.join(self._basedir, "index.db")
        self._store_metadata_path = os.path.join(self._basedir, STORE_MD_FILENAME)
        self._store_metadata = _extract_store_metadata(self._store_metadata_path)
        
        # this information should be read from index_information table in index.db
        (idtrackerai_conf_path, self._idtrackerai_conf), (self._flyhostel_id, self._number_of_animals, self._date_time) = parse_experiment_properties(basedir=self._basedir)
        # resolve the metadata file itself (not just basedir) so that it works
        # when metadata.yaml is a symlink staged in a nextflow work directory
        self._experiment = parse_experiment_name(get_experiment_identifier(os.path.dirname(os.path.realpath(self._store_metadata_path))))
        self._arena = self._experiment["arena"]
        # with open(idtrackerai_conf_path, "r", encoding="utf8") as filehandle:
        #     self._idtrackerai_conf_str = filehandle.read()

        import json
        self._idtrackerai_conf_str = json.dumps(self._idtrackerai_conf)

        matches = glob.glob(os.path.join(self._basedir, "*pfs"))
        if matches:
            self._camera_metadata_path = matches[0]
        else:
            metadata_not_found("Camera metadata (.pfs file) not found")
            self._camera_metadata_path = None

        super(MetadataExporter, self).__init__(*args, **kwargs)


    @property
    def framerate(self):
        return self._store_metadata["framerate"]


    def init_metadata_table(self, dbfile, reset=True):
        """Initialize the METADATA table
        """
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute("DROP TABLE IF EXISTS METADATA;")
            cur.execute("CREATE TABLE IF NOT EXISTS METADATA (field char(100), value varchar(4000));")

    def write_metadata_table(self, dbfile):
        """Populate the METADATA table
        """

        machine_id = "0" * 32
        machine_name = self._experiment["flyhostel"]

        created_utc=self._store_metadata["created_utc"].split(".")[0]
        dt = datetime.datetime.strptime(created_utc, "%Y-%m-%dT%H:%M:%S")
        timestamp = dt.timestamp()

        ### NOTE:
        ## Make sure the saved timestamp is relative to the UTC timezone
        ## and not the TZ of the computer where this is running
        timestamp += dt.astimezone().tzinfo.utcoffset(dt).seconds
        ####################################################


        if self._camera_metadata_path is not None and os.path.exists(self._camera_metadata_path):
            with open(self._camera_metadata_path, "r", encoding="utf8") as filehandle:
                camera_metadata_str = filehandle.read()
        else:
            camera_metadata_str=""

        ethoscope_metadata_path = os.path.join(self._basedir, METADATA_FILE)

        if os.path.exists(ethoscope_metadata_path) or self.download_metadata(ethoscope_metadata_path) == 0:
            with open(ethoscope_metadata_path, "r", encoding="utf8") as filehandle:
                ethoscope_metadata_str = filehandle.read()

        else:
            ethoscope_metadata_str = ""


        try:
            pixels_per_cm = self._store_metadata["pixels_per_cm"]
        except KeyError as exc:
            raise ValueError(
                f"Please enter the pixels_per_cm parameter in {self._store_metadata_path}"
            ) from exc

        with sqlite3.connect(self._index_dbfile, check_same_thread=False) as index_db:
            cur=index_db.cursor()
            cur.execute("SELECT chunk FROM frames ORDER BY chunk ASC LIMIT 1;")
            first_chunk = int(cur.fetchone()[0])
            cur.execute("SELECT chunk FROM frames ORDER BY chunk DESC LIMIT 1;")
            last_chunk = int(cur.fetchone()[0])
            cur.execute("SELECT frame_time FROM frames WHERE frame_number = 0;")
            first_time = int(cur.fetchone()[0] / 1000)

        chunks = f"{first_chunk},{last_chunk}"


        data = [
            ("machine_id", machine_id),
            ("machine_name", machine_name),
            ("date_time", timestamp),
            ("frame_width", self._store_metadata["imgshape"][1]),
            ("frame_height", self._store_metadata["imgshape"][0]),
            ("framerate", self.framerate),
            ("data_framerate", self.data_framerate),
            ("chunksize", self._store_metadata["chunksize"]),
            ("pixels_per_cm", pixels_per_cm),
            ("version", "1"),
            ("ethoscope_metadata", ethoscope_metadata_str),
            ("camera_metadata", camera_metadata_str),
            ("idtrackerai_conf", self._idtrackerai_conf_str),
            ("chunks", chunks),
            ("first_time", first_time),
        ]

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            conn.executemany(
                "INSERT INTO METADATA (field, value) VALUES (?, ?);",
                data
            )


    def download_metadata(self, path):
        """
        Download the metadata from a Google Sheets database

        Returns 0 if the metadata was saved to path, 1 otherwise
        """
        from googleapi.download.flyhostel import fetch_flyhostel_metadata

        path=path.replace(" ", "_")
        print(f"Downloading metadata to {path}")
        try:
            fetch_flyhostel_metadata(
                self._date_time, self._flyhostel_id,
                output=path, arena=self._arena
            )
        except Exception as error:
            logger.error("Could not download metadata to %s: %s", path, error)
            return 1

        if not os.path.exists(path):
            return 1
        return 0

    def update_ethoscope_metadata(self, dbfile, columns=[]):
        metadata_file=tempfile.NamedTemporaryFile(suffix=".csv", prefix=self._basedir.replace(os.path.sep, "_"))
        self.download_metadata(metadata_file.name)

        if os.path.exists(metadata_file.name):
            with open(metadata_file.name, "r", encoding="utf8") as filehandle:
                ethoscope_metadata_str = filehandle.read()
        
            df=pd.read_csv(metadata_file, index_col=0)

            if columns:
                df = df[columns]
            print(df)
        
            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                conn.execute(
                    "DELETE FROM METADATA WHERE field = 'ethoscope_metadata';" 
                )
            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                conn.execute(
                    "INSERT INTO METADATA (field, value) VALUES (?, ?);", ("ethoscope_metadata", ethoscope_metadata_str)
                )