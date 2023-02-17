import os.path
import datetime
import subprocess
import shlex
import sqlite3
import glob
import warnings
from abc import ABC

import yaml

from .constants import (
    METADATA_FILE,
    DOWNLOAD_FLYHOSTEL_METADATA,
    RAISE_EXCEPTION_IF_METADATA_NOT_FOUND,

)
from imgstore.constants import STORE_MD_FILENAME
from imgstore.stores.utils.mixins.extract import _extract_store_metadata


def metadata_not_found(message):

    if RAISE_EXCEPTION_IF_METADATA_NOT_FOUND:
        raise FileNotFoundError(message)
    else:
        warnings.warn(message)

class MetadataExporter(ABC):
    """Generate the METADATA table of a FlyHostel SQLite dataset
    """

    _basedir = None

    def __init__(self, *args, **kwargs):
        self._index_dbfile = os.path.join(self._basedir, "index.db")
        self._store_metadata_path = os.path.join(self._basedir, STORE_MD_FILENAME)
        self._store_metadata = _extract_store_metadata(self._store_metadata_path)

        self._idtrackerai_conf_path = os.path.join(
            self._basedir,
            f"{os.path.basename(self._basedir)}.conf"
        )

        with open(self._idtrackerai_conf_path, "r", encoding="utf8") as filehandle:
            self._idtrackerai_conf = yaml.load(filehandle, yaml.SafeLoader)

        matches = glob.glob(os.path.join(self._basedir, "*pfs"))
        if matches:
            self._camera_metadata_path = matches[0]
        else:
            metadata_not_found("Camera metadata (.pfs file) not found")
            self._camera_metadata_path = None

        super(MetadataExporter, self).__init__(*args, **kwargs)


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
        machine_name = os.path.basename(os.path.dirname(os.path.dirname(self._basedir)))

        created_utc=self._store_metadata["created_utc"].split(".")[0]
        date_time = datetime.datetime.strptime(created_utc, "%Y-%m-%dT%H:%M:%S").timestamp()


        with open(self._idtrackerai_conf_path, "r", encoding="utf8") as filehandle:
            idtrackerai_conf_str = filehandle.read()


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
        except KeyError:
            raise ValueError(
                f"Please enter the pixels_per_cm parameter in {self._store_metadata_path}"
            )

        with sqlite3.connect(self._index_dbfile, check_same_thread=False) as index_db:
            cur=index_db.cursor()
            cur.execute("SELECT chunk FROM frames ORDER BY chunk ASC LIMIT 1;")
            first_chunk = int(cur.fetchone()[0])
            cur.execute("SELECT chunk FROM frames ORDER BY chunk DESC LIMIT 1;")
            last_chunk = int(cur.fetchone()[0])
        chunks = f"{first_chunk},{last_chunk}"


        values = [
            ("machine_id", machine_id),
            ("machine_name", machine_name),
            ("date_time", date_time),
            ("frame_width", self._store_metadata["imgshape"][1]),
            ("frame_height", self._store_metadata["imgshape"][0]),
            ("framerate", self._store_metadata["framerate"]),
            ("chunksize", self._store_metadata["chunksize"]),
            ("pixels_per_cm", pixels_per_cm),
            ("version", "1"),
            ("ethoscope_metadata", ethoscope_metadata_str),
            ("camera_metadata", camera_metadata_str),
            ("idtrackerai_conf", idtrackerai_conf_str),
            ("chunks", chunks),
        ]

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            for val in values:

                cur.execute(
                    "INSERT INTO METADATA (field, value) VALUES (?, ?);",
                    val
                )


    @staticmethod
    def download_metadata(path):
        """
        Download the metadata from a Google Sheets database
        """

        # try:
        if DOWNLOAD_FLYHOSTEL_METADATA is None:
            raise ModuleNotFoundError("Please define DOWNLOAD_FLYHOSTEL_METADATA     as the path to the download-behavioral-data Python binary")

        path=path.replace(" ", "_")
        cmd = f'{DOWNLOAD_FLYHOSTEL_METADATA} --metadata {path}'
        cmd_list = shlex.split(cmd)
        process = subprocess.Popen(cmd_list)
        process.communicate()
        print(f"Downloading metadata to {path}")
        return 0
        # except:
        #     metadata_not_found(f"Could not download metadata to {path}")
        #     return 1
