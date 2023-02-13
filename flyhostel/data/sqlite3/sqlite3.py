"""
Centralize the results obtained in the FlyHostel pipeline into a single SQLite file
that can be used to perform all downstream analyses
"""

import sqlite3
import warnings
import subprocess
import datetime
import pickle
import os.path
import tempfile
import logging
import shlex
import glob
import yaml
import json
from ast import literal_eval

import cv2
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import h5py

from imgstore.stores.utils.mixins.extract import _extract_store_metadata
from imgstore.constants import STORE_MD_FILENAME
from .async_writer import AsyncSQLiteWriter
from .constants import (
    TABLES,
    METADATA_FILE,
    RAISE_EXCEPTION_IF_METADATA_NOT_FOUND,
    DOWNLOAD_BEHAVIORAL_DATA
)
from .utils import table_is_not_empty

logger = logging.getLogger(__name__)

def metadata_not_found(message):

    if RAISE_EXCEPTION_IF_METADATA_NOT_FOUND:
        raise ModuleNotFoundError(message)
    else:
        warnings.warn(message)


class SQLiteExporter:

    _CLASSES = {0: "head"}
    _AsyncWriter = AsyncSQLiteWriter


    def __init__(self, basedir, n_jobs=1):
        self._basedir=basedir

        self._store_metadata_path = os.path.join(self._basedir, STORE_MD_FILENAME)
        self._store_metadata = _extract_store_metadata(self._store_metadata_path)

        self._idtrackerai_conf_path = os.path.join(self._basedir, f"{os.path.basename(self._basedir)}.conf")
        with open(self._idtrackerai_conf_path, "r", encoding="utf8") as filehandle:
            self._idtrackerai_conf = yaml.load(filehandle, yaml.SafeLoader)

        matches = glob.glob(os.path.join(self._basedir, "*pfs"))
        if matches:
            self._camera_metadata_path = matches[0]
        else:
            metadata_not_found("Camera metadata (.pfs file) not found")
            self._camera_metadata_path = None


        self._temp_path = tempfile.mktemp(prefix="flyhostel_", suffix=".jpg")
        self._number_of_animals = None
        self._index_dbfile = os.path.join(self._basedir, "index.db")
        self._n_jobs=n_jobs
        self._writers = {}



    @staticmethod
    def download_metadata(path):
        """
        Download the metadata from a Google Sheets database
        """

        # try:
        if DOWNLOAD_BEHAVIORAL_DATA is None:
            raise Exception("Please define DOWNLOAD_BEHAVIORAL_DATA as the path to the download-behavioral-data Python binary")

        path=path.replace(" ", "_")
        cmd = f'{DOWNLOAD_BEHAVIORAL_DATA} --metadata {path}'
        cmd_list = shlex.split(cmd)
        process = subprocess.Popen(cmd_list)
        process.communicate()
        print(f"Downloading metadata to {path}")
        return 0
        # except:
        #     metadata_not_found(f"Could not download metadata to {path}")
        #     return 1


    def export(self, dbfile, chunks, tables=None, mode=["w", "a"], reset=False):

        if tables is None or tables == "all":
            tables = TABLES

        if os.path.exists(dbfile):
            if reset and False:
                warnings.warn(f"{dbfile} exists. Remaking from scratch and ignoring mode")
                os.remove(dbfile)
            elif mode == "w":
                warnings.warn(f"{dbfile} exists. Overwriting (mode=w)")
            elif mode == "a":
                warnings.warn(f"{dbfile} exists. Appending (mode=a)")

        print(f"Initializing file {dbfile}")
        self.init_tables(dbfile, tables)
        print(f"Writing tables: {tables}")

        if "CONCATENATION" in tables and not table_is_not_empty(dbfile, "CONCATENATION"):
            self.write_concatenation_table(dbfile)

        if "METADATA" in tables:
            self.write_metadata_table(dbfile)

        if "IMG_SNAPSHOTS" in tables:
            self.write_snapshot_table(dbfile, chunks=chunks)

        if "ROI_MAP" in tables:
            self.write_roi_map_table(dbfile)

        if "ENVIRONMENT" in tables:
            self.write_environment_table(dbfile, chunks=chunks)

        if "VAR_MAP" in tables:
            self.write_var_map_table(dbfile)

        if "STORE_INDEX" in tables:
            self.write_index_table(dbfile)

        if "AI" in tables:
            self.write_ai_table(dbfile)


    @property
    def number_of_animals(self):
        if self._number_of_animals is None:
            self._number_of_animals = int(self._idtrackerai_conf["_number_of_animals"]["value"])
        return self._number_of_animals


    def init_tables(self, dbfile, tables):

        #TODO
        # Add if for all tables

        if "METADATA" in tables:
            self.init_metadata_table(dbfile)

        self.init_snapshot_table(dbfile)
        self.init_roi_map_table(dbfile)
        self.init_environment_table(dbfile)
        self.init_var_map_table(dbfile)
        self.init_index_table(dbfile)

        self.init_ai_table(dbfile)
        self.init_concatenation_table(dbfile)

    def build_blobs_collection(self, chunk):
        return os.path.join(self._basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")

    def build_video_object(self, chunk):
        return os.path.join(self._basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "video_object.npy")


    # METADATA
    def init_metadata_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS METADATA (field char(100), value varchar(4000));")

    def write_metadata_table(self, dbfile):

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
            raise Exception(f"Please enter the pixels_per_cm parameter in {self._store_metadata_path}")

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

    def init_snapshot_table(self, dbfile):

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:

            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS IMG_SNAPSHOTS (frame_number int(11), img longblob)")

    def write_snapshot_table(self, dbfile, chunks):


        with sqlite3.connect(self._index_dbfile, check_same_thread=False) as index_db:
            index_cursor = index_db.cursor()

            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                cur = conn.cursor()

                for chunk in chunks:
                    index_cursor.execute(f"SELECT frame_number FROM frames WHERE chunk = {chunk} AND frame_idx = 0;")
                    frame_number = int(index_cursor.fetchone()[0])

                    snapshot_path = os.path.join(self._basedir, f"{str(chunk).zfill(6)}.png")
                    if not os.path.exists(snapshot_path):
                        raise Exception(f"Cannot save chunk {chunk} snapshot. {snapshot_path} does not exist")
                    arr=cv2.imread(snapshot_path)
                    bstring = self.serialize_arr(arr, self._temp_path)
                    cur.execute(
                        "INSERT INTO IMG_SNAPSHOTS (frame_number, img) VALUES (?, ?)",
                        (frame_number, bstring)
                    )

    # ROI_MAP
    def init_roi_map_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS ROI_MAP (roi_idx smallint(6), roi_value smallint(6), x smallint(6), y smallint(6), w smallint(6), h smallint(6), mask longblob);")

    @staticmethod
    def serialize_arr(arr, path):
        """
        Transform an image (np.array) to bytes for export in SQLite
        """

        cv2.imwrite(path, arr, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

        with open(path, "rb") as f:
            bstring = f.read()

        return bstring


    def write_roi_map_table(self, dbfile):

        roi = np.array(literal_eval(self._idtrackerai_conf["_roi"]["value"][0][0]))
        x, y, w, h = cv2.boundingRect(roi)

        mask = np.zeros(self._store_metadata["imgshape"])
        mask = cv2.drawContours(mask, [roi], -1, 255, -1)
        mask = self.serialize_arr(mask, self._temp_path)

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO ROI_MAP (roi_idx, roi_value, x, y, w, h, mask) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (0, 0, x, y, w, h, mask)
            )

    def init_environment_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS ENVIRONMENT (frame_number int(11), camera_temperature real(6), temperature real(6), humidity real(6), light real(6));")


    def write_environment_table(self, dbfile, chunks):

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()

            for chunk in chunks:

                extra_json = os.path.join(self._basedir, f"{str(chunk).zfill(6)}.extra.json")

                if not os.path.exists(extra_json):
                    warnings.warn(f"No environmental data available for chunk {chunk}")
                    return


                with open(extra_json, "r") as filehandle:
                    extra_data = json.load(filehandle)

                for row in extra_data:

                    values = (row["frame_number"], row["camera_temperature"], row["temperature"], row["humidity"], row["light"])

                    cur.execute(
                        "INSERT INTO ENVIRONMENT (frame_number, camera_temperature, temperature, humidity, light) VALUES (?, ?, ?, ?, ?);",
                        values
                    )


    # VAR_MAP
    def init_var_map_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS VAR_MAP (var_name char(100), sql_type char(100), functional_type char(100));")


    def write_var_map_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()

            values = [
                ("frame_number", "INT", "count"),
                ("in_frame_index", "INT", "count"),
                ("x", "REAL", "distance"),
                ("y", "REAL", "distance"),
                ("area", "INT", "count"),
                ("modified", "INT", "bool"),
            ]

            for val in values:
                cur.execute(
                    "INSERT INTO VAR_MAP (var_name, sql_type, functional_type) VALUES (?, ?, ?);",
                    val
                )


    def init_index_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS STORE_INDEX (chunk int(3), frame_number int(11), frame_time int(11));")


    def init_ai_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute("DROP TABLE IF EXISTS AI;")
            cur.execute("CREATE TABLE IF NOT EXISTS AI (frame_number int(11), ai int(2));")

    def init_concatenation_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute("DROP TABLE IF EXISTS CONCATENATION;")
            cur.execute("CREATE TABLE IF NOT EXISTS CONCATENATION (chunk int(3), in_frame_index int(2), in_frame_index_after int(2), local_identity int(2), local_identity_after int(2), identity int(2));")

    def write_concatenation_table(self, dbfile):
        csv_file = os.path.join(self._basedir, "idtrackerai", "concatenation-overlap.csv")

        if os.path.exists(csv_file):
            concatenation=pd.read_csv(csv_file, index_col=0)
            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                cur = conn.cursor()
                for i, row in concatenation.iterrows():
                    args = (
                        row["chunk"], row["in_frame_index_before"], row["in_frame_index_after"],
                        row["local_identity"], row["local_identity_after"], row["identity"]
                    )
                    args=tuple([e.item() for e in args])
                    cur.execute(
                        "INSERT INTO CONCATENATION (chunk, in_frame_index, in_frame_index_after, local_identity, local_identity_after, identity) VALUES (?, ?, ?, ?, ?, ?);",
                        args
                    )
        else:
            warnings.warn(f"concatenation_overlap.csv not found. Please make sure idtrackerai_concatenation step is run for {self._basedir}")

        print("CONCATENATION table done")

    def write_ai_table(self, dbfile):

        pickle_files = sorted(glob.glob(os.path.join(self._basedir, "idtrackerai", "session_*", "preprocessing", "ai.pkl")))
        if pickle_files:
            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                cur = conn.cursor()

                for file in pickle_files:
                    with open(file, "rb") as filehandle:
                        ai_mods = pickle.load(filehandle)
                        frames=ai_mods["success"]

                    for frame_number in frames:
                        cur.execute("INSERT INTO AI (frame_number, ai) VALUES (?, ?);", (frame_number, "YOLOv7"))

    def write_index_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            with sqlite3.connect(self._index_dbfile, check_same_thread=False) as index_db:
                index_db_cur = index_db.cursor()

                index_db_cur.execute("SELECT COUNT(*) FROM frames;")
                count = int(index_db_cur.fetchone()[0])

                index_db_cur.execute("SELECT chunk, frame_number, frame_time FROM frames;")
                pb=tqdm(total=count)

                for chunk, frame_number, frame_time in index_db_cur:
                    cur.execute(
                        "INSERT INTO STORE_INDEX (chunk, frame_number, frame_time) VALUES (?, ?, ?);",
                        (chunk, frame_number, frame_time)
                    )
                    pb.update(1)


    @staticmethod
    def fetch_angle_from_label_file(label_file):
        """
        Return the angle of the best detection
        """

        with open(label_file, "r", encoding="utf8") as filehandle:
            lines = filehandle.readlines()

        data = []
        for line in lines:
            line = line.strip().split(" ")
            assert len(line) == 7 # id, x, y, w, h, conf, angle
            data.append(line)

        data=sorted(data, key=lambda line: float(line[5]))[::-1]
        return data[0][-1]

    @staticmethod
    def fetch_angle_from_angle_file(angle_file, classes=None, top_angles=1):
        """
        """

        with open(angle_file, "r") as filehandle:
            lines = filehandle.readlines()

        data = []
        for line in lines:
            line = line.strip().split(" ")
            assert len(line) == 3 # id, conf, angle
            if classes is None or int(line[0]) in classes:
                data.append(line)

        #if len(data) == 0:
        #    warnings.warn(f"No angle of classes {classes} found in {angle_file}")
        #    return None

        data=sorted(data, key=lambda line: float(line[1]))[::-1]
        selected_lines=data[:top_angles]
        angles=[line[-1] for line in selected_lines]
        return angles

