"""
Centralize the results obtained in the FlyHostel pipeline into a single SQLite file
that can be used to perform all downstream analyses
"""
from abc import ABC
import sqlite3
import warnings
import os.path
import logging
from ast import literal_eval
import json

import cv2
import numpy as np

from .async_writer import AsyncSQLiteWriter
from .utils import table_is_not_empty, serialize_arr
from .store_index import StoreIndexExporter
from .ai import AIExporter
from .concatenation import ConcatenationExporter
from .metadata import MetadataExporter
from .snapshot import SnapshotExporter
logger = logging.getLogger(__name__)

class SQLiteExporter(SnapshotExporter, AIExporter, ConcatenationExporter, MetadataExporter, StoreIndexExporter, ABC):

    _CLASSES = {0: "head"}
    _AsyncWriter = AsyncSQLiteWriter
    _basedir=None


    def __init__(self, *args, **kwargs):
        self._number_of_animals = None
        self._index_dbfile = os.path.join(self._basedir, "index.db")
        super(SQLiteExporter, self).__init__(*args, **kwargs)
        self._must_have_snapshot = self._store_metadata.get("must_have_snapshot", True)


    def export(self, dbfile, chunks, tables, reset=True, **kwargs):

        self.init_tables(dbfile, tables, reset=reset, **kwargs)
        print(f"Writing tables: {tables}")
        
        if "CONCATENATION" in tables:
            print("Writing CONCATENATION")
            self.write_concatenation_table(dbfile, chunks=chunks)
            print("CONCATENATION done")

        if "METADATA" in tables:
            print("Writing METADATA")
            self.write_metadata_table(dbfile)
            print("METADATA done")

        if "IMG_SNAPSHOTS" in tables:
            print("Writing IMG_SNAPSHOTS")
            self.write_snapshot_table(dbfile, chunks=chunks)
            print("IMG_SNAPSHOTS done")

        if "ROI_MAP" in tables:
            print("Writing ROI_MAP")
            self.write_roi_map_table(dbfile)
            print("ROI_MAP done")

        if "ENVIRONMENT" in tables:
            print("Writing ENVIRONMENT")
            self.write_environment_table(dbfile, chunks=chunks)
            print("ENVIRONMENT done")

        if "VAR_MAP" in tables:
            print("Writing VAR_MAP")
            self.write_var_map_table(dbfile)
            print("VAR_MAP done")

        if "STORE_INDEX" in tables:
            print("Writing STORE_INDEX")
            self.write_index_table(dbfile)
            print("STORE_INDEX done")

        if "AI" in tables:
            print("Writing AI")
            self.write_ai_table(dbfile, chunks=chunks)
            print("AI done")

        if "LANDMARKS" in tables:
            self.write_landmarks_table(dbfile, chunks=chunks)
            print("LANDMARKS done")



    @property
    def number_of_animals(self):
        if self._number_of_animals is None:
            self._number_of_animals = int(self._idtrackerai_conf["_number_of_animals"]["value"])
        return self._number_of_animals


    def init_tables(self, dbfile, tables, reset=True):

        if "METADATA" in tables:
            self.init_metadata_table(dbfile, reset=reset)
        if "IMG_SNAPSHOTS" in tables:
            self.init_snapshot_table(dbfile, reset=reset)
        if "ROI_MAP" in tables:
            self.init_roi_map_table(dbfile, reset=reset)
        if "ENVIRONMENT" in tables:
            self.init_environment_table(dbfile, reset=reset)
        if "VAR_MAP" in tables:
            self.init_var_map_table(dbfile, reset=reset)
        if "STORE_INDEX" in tables:
            self.init_index_table(dbfile, reset=reset)
        if "AI" in tables:
            self.init_ai_table(dbfile, reset=reset)
        if "CONCATENATION" in tables:
            self.init_concatenation_table(dbfile, reset=reset)
        if "LANDMARKS" in tables:
            self.init_landmarks_table(dbfile, reset=reset)



    def build_blobs_collection(self, chunk):
        path=os.path.join(self._basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "tracking", "blobs_collection.npy")
        if not os.path.exists(path):
            path=os.path.join(self._basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")
        return path
    

    def build_video_object(self, chunk):
        return os.path.join(self._basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "video_object.npy")

    # ROI_MAP
    def init_roi_map_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute("DROP TABLE IF EXISTS ROI_MAP;")
            cur.execute("CREATE TABLE IF NOT EXISTS ROI_MAP (roi_idx smallint(6), roi_value smallint(6), x smallint(6), y smallint(6), w smallint(6), h smallint(6), mask longblob);")

    def write_roi_map_table(self, dbfile):

        roi = np.array(literal_eval(self._idtrackerai_conf["_roi"]["value"][0][0]))
        x_coord, y_coord, width, height = cv2.boundingRect(roi)

        mask = np.zeros(self._store_metadata["imgshape"])
        mask = cv2.drawContours(mask, [roi], -1, 255, -1)
        mask = serialize_arr(mask, self._temp_path)

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO ROI_MAP (roi_idx, roi_value, x, y, w, h, mask) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (0, 0, x_coord, y_coord, width, height, mask)
            )

    def init_environment_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                print("Dropping ENVIRONMENT")
                cur.execute("DROP TABLE IF EXISTS ENVIRONMENT;")
            cur.execute("CREATE TABLE IF NOT EXISTS ENVIRONMENT (frame_number int(11), camera_temperature real(6), temperature real(6), humidity real(6), light real(6));")


    def write_environment_table(self, dbfile, chunks):

        data=[]

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            for chunk in chunks:

                extra_json = os.path.join(self._basedir, f"{str(chunk).zfill(6)}.extra.json")

                if not os.path.exists(extra_json):
                    warnings.warn(f"No environmental data available for chunk {chunk}")
                    return

                with open(extra_json, "r", encoding="utf8") as filehandle:
                    extra_data = json.load(filehandle)

                for row in extra_data:
                    data.append((row["frame_number"], row["camera_temperature"], row["temperature"], row["humidity"], row["light"]))

                conn.executemany(
                    "INSERT INTO ENVIRONMENT (frame_number, camera_temperature, temperature, humidity, light) VALUES (?, ?, ?, ?, ?);",
                    data
                )


    # VAR_MAP
    def init_var_map_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute("DROP TABLE IF EXISTS VAR_MAP;")
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
