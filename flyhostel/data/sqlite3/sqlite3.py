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
from .constants import (
    TABLES,
)
from .utils import table_is_not_empty, serialize_arr
from .store_index import StoreIndexExporter
from .ai import AIExporter
from .concatenation import ConcatenationExporter
from .metadata import MetadataExporter
from .snapshot import SnapshotExporter
logger = logging.getLogger(__name__)


class SQLiteExporter(StoreIndexExporter, SnapshotExporter, AIExporter, ConcatenationExporter, MetadataExporter, ABC):

    _CLASSES = {0: "head"}
    _AsyncWriter = AsyncSQLiteWriter
    _basedir=None


    def __init__(self, *args, **kwargs):
        self._number_of_animals = None
        self._index_dbfile = os.path.join(self._basedir, "index.db")
        super(SQLiteExporter, self).__init__(*args, **kwargs)


    def export(self, dbfile, chunks, tables=None, mode="a", reset=False):

        if tables is None or tables == "all":
            tables = TABLES

        if os.path.exists(dbfile):
            if reset and False:
                warnings.warn(f"{dbfile} exists. Remaking from scratch and ignoring mode")
                print(f"Remaking file {dbfile}")
                os.remove(dbfile)
            elif mode == "w":
                print(f"Resuming file {dbfile}")
                warnings.warn(f"{dbfile} exists. Overwriting (mode=w)")
            elif mode == "a":
                print(f"Resuming file {dbfile}")


        self.init_tables(dbfile, tables, reset=reset)
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


    def init_tables(self, dbfile, tables, reset=True):

        #TODO
        # Add if for all tables

        if "METADATA" in tables:
            self.init_metadata_table(dbfile)

        self.init_snapshot_table(dbfile)
        self.init_roi_map_table(dbfile)
        self.init_environment_table(dbfile)
        self.init_var_map_table(dbfile)
        self.init_index_table(dbfile)

        self.init_ai_table(dbfile, reset=reset)
        self.init_concatenation_table(dbfile, reset=reset)

    def build_blobs_collection(self, chunk):
        return os.path.join(self._basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")

    def build_video_object(self, chunk):
        return os.path.join(self._basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "video_object.npy")

    # ROI_MAP
    def init_roi_map_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
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


                with open(extra_json, "r", encoding="utf8") as filehandle:
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
