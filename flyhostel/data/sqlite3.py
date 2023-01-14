import os.path
import warnings
import sqlite3
import yaml
import datetime
import logging
import cv2
import numpy as np

from idtrackerai.list_of_blobs import ListOfBlobs
from imgstore.stores.utils.mixins.extract import _extract_store_metadata
from imgstore.constants import STORE_MD_FILENAME

logger = logging.getLogger(__name__)

class IdtrackeraiExporter:

    def init_data(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            table_name = "ROI_0"

            cols_list = ["frame_number int(11)", "blob_index int(2)", "x real(10)", "area int(11)", "y real(10)", "modified int(1)"]
            
            formated_cols_names = ", ".join(cols_list)
            command = "CREATE TABLE %s (%s)" % (table_name ,formated_cols_names)
            cur.execute(command)

    def write_data(self, dbfile, chunk):

        list_of_blobs = ListOfBlobs.load(self.build_blobs_collection(chunk))

        for blobs_in_frame in list_of_blobs.blobs_in_video:
            for blob in blobs_in_frame:
                self.add_blob(dbfile, blob)

    def add_blob(self, dbfile, blob):

        frame_number = blob.frame_number
        blob_index = blob.blob_index
        x, y = blob.centroid
        area = int(round(blob.area))
        modified = blob.modified
        identity = blob.final_identities[0]
        if identity is None:
            identity = 0


        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            command = "INSERT INTO ROI_0 (frame_number, blob_index, x, y, area, modified) VALUES(?, ?, ?, ?, ?, ?);"
            cur.execute(command, [frame_number, blob_index, x, y, area, modified])
            command = "INSERT INTO IDENTITY (frame_number, blob_index, identity) VALUES(?, ?, ?);"
            cur.execute(command, [frame_number, blob_index, identity])


class SQLiteExporter(IdtrackeraiExporter):

    def __init__(self, basedir):

        self._basedir = os.path.realpath(basedir)
        self._store_metadata = _extract_store_metadata(os.path.join(self._basedir, STORE_MD_FILENAME)) 
        with open(os.path.join(self._basedir, f"{os.path.basename(self._basedir)}.conf"), "r") as filehandle:
            self._idtrackerai_conf = yaml.load(filehandle, yaml.SafeLoader)

        self._number_of_animals = None


    def export(self, dbfile, mode=["w", "a"], **kwargs):
        assert dbfile.endswith(".db")
        if os.path.exists(dbfile):
            if mode == "w":
                warnings.warn(f"{dbfile} exists. Overwriting (mode=w)")
            elif mode == "a":
                warnings.warn(f"{dbfile} exists. Appending (mode=a)")


        self.init_tables(dbfile)
        self.write_metadata_table(dbfile)
        self.write_roi_map_table(dbfile)
        self.write_var_map_table(dbfile)
        self.write_data(dbfile, **kwargs)


    def write_data(self, dbfile, chunks):
    
        for chunk in chunks:
            logger.debug(f"Exporting chunk {chunk}")
            super(self, SQLiteExporter).write_data(dbfile, chunk)


    @property
    def number_of_animals(self):
        if self._number_of_animals is None:
            self._number_of_animals = int(self._idtrackerai_conf["_number_of_animals"]["value"])
        return self._number_of_animals
    
    def init_tables(self, dbfile):

        self.init_metadata_table(dbfile)
        # self.init_start_events_table()
        # self.init_qc_table()
        self.init_roi_map_table(dbfile)
        self.init_var_map_table(dbfile)
        self.init_identity_table(dbfile)
        self.init_data(dbfile)

    @staticmethod
    def build_blobs_collection(self, chunk):
        return os.path.join(self._basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")


    # METADATA
    def init_metadata_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute(f"CREATE TABLE METADATA (field char(100), value varchar(4000));")

    def write_metadata_table(self, dbfile):

        machine_id = "0" * 32
        machine_name = ""
        created_utc=self.store_metadata["created_utc"].split(".")[0]
        date_time = datetime.datetime.strptime(created_utc, "%Y-%m-%dT%H:%M:%S").timestamp()


        values = [
            ("machine_id", machine_id),
            ("machine_name", machine_name),
            ("date_time", date_time),
            ("frame_width", self._store_metadata["imgshape"][1]),
            ("version", self._store_metadata["imgshape"][0]),
            ("experimental_info", ""),
            ("selected_options", ""),
            # TODO
            ("ethoscope_metadata", "")
        ]

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            for val in values:

                cur.execute(
                    f"INSERT INTO METADATA (field, value) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                    val
                )

    # ROI_MAP
    def init_roi_map_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute(f"CREATE TABLE ROI_MAP (roi_idx smallint(6), roi_value smallint(6), x smallint(6), y smallint(6), w smallint(6), h smallint(6), mask longblob);")


    def write_roi_map_table(self, dbfile):

        roi = np.array(eval(self._idtrackerai_conf["_roi"]["value"][0][0]))

        x, y, w, h = cv2.boundingRect(roi)
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            for i in range(1, self.number_of_animals+1):
                cur.execute(
                    f"INSERT INTO ROI_MAP (roi_idx, roi_value, x, y, w, h, mask) VALUES (?, ?, ?, ?, ?, ?, ?);",
                    [i, i, x, y, w, h, roi]
                )

    # VAR_MAP
    def init_var_map_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute(f"CREATE TABLE VAR_MAP (var_name char(100), sql_type char(100), functional_type char(100));")


    def write_var_map_table(self, dbfile):
       with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()

            values = [
                ("frame_number", "INT", "count"),
                ("blob_index", "INT", "count"),
                ("x", "REAL", "distance"),
                ("y", "REAL", "distance"),
                ("area", "INT", "count"),
                ("modified", "INT", "bool"),
            ]

            for val in values:
                cur.execute(
                    f"INSERT INTO VAR_MAP (var_name, sql_type, functional_type) VALUES (?, ?, ?);",
                    val
                )

    # IDENTITY
    def init_identity_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute(f"CREATE TABLE IDENTITY (frame_number int(11), blob_index int(2), identity int(2));")

