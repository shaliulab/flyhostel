"""
Centralize the results obtained in the FlyHostel pipeline into a single SQLite file
that can be used to perform all downstream analyses
"""

import os.path
import pickle
import warnings
import sqlite3
import yaml
import datetime
import json
import logging
import tempfile
import glob
import subprocess
import shlex
import math

from tqdm.auto import tqdm
import cv2
import numpy as np

from idtrackerai.list_of_blobs import ListOfBlobs
from imgstore.stores.utils.mixins.extract import _extract_store_metadata
from imgstore.constants import STORE_MD_FILENAME
from flyhostel.data.deepethogram import H5Reader

logger = logging.getLogger(__name__)

METADATA_FILE = "metadata.csv"
RAISE_EXCEPTION_IF_METADATA_NOT_FOUND=True
DEEPETHOGRAM_DATA="/staging/leuven/stg_00115/Data/flyhostel_data/flyhostel_deepethogram/DATA/"

try:
    DOWNLOAD_BEHAVIORAL_DATA=os.environ.get("DOWNLOAD_BEHAVIORAL_DATA", None)
    assert DOWNLOAD_BEHAVIORAL_DATA is not None and os.path.exists(DOWNLOAD_BEHAVIORAL_DATA)

except AssertionError:
    warnings.warn(
        """
        download-behavioral-data not found. Automatic download of metadata not available.
        Please ensure the DOWNLOAD_BEHAVIORAL_DATA environment variable is set and pointing to a download-behavioral-data executable
        """)
    DOWNLOAD_BEHAVIORAL_DATA = None

    

TABLES = ["METADATA", "IMG_SNAPSHOTS", "ROI_MAP", "VAR_MAP", "ROI_0", "IDENTITY", "BEHAVIORS", "STORE_INDEX", "ENVIRONMENT", "AI"]


def metadata_not_found(message):

    if RAISE_EXCEPTION_IF_METADATA_NOT_FOUND:
        raise Exception(message)
    else:
        warnings.warn(message)



class IdtrackeraiExporter:

    def init_data(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            table_name = "ROI_0"

            cols_list = ["frame_number int(11)", "in_frame_index int(2)", "x real(10)", "area int(11)", "y real(10)", "modified int(1)"]
            
            formated_cols_names = ", ".join(cols_list)
            command = "CREATE TABLE IF NOT EXISTS %s (%s)" % (table_name ,formated_cols_names)
            cur.execute(command)

    def write_trajectory_and_identity(self, dbfile, chunk, **kwargs):

        blobs_collection = self.build_blobs_collection(chunk)

        if os.path.exists(blobs_collection):

            list_of_blobs = ListOfBlobs.load(blobs_collection)

            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                cur = conn.cursor()

                for blobs_in_frame in tqdm(list_of_blobs.blobs_in_video, desc=f"Exporting chunk {chunk}"):
                    for blob in blobs_in_frame:
                        self.add_blob(cur, blob, **kwargs)
        
        else:
            warnings.warn(f"{blobs_collection} not found")

    def add_blob(self, cur, blob, w_trajectory=True, w_identity=True):

        frame_number = blob.frame_number
        in_frame_index = blob.in_frame_index
        x, y = blob.centroid
        area = int(round(blob.area))
        modified = blob.modified
        identity = blob.final_identities[0]
        if identity is None:
            identity = 0


        if w_trajectory:
            command = "INSERT INTO ROI_0 (frame_number, in_frame_index, x, y, area, modified) VALUES(?, ?, ?, ?, ?, ?);"
            cur.execute(command, [frame_number, in_frame_index, x, y, area, modified])
        
        if w_identity:
            command = "INSERT INTO IDENTITY (frame_number, in_frame_index, identity) VALUES(?, ?, ?);"
            cur.execute(command, [frame_number, in_frame_index, identity])


class SQLiteExporter(IdtrackeraiExporter):

    def __init__(self, basedir, deepethogram_data=None, n_jobs=1):

        self._basedir = os.path.realpath(basedir)
        
        self._store_metadata_path = os.path.join(self._basedir, STORE_MD_FILENAME)
        self._store_metadata = _extract_store_metadata(self._store_metadata_path) 
        
        self._idtrackerai_conf_path = os.path.join(self._basedir, f"{os.path.basename(self._basedir)}.conf")
        with open(self._idtrackerai_conf_path, "r") as filehandle:
            self._idtrackerai_conf = yaml.load(filehandle, yaml.SafeLoader)

        matches = glob.glob(os.path.join(self._basedir, "*pfs"))
        if matches:
            self._camera_metadata_path = matches[0]
        else:
            metadata_not_found(f"Camera metadata (.pfs file) not found")
            self._camera_metadata_path = None

            
        self._temp_path = tempfile.mktemp(prefix="flyhostel_", suffix=".jpg")
        self._number_of_animals = None
        self._index_dbfile = os.path.join(self._basedir, "index.db")
        self._deepethogram_data = deepethogram_data
        self._n_jobs=n_jobs


    @staticmethod
    def download_metadata(path):
        """
        Download the metadata from a Google Sheets database
        """

        try:
            if DOWNLOAD_BEHAVIORAL_DATA is None:
                raise Exception("Please define DOWNLOAD_BEHAVIORAL_DATA as the path to the download-behavioral-data Python binary")

            path=path.replace(" ", "_")
            cmd = f'{DOWNLOAD_BEHAVIORAL_DATA} --metadata {path}'
            cmd_list = shlex.split(cmd)
            process = subprocess.Popen(cmd_list)
            process.communicate()
            print(f"Downloading metadata to {path}")
            return 0
        except:
            metadata_not_found(f"Could not download metadata to {path}")
            return 1


    def export(self, dbfile, mode=["w", "a"], overwrite=False, behaviors=None, tables=None, **kwargs):
        print(f"Saving to --> {dbfile}")
        assert dbfile.endswith(".db")

        if tables is None or tables == "all":
            tables = TABLES
        if os.path.exists(dbfile):
            if overwrite:
                warnings.warn(f"{dbfile} exists. Remaking from scratch and ignoring mode")
                os.remove(dbfile)
            elif mode == "w":
                warnings.warn(f"{dbfile} exists. Overwriting (mode=w)")
            elif mode == "a":
                warnings.warn(f"{dbfile} exists. Appending (mode=a)")
           
        print(f"Initializing file {dbfile}")
        self.init_tables(dbfile)

        print(f"Writing tables: {tables}")
        
        if "METADATA" in tables:
            self.write_metadata_table(dbfile)

        if "IMG_SNAPSHOTS" in tables:
            self.write_snapshot_table(dbfile, **kwargs)

        if "ROI_MAP" in tables:
            self.write_roi_map_table(dbfile)

        if "ENVIRONMENT" in tables:
            self.write_environment_table(dbfile, **kwargs)

        if "VAR_MAP" in tables:
            self.write_var_map_table(dbfile)

        if "ROI_0" in tables and "IDENTITY" in tables:
            self.write_trajectory_and_identity(dbfile, **kwargs)

        if "STORE_INDEX" in tables:
            self.write_index_table(dbfile)

        if "BEHAVIORS" in tables:
            self.write_behaviors_table(dbfile, behaviors=behaviors)

        if "AI" in tables:
            self.write_ai_table(dbfile)
            

    def write_trajectory_and_identity(self, dbfile, chunks):
    
        for chunk in chunks:
            logger.debug(f"Exporting chunk {chunk}")
            super(SQLiteExporter, self).write_trajectory_and_identity(dbfile, chunk)


    @property
    def number_of_animals(self):
        if self._number_of_animals is None:
            self._number_of_animals = int(self._idtrackerai_conf["_number_of_animals"]["value"])
        return self._number_of_animals
    
    def init_tables(self, dbfile):

        self.init_metadata_table(dbfile)
        # self.init_start_events_table()
        # self.init_qc_table()
        self.init_snapshot_table(dbfile)
        self.init_roi_map_table(dbfile)
        self.init_environment_table(dbfile)
        self.init_var_map_table(dbfile)
        self.init_identity_table(dbfile)
        self.init_orientation_table(dbfile)
        self.init_data(dbfile)
        self.init_index_table(dbfile)
        self.init_behaviors_table(dbfile)
        self.init_ai_table(dbfile)

    def build_blobs_collection(self, chunk):
        return os.path.join(self._basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "blobs_collection.npy")


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


        with open(self._idtrackerai_conf_path, "r") as filehandle:
            idtrackerai_conf_str = filehandle.read()

        
        if self._camera_metadata_path is not None and os.path.exists(self._camera_metadata_path):
            with open(self._camera_metadata_path, "r") as filehandle:
                camera_metadata_str = filehandle.read()
        else:
            camera_metadata_str=""

        ethoscope_metadata_path = os.path.join(self._basedir, METADATA_FILE)

        if os.path.exists(ethoscope_metadata_path) or self.download_metadata(ethoscope_metadata_path) == 0:
            with open(ethoscope_metadata_path, "r") as filehandle:
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
            first_chunk = int(cur.execute.fetchone()[0])
            cur.execute("SELECT chunk FROM frames ORDER BY chunk DESC LIMIT 1;")
            last_chunk = int(cur.execute.fetchone()[0])
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
                    f"INSERT INTO METADATA (field, value) VALUES (?, ?);",
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

        roi = np.array(eval(self._idtrackerai_conf["_roi"]["value"][0][0]))
        x, y, w, h = cv2.boundingRect(roi)

        mask = np.zeros(self._store_metadata["imgshape"])
        mask = cv2.drawContours(mask, [roi], -1, 255, -1)
        mask = self.serialize_arr(mask, self._temp_path)

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute(
                f"INSERT INTO ROI_MAP (roi_idx, roi_value, x, y, w, h, mask) VALUES (?, ?, ?, ?, ?, ?, ?);",
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
                        f"INSERT INTO ENVIRONMENT (frame_number, camera_temperature, temperature, humidity, light) VALUES (?, ?, ?, ?, ?);",
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
                    f"INSERT INTO VAR_MAP (var_name, sql_type, functional_type) VALUES (?, ?, ?);",
                    val
                )


    def init_index_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS STORE_INDEX (chunk int(3), frame_number int(11), frame_time int(11));")

    def init_behaviors_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute(f"DROP TABLE IF EXISTS BEHAVIORS;")
            cur.execute("CREATE TABLE IF NOT EXISTS BEHAVIORS (frame_number int(11), in_frame_index int(2), behavior char(100), probability float(5));")

    def init_ai_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute(f"DROP TABLE IF EXISTS AI;")
            cur.execute("CREATE TABLE IF NOT EXISTS AI (frame_number int(11), ai int(2);")


    def write_ai_table(self, dbfile):

        pickle_files = sorted(glob.glob(os.path.join(self._basedir, "idtrackerai", "session_*", "preprocessing", "ai.pkl")))

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()

            for file in pickle_files:
                with open(file, "rb") as filehandle:
                    ai_mods = pickle.load(filehandle)
                    frames=ai_mods["success"]
                
                for frame_number in frames:
                    cur.execute("INSERT INTO AI (frame_number, ai) VALUES (?, ?)", (frame_number, "YOLOv7"))

    def write_behaviors_table(self, *args, **kwargs):

        for in_frame_index in self.range(self.number_of_animals):
            self.write_behaviors_table_single_blob(*args, in_frame_index, **kwargs)

    def write_behaviors_table_single_blob(self, dbfile, in_frame_index, behaviors=None, chunks=None):

        if self._deepethogram_data is None:
            warnings.warn(f"Please pass a deepethogram data folder")
            return

        prefix = "_".join(self._basedir.split(os.path.sep)[-3:])
        reader = H5Reader.from_outputs(data_dir=self._deepethogram_data, prefix=prefix, in_frame_index=in_frame_index, fps=self._store_metadata["framerate"])

        if behaviors is None:
            behaviors=reader.class_names


        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            with sqlite3.connect(self._index_dbfile, check_same_thread=False) as index_db:

                index_db_cur = index_db.cursor()

                for behavior_idx, behavior in enumerate(behaviors):

                    chunks_avail, P = reader.load(behavior, n_jobs=self._n_jobs)
                    pb=tqdm(total=len(chunks_avail), desc=f"Loading {behavior} instances for blob index {in_frame_index}", position=behavior_idx, unit="chunk")

                    for chunk_idx, chunk in enumerate(chunks_avail):
                        if chunks is not None and chunk not in chunks:
                            continue
                        index_db_cur.execute("SELECT frame_number FROM frames WHERE chunk = ?;", (chunk, ))
                        frame_numbers = [int(x[0]) for x in index_db_cur.fetchall()]
                        assert P[chunk_idx].shape[0] == len(frame_numbers)
                        
                        for frame_number_idx, frame_number in enumerate(frame_numbers):
                            args=(frame_number, in_frame_index, behavior, P[chunk_idx][frame_number_idx].item())
                            cur.execute("INSERT INTO BEHAVIORS (frame_number, in_frame_index, behavior, probability) VALUES (?, ?, ?, ?);", args)
                        
                        pb.update(1)



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


    # ORIENTATION
    def init_orientation_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS ORIENTATION (frame_number int(11), in_frame_index int(2), angle float(5), is_inferred int(1));")


    @staticmethod
    def fetch_angle(label_file):
        """
        Return the angle of the best detection
        """

        with open(label_file, "r") as filehandle:
            lines = filehandle.readlines()

        data = []
        for line in lines:
            line = line.strip().split(" ")
            assert len(line) == 7 # id, x, y, w, h, conf, angle
            data.append(line)
        
        data=sorted(data, key=lambda line: float(line[5]))[::-1]
        return data[0][-1]

    def _write_orientation_table(self, conn, dbfile, in_frame_index=0):

        label_database_path = os.path.join(self._basedir, "angles", "FlyHead", "labels")
        label_files = glob.glob(os.path.join(label_database_path, f"*-{in_frame_index}.txt"))
        assert len(label_files) > 0, f"No label database found under {label_database_path} for in_frame_index {in_frame_index}"
        label_database = [(int(os.path.basename(label_file).split("_")[0]), label_file) for label_file in label_files]
        label_database = sorted(label_database, key=lambda entry: entry[0])
        pointer = 0
        report_every_n_lines=math.inf
 
        fns = conn.cursor()
        frame_numbers = fns.execute("SELECT frame_number FROM ROI_0;")
        cur = conn.cursor()

        for row in tqdm(frame_numbers, desc=f"Writing orientation data from in_frame_index {in_frame_index} to {dbfile}"):
            frame_number = row[0]
            entry = label_database[pointer]
            if frame_number == entry[0]:
                label_file = entry[1]
                angle = self.fetch_angle(label_file)
                is_inferred=False
                pointer+=1
                if pointer % report_every_n_lines == 0:
                    print(f"{pointer} labels have been added to {dbfile}")
            else: # no matches
                angle = None
                is_inferred=True
    
            cur.execute(
                f"INSERT INTO ORIENTATION (frame_number, in_frame_index, angle, is_inferred) VALUES (?, ?, ?, ?);",
                (frame_number, in_frame_index, angle, is_inferred)
            )
            if pointer == len(label_database):
                return

    def write_orientation_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            for in_frame_index in range(self.number_of_animals):
                self._write_orientation_table(conn, dbfile, in_frame_index)

    # IDENTITY
    def init_identity_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS IDENTITY (frame_number int(11), in_frame_index int(2), identity int(2));")


    def write_identity_table(self, dbfile):
        raise NotImplementedError


def export_dataset(metadata, chunks, overwrite=False, tables=None):

    basedir = os.path.dirname(metadata)
    dbfile_basename = "_".join(basedir.split(os.path.sep)[-3:]) + ".db"

    dbfile = os.path.join(basedir, dbfile_basename)

    dataset = SQLiteExporter(basedir, deepethogram_data=DEEPETHOGRAM_DATA)
    dataset.export(dbfile=dbfile, mode="w", chunks=chunks, overwrite=overwrite, tables=tables)
