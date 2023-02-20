from abc import ABC
import os.path
import sqlite3
import warnings
import re
import logging
import glob

import h5py
from tqdm.auto import tqdm

from .utils import table_is_not_empty

logger = logging.getLogger(__name__)

class OrientationExporter(ABC):
    """Generate the ORIENTATION table of a FlyHostel SQLite dataset

    This table contains the angle between head and animal centroid in degrees
    The angles are read from a database of .hdf5 files (one per chunk) under

    basedir/angles/FlyHead/angles

    where each angle is stored under a systematically named key (frameNumber_chunk_inFrameIndex) with value a tuple
    where the third element ([2]) is the angle. The first two are the class id (always 0 for the head) and confidence
    (0 to 1 where 0 is lowest confidence and 1 is highest confidence)
    The angle is store
    """

    _basedir = None
    number_of_animals=None

    @staticmethod
    def _parse_chunk_from_angle_file(path):
        return int(
            re.search("angles_([0-9][0-9][0-9][0-9][0-9][0-9]).hdf5",
            os.path.basename(path)).group(1)
        )


    @staticmethod
    def _parse_dataset(dataset):
        frame_number, chunk, in_frame_index = dataset.split("_")
        frame_number=int(frame_number)
        chunk=int(chunk)
        in_frame_index=int(in_frame_index)
        return frame_number, chunk, in_frame_index


    @staticmethod
    def fetch_angle_from_h5py(filehandle, dataset):
        try:
            # class_id, conf, angle
            _, _, angle = filehandle[dataset][:]
        except KeyError:
            angle=None
        except Exception as error:
            raise error

        return angle


    def init_orientation_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute("DROP TABLE IF EXISTS ORIENTATION;")
            cur.execute("CREATE TABLE IF NOT EXISTS ORIENTATION (frame_number int(11), in_frame_index int(2), angle float(5), is_inferred int(1));")

            print("Creating indices for ORIENTATION table")
            cur.execute("CREATE INDEX ori_fn ON ORIENTATION (frame_number);")

    def _write_orientation_table(self, conn, dbfile, chunks, in_frame_index=0):

        angle_database_path = os.path.join(self._basedir, "angles", "FlyHead", "angles")

        angle_database = sorted(glob.glob(os.path.join(angle_database_path, "*.hdf5")))
        angle_database = {self._parse_chunk_from_angle_file(path): path for path in angle_database}
        is_inferred=False

        for chunk in tqdm(chunks, desc=f"Exporting orientation data for in_frame_index {in_frame_index}", unit="chunk"):
            accum = 0
            try:
                h5py_file = angle_database[chunk]
            except KeyError:
                warnings.warn(f"No angles for chunk {chunk}")
                continue

            try:
                with h5py.File(h5py_file, "r") as filehandle:
                    keys = list(filehandle.keys())
                    data=[]
                    for dataset in keys:
                        frame_number, chunk_, in_frame_index_ = self._parse_dataset(dataset)
                        assert chunk_ == chunk
                        if in_frame_index_ != in_frame_index:
                            continue

                        angle = self.fetch_angle_from_h5py(filehandle, dataset)
                        accum+=1

                        args=(frame_number, in_frame_index, angle, is_inferred)
                        data.append(args)
                    if data:
                        conn.executemany(
                            "INSERT INTO ORIENTATION (frame_number, in_frame_index, angle, is_inferred) VALUES (?, ?, ?, ?);",
                            data
                        )

            except OSError as error:
                print(f"Unable to open file {h5py_file}")
                raise error

            logger.debug(f"Wrote {accum} angles for chunk {chunk} in {dbfile}")


    def write_orientation_table(self, dbfile, chunks):

        if not table_is_not_empty(dbfile, "STORE_INDEX"):
            raise ValueError("ORIENTATION table requires STORE_INDEX")

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            for in_frame_index in range(self.number_of_animals):
                self._write_orientation_table(conn, dbfile, chunks=chunks, in_frame_index=in_frame_index)
