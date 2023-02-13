from abc import ABC
import os.path
import sqlite3
import warnings
import re
import glob

import h5py
from tqdm.auto import tqdm

from .utils import table_is_not_empty

class OrientationExporter(ABC):

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


    def init_orientation_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS ORIENTATION (frame_number int(11), in_frame_index int(2), angle float(5), is_inferred int(1));")


    def _write_orientation_table(self, conn, dbfile, chunks, in_frame_index=0, queue=None):

        angle_database_path = os.path.join(self._basedir, "angles", "FlyHead", "angles")

        angle_database = sorted(glob.glob(os.path.join(angle_database_path, "*.hdf5")))
        angle_database = {self._parse_chunk_from_angle_file(path): path for path in angle_database}

        cur = conn.cursor()
        is_inferred=False

        for chunk in tqdm(chunks, desc=f"Writing orientation data from in_frame_index {in_frame_index} to {dbfile}"):
            accum = 0
            try:
                h5py_file = angle_database[chunk]
            except KeyError:
                warnings.warn(f"No angles for chunk {chunk}")
                continue

            with h5py.File(h5py_file, "r") as filehandle:
                keys = list(filehandle.keys())
                for dataset in keys:
                    frame_number, chunk_, in_frame_index_ = self._parse_dataset(dataset)
                    assert chunk_ == chunk
                    if in_frame_index_ != in_frame_index:
                        continue

                    angle = self.fetch_angle_from_h5py(filehandle, dataset)
                    accum+=1

                    data=(frame_number, in_frame_index, angle, is_inferred)
                    if queue is None:
                        cur.execute(
                            "INSERT INTO ORIENTATION (frame_number, in_frame_index, angle, is_inferred) VALUES (?, ?, ?, ?);",
                            data
                        )
                    else:
                        queue.put(tuple([str(e) for e in data]), timeout=30, block=True)

            print(f"Wrote {accum} angles for chunk {chunk} in {dbfile}")


    def write_orientation_table(self, dbfile, chunks):

        if not table_is_not_empty(dbfile, "STORE_INDEX"):
            raise ValueError("ORIENTATION table requires STORE_INDEX")

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:

            for in_frame_index in range(self.number_of_animals):
                self._write_orientation_table(conn, dbfile, chunks=chunks, in_frame_index=in_frame_index, queue=None)
