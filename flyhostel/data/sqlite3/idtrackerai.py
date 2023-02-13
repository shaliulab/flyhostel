
import os.path
import warnings
import sqlite3

import logging

from tqdm.auto import tqdm
import numpy as np

from idtrackerai.list_of_blobs import ListOfBlobs
from .sqlite3 import SQLiteExporter
from .deepethogram import DeepethogramExporter
from .orientation import OrientationExporter
from .constants import TABLES
from .utils import (
    table_is_not_empty,
    ensure_type
)

logger = logging.getLogger(__name__)


class IdtrackeraiExporter(SQLiteExporter, DeepethogramExporter, OrientationExporter):

    def __init__(self, basedir, deepethogram_data, *args, **kwargs):
        self._basedir = basedir
        self._deepethogram_data = deepethogram_data
        super(IdtrackeraiExporter, self).__init__(*args, **kwargs)

    # Init tables
    def init_data(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            table_name = "ROI_0"

            cols_list = [
                "frame_number int(11)", "in_frame_index int(2)", "x real(10)",
                "area int(11)", "y real(10)", "modified int(1)", "class_name char(10)"
            ]

            formated_cols_names = ", ".join(cols_list)
            command = f"CREATE TABLE IF NOT EXISTS {table_name} ({formated_cols_names})"
            cur.execute(command)

   # IDENTITY
    def init_identity_table(self, dbfile):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS IDENTITY (frame_number int(11), in_frame_index int(2), local_identity int(2), identity int(2));")



    # write
    def write_trajectory_and_identity_single_chunk(self, dbfile, chunk, **kwargs):

        blobs_collection = self.build_blobs_collection(chunk)
        video_path = self.build_video_object(chunk)

        if os.path.exists(blobs_collection):

            list_of_blobs = ListOfBlobs.load(blobs_collection)
            video_object = np.load(video_path, allow_pickle=True).item()

            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                cur = conn.cursor()

                start_end=(
                    video_object.episodes_start_end[0][0],
                    video_object.episodes_start_end[-1][-1]
                )
                blobs_in_video=list_of_blobs.blobs_in_video[start_end[0]:start_end[1]]

                for blobs_in_frame in tqdm(blobs_in_video, desc=f"Exporting chunk {chunk}", unit="frame"):
                    for blob in blobs_in_frame:
                        self.add_blob(cur, blob, **kwargs)

        else:
            warnings.warn(f"{blobs_collection} not found")


    def write_trajectory_and_identity(self, dbfile, chunks, **kwargs):

        for chunk in chunks:
            logger.debug("Exporting chunk %s", chunk)
            self.write_trajectory_and_identity_single_chunk(dbfile, chunk=chunk, **kwargs)


    def add_blob(self, cur, blob, w_trajectory=True, w_identity=True):
        if w_trajectory:
            self.write_blob_trajectory(cur, blob)

        if w_identity:
            self.write_blob_identity(cur, blob)


    def write_blob_trajectory(self, cur, blob):

        frame_number = ensure_type(blob.frame_number, "frame_number", int)
        in_frame_index = ensure_type(blob.in_frame_index, "in_frame_index", int)
        x_coord, y_coord = blob.centroid
        area = int(round(blob.area))
        modified = blob.modified
        if modified:
            class_name = blob._annotation["class"]
        else:
            class_name=None

        command = "INSERT INTO ROI_0 (frame_number, in_frame_index, x, y, area, modified, class_name) VALUES(?, ?, ?, ?, ?, ?, ?);"
        cur.execute(command, [frame_number, in_frame_index, x_coord, y_coord, area, modified, class_name])

    def write_blob_identity(self, cur, blob):
        frame_number = ensure_type(blob.frame_number, "frame_number", int)
        in_frame_index = ensure_type(blob.in_frame_index, "in_frame_index", int)

        local_identity = self._get_blob_local_identity(blob)
        identity_reference_to_ref_chunk = self._get_blob_identity(cur, blob, local_identity)
        command = "INSERT INTO IDENTITY (frame_number, in_frame_index, local_identity, identity) VALUES(?, ?, ?, ?);"
        cur.execute(command, [frame_number, in_frame_index, local_identity, identity_reference_to_ref_chunk])


    def _get_blob_local_identity(self, blob):
        local_identity = blob.final_identities[0]
        if local_identity is None:
            local_identity = 0

        local_identity=ensure_type(local_identity, "local_identity", int)
        return local_identity

    def _get_blob_identity(self, cur, blob, local_identity):

        chunk=blob.chunk
        chunk=ensure_type(chunk, "chunk", int)

        cmd="SELECT identity FROM CONCATENATION WHERE chunk = ? AND local_identity=?;"
        args=(chunk, local_identity)
        cur.execute(cmd, args)
        try:
            identity_reference_to_ref_chunk = int(cur.fetchone()[0])
        except Exception as error:
            print(f"Query {cmd} with args {args} args failed")
            raise error
            # import ipdb; ipdb.set_trace()

        return identity_reference_to_ref_chunk




    def init_tables(self, dbfile, tables, reset=True):
        super(IdtrackeraiExporter, self).init_tables(dbfile, tables, reset=reset)
        if "IDENTITY" in tables:
            self.init_identity_table(dbfile)

        if "ROI_0" in tables:
            self.init_data(dbfile)

        if "ORIENTATION" in tables:
            self.init_orientation_table(dbfile)

        if "BEHAVIORS" in tables:
            self.init_behaviors_table(dbfile)


    def export(self, dbfile, chunks, tables="all", mode="w", reset=False, behaviors=None):
        """
        Export datasets into single SQLite file

        Args:

            dbfile (str): Path to SQLite output file
            chunks (list): Chunks to be processed
            tables (list, str): List of tables to be exported or "all" if all should be
            mode (str)
            reset (bool):
            behaviors (list):
        """

        if tables is None or tables == "all":
            tables = TABLES

        assert chunks is not None

        super(IdtrackeraiExporter, self).export(dbfile, chunks=chunks, tables=tables, mode=mode, reset=reset)

        if "ROI_0" in tables or "IDENTITY" in tables:
            w_trajectory="ROI_0" in tables
            w_identity="IDENTITY" in tables

            if w_identity and not table_is_not_empty(dbfile, "CONCATENATION"):
                raise ValueError("IDENTITY table requires CONCATENATION;")

            self.write_trajectory_and_identity(
                dbfile,
                w_trajectory=w_trajectory,
                w_identity=w_identity,
                chunks=chunks
            )

        if "ORIENTATION" in tables:
            self.write_orientation_table(dbfile, chunks=chunks)

        if "BEHAVIORS" in tables:
            self.write_behaviors_table(dbfile, behaviors=behaviors)
