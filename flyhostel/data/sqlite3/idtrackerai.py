import time
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
from .sleap import SleapExporter
from .qc import QCExporter

from .utils import (
    table_is_not_empty,
    ensure_type,
)
from .constants import TABLES

logger = logging.getLogger(__name__)


# only use these indices: id_fn      id_id      id_lid     idx_chunk  idx_fn     idx_hs     roi0_fn

class IdtrackeraiExporter(SQLiteExporter, QCExporter, SleapExporter, DeepethogramExporter, OrientationExporter):

    def __init__(self, basedir, deepethogram_data, *args, framerate=None, **kwargs):
        """
        Arguments:
            basedir (str): Path to flyhostel experiment folder with raw recordings
            deepethogram_data (str): Path where the deg results are to be found
            framerate (int): Framerate of the output. For now, only None is supported
        """
        self._basedir = basedir
        self._deepethogram_data = deepethogram_data
        self._data_framerate = framerate
        super(IdtrackeraiExporter, self).__init__(*args, **kwargs)


    @property
    def data_framerate(self):
        if self._data_framerate is None:
            return self.framerate
        else:
            return self._data_framerate

    @property
    def step(self):
        return max(int(self.framerate / self.data_framerate), 1)

    # Init tables
    def init_data(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()

            cols_list = [
                "id INTEGER PRIMARY KEY AUTOINCREMENT",
                "frame_number int(11)",
                "in_frame_index int(2)",
                "x real(10)",
                "y real(10)",
                "fragment int(3)",
                "area int(11)",
                "modified int(1)",
                "class_name char(10)"
            ]

            formated_cols_names = ", ".join(cols_list)
            command = f"CREATE TABLE IF NOT EXISTS ROI_0 ({formated_cols_names})"
            if reset:
                cur.execute("DROP TABLE IF EXISTS ROI_0;")

            cur.execute(command)
            print("Creating indices for ROI_0 table")
            cur.execute("CREATE INDEX roi0_fn ON ROI_0 (frame_number);")
            # cur.execute("CREATE INDEX roi0_ifi ON ROI_0 (in_frame_index);")


   # IDENTITY
    def init_identity_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute("DROP TABLE IF EXISTS IDENTITY;")
            print("Creating indices for IDENTITY table")
            cur.execute("CREATE TABLE IF NOT EXISTS IDENTITY (id INTEGER PRIMARY KEY AUTOINCREMENT, frame_number int(11), in_frame_index int(2), local_identity int(2), identity int(2));")
            cur.execute("CREATE INDEX id_fn ON IDENTITY (frame_number);")
            cur.execute("CREATE INDEX id_id ON IDENTITY (identity);")
            cur.execute("CREATE INDEX id_lid ON IDENTITY (local_identity);")
            # cur.execute("CREATE INDEX id_ifi ON IDENTITY (in_frame_index);")


    # write
    def write_trajectory_and_identity_single_chunk(self, dbfile, chunk, **kwargs):

        blobs_collection = self.build_blobs_collection(chunk)
        video_path = self.build_video_object(chunk)

        if os.path.exists(blobs_collection):

            list_of_blobs = ListOfBlobs.load(blobs_collection)

            cwd = os.getcwd()
            os.chdir("idtrackerai")
            logger.debug("Loading %s", video_path)
            video_object = np.load(video_path, allow_pickle=True).item()
            os.chdir(cwd)

            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                cur = conn.cursor()

                start_end=(
                    video_object.episodes_start_end[0][0],
                    video_object.episodes_start_end[-1][-1]
                )
                blobs_in_video=list_of_blobs.blobs_in_video[start_end[0]:start_end[1]]

                r0_data =[]
                id_data=[]
                mod_data=set()

                cmd="SELECT local_identity, identity FROM CONCATENATION WHERE chunk = ?"
                cur.execute(cmd, (chunk, ))

                id_mapping = {
                    local_identity: identity
                    for local_identity, identity in cur.fetchall()
                }


                for frame_idx, blobs_in_frame in tqdm(enumerate(blobs_in_video), desc=f"Exporting trajectory/identity data for chunk {chunk}", unit="frame"):
                    if frame_idx % (self.step) == 0 or any((blob.modified for blob in blobs_in_frame)):
                        for blob in blobs_in_frame:
                            r0_args, id_args, mod_args = self.add_blob(blob, id_mapping=id_mapping, **kwargs)
                            if r0_args is not None:
                                r0_data.append(r0_args)
                            if id_args is not None:
                                id_data.append(id_args)
                            if mod_args is not None:
                                mod_data.add(mod_args)


                if r0_data:
                    command = "INSERT INTO ROI_0 (frame_number, in_frame_index, x, y, fragment, area, modified, class_name) VALUES(?, ?, ?, ?, ?, ?, ?, ?);"
                    print(command)
                    conn.executemany(command, r0_data)

                if id_data:
                    command = "INSERT INTO IDENTITY (frame_number, in_frame_index, local_identity, identity) VALUES(?, ?, ?, ?);"
                    print(command)
                    conn.executemany(command, id_data)

                if mod_data:
                    mod_data=list(mod_data)
                    command="INSERT INTO AI (frame_number, ai) VALUES (?, ?);"
                    print(command)
                    conn.executemany(command, mod_data)


        else:
            warnings.warn(f"{blobs_collection} not found")


    def write_trajectory_and_identity(self, dbfile, chunks, **kwargs):

        for chunk in chunks:
            before=time.time()
            self.write_trajectory_and_identity_single_chunk(dbfile, chunk=chunk, **kwargs)
            after=time.time()
            logger.debug("Exporting chunk %s in %s seconds", chunk, after-before)


    def add_blob(self, blob, id_mapping, w_trajectory=True, w_identity=True):

        if w_trajectory:
            r0_args=self.write_blob_trajectory(blob)
        else:
            r0_args=None

        if w_identity:
            id_args=self.write_blob_identity(blob, id_mapping)
        else:
            id_args=None

        mod_args=None
        if blob.modified:
            mod_args=(blob.frame_number, "YOLOv7")

        return r0_args, id_args, mod_args


    def write_blob_trajectory(self, blob):

        frame_number = ensure_type(blob.frame_number, "frame_number", int)
        in_frame_index = ensure_type(blob.in_frame_index, "in_frame_index", int)
        x_coord, y_coord = blob.centroid
        area = int(round(blob.area))
        modified = blob.modified
        fragment = blob.fragment_identifier
        if fragment is None:
            fragment=0


        if modified:
            annotation = getattr(blob, "_annotation", None)
            if annotation is None:
                class_name = "UNKNOWN"
            else:
                class_name = annotation.get("class", "UNKNOWN")
        else:
            class_name=None

        # the order here must match the one seen in the line containing 'command = "INSERT INTO ROI_0 ('
        args = (frame_number, in_frame_index, x_coord, y_coord, fragment, area, modified, class_name)
        return args

    def write_blob_identity(self, blob, id_mapping):
        frame_number = ensure_type(blob.frame_number, "frame_number", int)
        in_frame_index = ensure_type(blob.in_frame_index, "in_frame_index", int)

        local_identity = self._get_blob_local_identity(blob)
        try:
            identity_reference_to_ref_chunk = id_mapping[local_identity]
        except KeyError:
            identity_reference_to_ref_chunk=0

        args=(frame_number, in_frame_index, local_identity, identity_reference_to_ref_chunk)
        return args


    def _get_blob_local_identity(self, blob):
        local_identity = blob.final_identities[0]
        if local_identity is None:
            local_identity = 0

        local_identity=ensure_type(local_identity, "local_identity", int)
        return local_identity

    def _get_blob_identity_sqlite3(self, cur, blob, local_identity):

        chunk=blob.chunk
        chunk=ensure_type(chunk, "chunk", int)

        cmd="SELECT identity FROM CONCATENATION WHERE chunk = ? AND local_identity=?;"
        args=(chunk, local_identity)
        cur.execute(cmd, args)
        try:
            data = cur.fetchone()
            if data is None:
                cur.execute(cmd, (chunk, 0))
                data = cur.fetchall()
                if len(data) == 1:
                    # this happens if the local_identity is lost in the current chunk
                    # (i.e. it did not reach the end of the chunk)
                    # just return 0 then (data[0] contains a list with a 0 in it)
                    data=data[0]
                elif len(data) == 0:
                    # Explanation: BUG in idtrackerai_app.cli.utils.overlap where the last chunk is not in the concatenation table
                    data=[0]
                    # raise ValueError(f"Concatenation is corrupted in chunk {chunk}")
                else:
                    # more than one local_identity was lost in the current chunk
                    raise ValueError(f"Please validate chunk {chunk}")

            identity_reference_to_ref_chunk = int(data[0])

        except Exception as error:
            print(f"Query {cmd} with args {args} args failed")
            # raise error
            import ipdb; ipdb.set_trace()

        return identity_reference_to_ref_chunk


    def init_tables(self, dbfile, tables, behaviors=None, nodes=None, reset=True):

        for table in tables:
            if table not in TABLES:
                raise Exception(f"{table} is not one of {TABLES}. Do you have a typo?")

        super(IdtrackeraiExporter, self).init_tables(dbfile, tables, reset=reset)
        if "IDENTITY" in tables:
            self.init_identity_table(dbfile, reset=reset)

        if "ROI_0" in tables:
            self.init_data(dbfile, reset=reset)

        if "ORIENTATION" in tables:
            self.init_orientation_table(dbfile, reset=reset)

        if "BEHAVIORS" in tables:
            self.init_behaviors_table(dbfile, behaviors=behaviors, reset=reset)

        if "POSE" in tables:
            self.init_pose_table(dbfile, nodes=nodes, reset=reset)

        if "QC" in tables:
            self.init_qc_table(dbfile, reset=reset)
        

    def export(self, dbfile, chunks, tables, reset=True, behaviors=None, nodes=None, **kwargs):
        """
        Export datasets into single SQLite file

        Args:

            dbfile (str): Path to SQLite output file
            chunks (list): Chunks to be processed
            tables (list, str): List of tables to be exported
            mode (str)
            reset (bool):
            behaviors (list):
        """

        super(IdtrackeraiExporter, self).export(
            dbfile, chunks=chunks,
            tables=tables,
            behaviors=behaviors,
            nodes=nodes,
            reset=reset
        )

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
            if "ROI_0" in tables:
                print("ROI_0 done")

            if "IDENTITY" in tables:
                print("IDENTITY done")


        if "ORIENTATION" in tables:
            self.write_orientation_table(dbfile, chunks=chunks)
            print("ORIENTATION done")

        if "BEHAVIORS" in tables:
            self.write_behaviors_table(dbfile, behaviors=behaviors)
            print("BEHAVIORS done")

        if "POSE" in tables:
            self.write_pose_table(dbfile, chunks=chunks, nodes=nodes, **kwargs)

        if "QC" in tables:
            print("Writing QC")
            self.write_qc_table(dbfile, chunks=chunks)
            print("QC done")
