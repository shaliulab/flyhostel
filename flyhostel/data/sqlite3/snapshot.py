from abc import ABC
import sqlite3
import tempfile
import os.path

import cv2

from .utils import serialize_arr


class SnapshotExporter(ABC):

    _basedir=None

    def __init__(self, *args, **kwargs):
        self._temp_path = tempfile.mktemp(prefix="flyhostel_", suffix=".jpg")
        self._index_dbfile = os.path.join(self._basedir, "index.db")
        super(SnapshotExporter, self).__init__(*args, **kwargs)


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
                        raise ValueError(f"Cannot save chunk {chunk} snapshot. {snapshot_path} does not exist")
                    arr=cv2.imread(snapshot_path)
                    bstring = serialize_arr(arr, self._temp_path)
                    cur.execute(
                        "INSERT INTO IMG_SNAPSHOTS (frame_number, img) VALUES (?, ?)",
                        (frame_number, bstring)
                    )