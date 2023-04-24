from abc import ABC
import sqlite3
import tempfile
import os.path
import warnings
import cv2

from .utils import serialize_arr


class SnapshotExporter(ABC):
    """Generate the IMG_SNAPSHOTS table of a FlyHostel SQLite dataset
    """

    _basedir=None
    _must_have_snapshot = None

    def __init__(self, *args, **kwargs):
        self._temp_path = tempfile.mktemp(prefix="flyhostel_", suffix=".jpg")
        self._index_dbfile = os.path.join(self._basedir, "index.db")
        super(SnapshotExporter, self).__init__(*args, **kwargs)


    def init_snapshot_table(self, dbfile, reset=True):
        """Initialize the IMG_SNAPSHOTS table of a FlyHostel SQLite dataset
        """

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()

            if reset:
                cur.execute("DROP TABLE IF EXISTS IMG_SNAPSHOTS;")
            cur.execute("CREATE TABLE IF NOT EXISTS IMG_SNAPSHOTS (frame_number int(11), img longblob)")

    def write_snapshot_table(self, dbfile, chunks):
        """Populate the IMG_SNAPSHOTS table of a FlyHostel SQLite dataset
        """
        data=[]

        with sqlite3.connect(self._index_dbfile, check_same_thread=False) as index_db:
            index_cursor = index_db.cursor()


            for chunk in chunks:
                index_cursor.execute(f"SELECT frame_number FROM frames WHERE chunk = {chunk} AND frame_idx = 0;")
                try:
                    frame_number = int(index_cursor.fetchone()[0])
                # this happens if fetchone() returns None (is not indexable)
                except TypeError:
                    warnings.warn(f"Cannot find first frame number of chunk {chunk}")
                    continue

                snapshot_path = os.path.join(self._basedir, f"{str(chunk).zfill(6)}.png")
                if not os.path.exists(snapshot_path):
                    message = f"Cannot save chunk {chunk} snapshot. {snapshot_path} does not exist"
                    if self._must_have_snapshot:
                        raise ValueError(message)
                    else:
                        warnings.warn(message)
                        continue

                arr=cv2.imread(snapshot_path)
                bstring = serialize_arr(arr, self._temp_path)
                data.append((frame_number, bstring))

            if data:
                with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                    conn.executemany("INSERT INTO IMG_SNAPSHOTS (frame_number, img) VALUES (?, ?)", data)
