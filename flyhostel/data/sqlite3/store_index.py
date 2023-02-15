from abc import ABC
import sqlite3
from tqdm.auto import tqdm

class StoreIndexExporter(ABC):
    """Save the index of the flyhostel-imgstore recording in a FlyHostel SQLite file
    """

    _index_dbfile = None

    def init_index_table(self, dbfile, reset=True):
        """Initialize STORE_INDEX table
        """
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()

            if reset:
                cur.execute("DROP TABLE IF EXISTS STORE_INDEX;")
            cur.execute("CREATE TABLE IF NOT EXISTS STORE_INDEX (chunk int(3), frame_number int(11), frame_time int(11));")

    def write_index_table(self, dbfile):
        """Populate STORE_INDEX table
        """
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
