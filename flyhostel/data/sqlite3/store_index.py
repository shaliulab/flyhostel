from abc import ABC
import sqlite3
from tqdm.auto import tqdm

class StoreIndexExporter(ABC):
    """Save the index of the flyhostel-imgstore recording in a FlyHostel SQLite file
    """

    _index_dbfile = None
    framerate = None

    def init_index_table(self, dbfile, reset=True):
        """Initialize STORE_INDEX table
        """
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                print("DROP TABLE IF EXISTS STORE_INDEX;")
                cur.execute("DROP TABLE IF EXISTS STORE_INDEX;")
                print("Done")
            cur.execute("CREATE TABLE IF NOT EXISTS STORE_INDEX (chunk int(3), frame_number int(11), frame_time int(11), half_second int(2));")
            print("Creating indices for STORE_INDEX table")
            cur.execute("CREATE INDEX idx_fn ON STORE_INDEX (frame_number);")
            cur.execute("CREATE INDEX idx_hs ON STORE_INDEX (half_second);")
            cur.execute("CREATE INDEX idx_chunk ON STORE_INDEX (chunk);")


    def write_index_table(self, dbfile):
        """Populate STORE_INDEX table
        """
        with sqlite3.connect(self._index_dbfile, check_same_thread=False) as index_db:
            index_db_cur = index_db.cursor()

            index_db_cur.execute("SELECT COUNT(*) FROM frames;")
            count = int(index_db_cur.fetchone()[0])

            index_db_cur.execute(f"SELECT chunk, frame_number, frame_time, frame_number % {self.framerate // 2} == 0 FROM frames;")
            pb=tqdm(total=count, desc="Writing index", unit="frame")
            data=[]

            for chunk, frame_number, frame_time, half_second in index_db_cur:
                data.append((chunk, frame_number, frame_time, half_second))
                pb.update(1)

            if data:
                with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                    conn.executemany(
                        "INSERT INTO STORE_INDEX (chunk, frame_number, frame_time, half_second) VALUES (?, ?, ?, ?);",
                        data
                    )

