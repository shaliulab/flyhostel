from abc import ABC
import os.path
import sqlite3

from tqdm.auto import tqdm
import pandas as pd

from flyhostel.data.sqlite3.utils import parse_experiment_properties
from flyhostel.data.deepethogram.video import build_key

class SleapExporter(ABC):
    _basedir = None

    def __init__(self, sleap_data, *args, **kwargs):
        self._sleap_data = sleap_data
        super(SleapExporter, self).__init__(*args, **kwargs)


    def init_pose_table(self, dbfile, nodes=None, reset=True):
        if nodes is None:
            return
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()

            if reset:
                print("DROP TABLE IF EXISTS POSE;")
                cur.execute("DROP TABLE IF EXISTS POSE;")
            cur.execute("CREATE TABLE IF NOT EXISTS POSE (frame_number int(11), local_identity int(3), node char(20), visible int(1), x int(4), y int(4), score float(5));")
            print(f"Creating indices for POSE table")
            cur.execute(f"CREATE INDEX pose_fn ON POSE (frame_number);")
            cur.execute(f"CREATE INDEX pose_lid ON POSE (local_identity);")

    def write_pose_table(self, dbfile, chunks, nodes=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            for chunk in tqdm(chunks):
                self._write_pose_table(conn, dbfile, chunk=chunk, nodes=nodes)
    
    def _write_pose_table(self, conn, dbfile, chunk, nodes):
        _, (flyhostel_id, number_of_animals, date_time) = parse_experiment_properties(self._basedir)
        key = build_key(flyhostel_id, number_of_animals, date_time, chunk, local_identity=None)
        csv_file = os.path.join(
            self._basedir, "sleap", key + ".csv"
        )
        pose = pd.read_csv(csv_file)
        pose=pose.loc[pose["node"].isin(nodes)]
        cur = conn.cursor()
        cur.execute("SELECT value FROM METADATA where field = 'chunksize';")
        chunksize=int(float(cur.fetchone()[0]))

        data=[]
        for i, row in pose.iterrows():
            command = "INSERT INTO POSE (frame_number, local_identity, node, visible, x, y, score) VALUES(?, ?, ?, ?, ?, ?);"
            frame_number=row["frame_idx"]+chunk*chunksize
            
            data.append((frame_number, row["local_identity"], row["node"], row["visible"], row["x"], row["y"], row["score"]))
        conn.executemany(command, data)

