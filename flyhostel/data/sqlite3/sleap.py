from abc import ABC
import glob
import os.path
import sqlite3
import h5py


from tqdm.auto import tqdm
import numpy as np
import pandas as pd

from flyhostel.data.sqlite3.utils import parse_experiment_properties
from flyhostel.data.deepethogram.video import build_key

class SleapExporter(ABC):
    _basedir = None # 

    def __init__(self, *args, **kwargs):
        super(SleapExporter, self).__init__(*args, **kwargs)

    def init_pose_table(self, dbfile, nodes=None, reset=True):
        if nodes is None:
            return
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()

            if reset:
                print("DROP TABLE IF EXISTS POSE;")
                cur.execute("DROP TABLE IF EXISTS POSE;")
            cur.execute("CREATE TABLE IF NOT EXISTS POSE (id INTEGER PRIMARY KEY AUTOINCREMENT, frame_number int(11), local_identity int(3), node char(20), visible int(1), x int(4), y int(4), score float(5));")
            print(f"Creating indices for POSE table")
            cur.execute(f"CREATE INDEX pose_fn ON POSE (frame_number);")
            cur.execute(f"CREATE INDEX pose_lid ON POSE (local_identity);")


    def write_pose_table(self, dbfile, chunks, nodes=True, **kwargs):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            for chunk in tqdm(chunks):
                self._write_pose_table(conn, dbfile, chunk=chunk, nodes=nodes, **kwargs)


    def _write_pose_table(self, conn, dbfile, chunk, nodes, **kwargs):
        return self._write_pose_table_from_h5_file(conn, dbfile, chunk, nodes, **kwargs)

    def _write_pose_table_from_h5_file(self, conn, dbfile, chunk, nodes, local_identities=None):
        _, (flyhostel_id, number_of_animals, date_time) = parse_experiment_properties(self._basedir)
        # key = build_key(flyhostel_id, number_of_animals, date_time, chunk, local_identity=None)

        cur = conn.cursor()
        cur.execute("SELECT value FROM METADATA where field = 'chunksize';")
        chunksize=int(float(cur.fetchone()[0]))

        if local_identities is None:
            if number_of_animals > 1:
                local_identities = list(range(1, number_of_animals+1))
            else:
                local_identities=[0]
            
        for local_identity in local_identities:
            csv_file_pattern = os.path.join(
                self._basedir, "flyhostel", "single_animal",
                str(local_identity).zfill(3),
                f"{str(chunk).zfill(6)}.*.predictions.h5"
            )
            hits=glob.glob(csv_file_pattern)
            assert len(hits) == 1, f'No hits for {csv_file_pattern}'
            h5_file = hits[0]

            with h5py.File(h5_file) as filehandle:
                # extract pose and score
                nodes_index = [node.decode() for node in filehandle["node_names"][:]]
                
                for node in nodes:
                    node_index=nodes_index.index(node)
                    pose=filehandle["tracks"][0, :, node_index, :]
                    score=filehandle["point_scores"][0, node_index, :]
                    keep=~np.isnan(pose).all(axis=0)

                    if keep.sum()==0:
                        keep[0]=True
                        visible=0
                    else:
                        visible=1

                    pose=pose[:, keep]
                    score=score[keep]

                    frame_indices=np.where(keep)[0].tolist()

                    data=[]
                    for i, frame_idx in enumerate(frame_indices):
                        command = "INSERT INTO POSE (frame_number, local_identity, node, visible, x, y, score) VALUES(?, ?, ?, ?, ?, ?, ?);"
                        frame_number=frame_idx+chunk*chunksize
                        data.append((frame_number, local_identity, node, visible, pose[0, i], pose[1, i], score[i]))
                    conn.executemany(command, data)



    def _write_pose_table_from_csv(self, conn, dbfile, chunk, nodes):
        _, (flyhostel_id, number_of_animals, date_time) = parse_experiment_properties(self._basedir)
        # key = build_key(flyhostel_id, number_of_animals, date_time, chunk, local_identity=None)

        cur = conn.cursor()
        cur.execute("SELECT value FROM METADATA where field = 'chunksize';")
        chunksize=int(float(cur.fetchone()[0]))

        if number_of_animals > 1:
            local_identities = list(range(1, number_of_animals+1))
        else:
            local_identities=[0]
            

        for local_identity in local_identities:
            csv_file_pattern = os.path.join(
                self._basedir, "flyhostel", "single_animal",
                str(local_identity).zfill(3),
                f"{str(chunk).zfill(6)}.*.predictions.csv"
            )
            hits=glob.glob(csv_file_pattern)
            assert len(hits) == 1, f'No hits for {csv_file_pattern}'
            csv_file = hits[0]

            pose = pd.read_csv(csv_file)
            pose=pose.loc[pose["node"].isin(nodes)]

            data=[]
            for _, row in pose.iterrows():
                command = "INSERT INTO POSE (frame_number, local_identity, node, x, y, score) VALUES(?, ?, ?, ?, ?);"
                frame_number=row["frame_idx"]+chunk*chunksize
                
                data.append((frame_number, row["local_identity"], row["node"], row["x"], row["y"], row["score"]))
            conn.executemany(command, data)

