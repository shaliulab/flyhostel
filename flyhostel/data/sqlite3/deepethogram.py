from abc import ABC
import os.path
import sqlite3
import time

from tqdm.auto import tqdm
from flyhostel.data.deepethogram import H5Reader

class DeepethogramExporter(ABC):

    number_of_animals=None
    _deepethogram_data=None
    _basedir=None
    _index_dbfile=None
    _store_metadata=None


    def __init__(self, *args, n_jobs=1, **kwargs):
        self._n_jobs=n_jobs
        super().__init__(*args, **kwargs)

    def init_behaviors_table(self, dbfile, behaviors=None, reset=True):
        if behaviors is None:
            return

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()

            for behavior in behaviors:
                behavior = behavior.upper()
                if reset:
                    print(f"DROP TABLE IF EXISTS {behavior};")
                    before=time.time()
                    cur.execute(f"DROP TABLE IF EXISTS {behavior};")
                    after=time.time()
                    print(f"Done in {after-before} seconds")

                cur.execute(f"CREATE TABLE IF NOT EXISTS {behavior} (frame_number int(11), local_identity int(3), probability float(5));")
                print(f"Creating indices for {behavior} table")
                cur.execute(f"CREATE INDEX {behavior.lower()}_fn ON {behavior} (frame_number);")
                cur.execute(f"CREATE INDEX {behavior.lower()}_lid ON {behavior} (local_identity);")

    def write_behaviors_table(self, *args, behaviors=None, **kwargs):
        if self.number_of_animals == 1:
            self.write_behaviors_table_single_blob(*args, local_identity=0, behaviors=behaviors, **kwargs)
        else:
            for local_identity in range(1, self.number_of_animals+1):
                self.write_behaviors_table_single_blob(
                    *args,
                    local_identity=local_identity,
                    behaviors=behaviors,
                    **kwargs
                )

    def write_behaviors_table_single_blob(self, dbfile, local_identity, behaviors=None, chunks=None):

        if self._deepethogram_data is None:
            raise ValueError("Please pass a deepethogram data folder")


        prefix = "_".join(self._basedir.split(os.path.sep)[-3:])
        if local_identity==0:
            deepethogram_identity=1
        else:
            deepethogram_identity=local_identity

        reader = H5Reader.from_outputs(
            data_dir=self._deepethogram_data, prefix=prefix,
            local_identity=deepethogram_identity,
            frequency=self._store_metadata["framerate"]
        )

        if behaviors is None:
            behaviors=reader.class_names

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur=conn.cursor()
            with sqlite3.connect(self._index_dbfile, check_same_thread=False) as index_db:

                index_db_cur = index_db.cursor()
                for behavior_idx, behavior in enumerate(behaviors):
                    chunks_avail, p_list = reader.load(behavior=behavior, n_jobs=self._n_jobs)
                    behavior=behavior.upper()

                    data=[]
                    progress_bar=tqdm(
                        total=len(chunks_avail),
                        desc=f"Exporting {behavior} data for local_identity {local_identity} for all chunks",
                        unit="chunk"
                    )

                    for chunk_idx, chunk in enumerate(chunks_avail):
                        if chunks is not None and chunk not in chunks:
                            continue

                        # print("SELECT COUNT(*) FROM frames WHERE chunk = ?;")
                        # before=time.time()
                        index_db_cur.execute("SELECT COUNT(*) FROM frames WHERE chunk = ?;", (chunk, ))
                        n_frames=int(index_db_cur.fetchone()[0])
                        # after=time.time()
                        # print(f"Done in {after-before} seconds")


                        if p_list[chunk_idx].shape[0] != n_frames:
                            raise ValueError(f"""
                                {p_list[chunk_idx].shape[0]} frames have an annotated behavior, but the chunk has {n_frames} frames.
                                Has deepethogram finished running there?
                                """
                            )

                        cur.execute("SELECT frame_number FROM STORE_INDEX WHERE chunk = ? LIMIT 1;", (chunk, ))
                        frame_number=last_fn=cur.fetchone()[0]
                        frame_number_idx = 0
                        cur.execute("SELECT frame_number FROM STORE_INDEX WHERE chunk = ? AND half_second = 1;", (chunk, ))

                        for row in cur.fetchall():
                            frame_number=row[0]
                            frame_number_idx += (frame_number - last_fn)
                            data.append((frame_number, local_identity, p_list[chunk_idx][frame_number_idx].item()))
                            last_fn = frame_number

                        progress_bar.update(1)


                    if data:
                        print(len(data))
                        cmd=f"INSERT INTO {behavior} (frame_number, local_identity, probability) VALUES (?, ?, ?);"
                        print(cmd)
                        before=time.time()
                        conn.executemany(cmd, data)
                        after=time.time()
                        print(f"Done in {after-before} seconds")

