from abc import ABC
import os.path
import sqlite3

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

    def init_behaviors_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute("DROP TABLE IF EXISTS BEHAVIORS;")
            cur.execute("CREATE TABLE IF NOT EXISTS BEHAVIORS (frame_number int(11), local_identity int(3), behavior char(100), probability float(5));")
            print("Creating indices for BEHAVIORS table")
            cur.execute("CREATE INDEX behav_fn ON BEHAVIORS (frame_number);")
            cur.execute("CREATE INDEX behav_lid ON BEHAVIORS (local_identity);")

    def write_behaviors_table(self, *args, **kwargs):
        if self.number_of_animals == 1:
            self.write_behaviors_table_single_blob(*args, 0, **kwargs)
        else:
            for local_identity in range(1, self.number_of_animals+1):
                self.write_behaviors_table_single_blob(*args, local_identity, **kwargs)

    def write_behaviors_table_single_blob(self, dbfile, local_identity, behaviors=None,chunks=None):

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
            with sqlite3.connect(self._index_dbfile, check_same_thread=False) as index_db:

                index_db_cur = index_db.cursor()
                progress_bar=tqdm(
                    total=len(behaviors),
                    desc=f"Exporting behavior data for local_identity {local_identity} for all chunks",
                    unit="behavior"
                )
                for behavior_idx, behavior in enumerate(behaviors):
                    chunks_avail, p_list = reader.load(behavior=behavior, n_jobs=self._n_jobs)

                    data=[]
                    for chunk_idx, chunk in enumerate(chunks_avail):
                        if chunks is not None and chunk not in chunks:
                            continue
                        index_db_cur.execute("SELECT frame_number FROM frames WHERE chunk = ?;", (chunk, ))
                        frame_numbers = [int(x[0]) for x in index_db_cur.fetchall()]

                        if p_list[chunk_idx].shape[0] != len(frame_numbers):
                            raise Exception(f"""
                                {p_list[chunk_idx].shape[0]} frames have an annotated behavior, but the chunk has {len(frame_numbers)} frames.
                                Has deepethogram finished running there?
                                """
                            )


                        for frame_number_idx, frame_number in enumerate(frame_numbers):
                            data.append((frame_number, local_identity, behavior, p_list[chunk_idx][frame_number_idx].item()))

                    if data:
                        conn.executemany("INSERT INTO BEHAVIORS (frame_number, local_identity, behavior, probability) VALUES (?, ?, ?, ?);", data)

                progress_bar.update(1)
