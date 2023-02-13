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
            cur.execute("CREATE TABLE IF NOT EXISTS BEHAVIORS (frame_number int(11), in_frame_index int(2), behavior char(100), probability float(5));")


    def write_behaviors_table(self, *args, **kwargs):

        for in_frame_index in range(self.number_of_animals):
            self.write_behaviors_table_single_blob(*args, in_frame_index, **kwargs)

    def write_behaviors_table_single_blob(self, dbfile, in_frame_index, behaviors=None,chunks=None):

        if self._deepethogram_data is None:
            raise ValueError("Please pass a deepethogram data folder")


        prefix = "_".join(self._basedir.split(os.path.sep)[-3:])
        reader = H5Reader.from_outputs(
            data_dir=self._deepethogram_data, prefix=prefix,
            in_frame_index=in_frame_index,
            fps=self._store_metadata["framerate"]
        )

        if behaviors is None:
            behaviors=reader.class_names


        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            with sqlite3.connect(self._index_dbfile, check_same_thread=False) as index_db:

                index_db_cur = index_db.cursor()

                for behavior_idx, behavior in enumerate(behaviors):

                    chunks_avail, p_array = reader.load(behavior, n_jobs=self._n_jobs)
                    progress_bar=tqdm(
                        total=len(chunks_avail),
                        desc=f"Loading {behavior} instances for blob index {in_frame_index}",
                        position=behavior_idx,
                        unit="chunk"
                    )

                    for chunk_idx, chunk in enumerate(chunks_avail):
                        if chunks is not None and chunk not in chunks:
                            continue
                        index_db_cur.execute("SELECT frame_number FROM frames WHERE chunk = ?;", (chunk, ))
                        frame_numbers = [int(x[0]) for x in index_db_cur.fetchall()]
                        assert p_array[chunk_idx].shape[0] == len(frame_numbers)

                        for frame_number_idx, frame_number in enumerate(frame_numbers):
                            args=(frame_number, in_frame_index, behavior, p_array[chunk_idx][frame_number_idx].item())
                            cur.execute("INSERT INTO BEHAVIORS (frame_number, in_frame_index, behavior, probability) VALUES (?, ?, ?, ?);", args)

                        progress_bar.update(1)
