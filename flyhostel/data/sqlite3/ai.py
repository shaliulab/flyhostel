
from abc import ABC
import sqlite3
import pickle
import os.path
import glob
import numpy as np

class AIExporter(ABC):

    """Keep track of which frames contain an intervention of an AI, and if so, which one
    """

    _basedir = None

    def init_ai_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute("DROP TABLE IF EXISTS AI;")
            cur.execute("CREATE TABLE IF NOT EXISTS AI (frame_number int(11), ai char(30));")


    def write_ai_table(self, dbfile, chunks=None):

        if chunks is None:
            pickle_files=sorted(glob.glob(
                os.path.join(self._basedir, "idtrackerai", "session_*", "preprocessing", "ai.pkl")
            ))
        else:
            pickle_files = []
            for chunk in chunks:
                path=os.path.join(self._basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "ai.pkl")
                if os.path.exists(path):
                    pickle_files.append(path)

        if pickle_files:
            data=[]
            for file in pickle_files:
                with open(file, "rb") as filehandle:
                    ai_mods = pickle.load(filehandle)
                    frames=ai_mods["success"]

                for frame_number in frames:
                    data.append((frame_number, "YOLOv7"))

            if data:
                with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                    conn.executemany("INSERT INTO AI (frame_number, ai) VALUES (?, ?);", data)

        if chunks is None:
            fragment_files=sorted(glob.glob(
                os.path.join(self._basedir, "idtrackerai", "session_*", "preprocessing", "fragments.npy")
            ))
        else:
            fragment_files = []
            for chunk in chunks:
                path = os.path.join(self._basedir, "idtrackerai", f"session_{str(chunk).zfill(6)}", "preprocessing", "fragments.npy")
                if os.path.exists(path):
                    fragment_files.append(path)

        if fragment_files:

            for file in fragment_files:
                frames = []
                list_of_fragments = np.load(file, allow_pickle=True).item()

                # maybe this would work too
                # has_been_accumulated=video_object.has_protocol1_finished

                has_been_accumulated=any((fragment.accumulation_step == 0 for fragment in list_of_fragments.fragments))
                if has_been_accumulated:
                    for fragment in list_of_fragments.fragments:
                        if (fragment.accumulation_step is None or fragment.accumulation_step > 1) and fragment.assigned_identities[0] != 0:
                            frames.append(fragment.start_end[0])

                    for frame_number in frames:
                            cur.execute("INSERT INTO AI (frame_number, ai) VALUES (?, ?);", (frame_number, "idtrackerai"))
