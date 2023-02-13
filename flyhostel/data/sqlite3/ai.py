
from abc import ABC
import sqlite3
import glob
import pickle
import os.path

class AIExporter(ABC):

    """
    """

    _basedir = None

    def init_ai_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                cur.execute("DROP TABLE IF EXISTS AI;")
            cur.execute("CREATE TABLE IF NOT EXISTS AI (frame_number int(11), ai int(2));")


    def write_ai_table(self, dbfile):

        pickle_files = sorted(glob.glob(os.path.join(self._basedir, "idtrackerai", "session_*", "preprocessing", "ai.pkl")))
        if pickle_files:
            with sqlite3.connect(dbfile, check_same_thread=False) as conn:
                cur = conn.cursor()

                for file in pickle_files:
                    with open(file, "rb") as filehandle:
                        ai_mods = pickle.load(filehandle)
                        frames=ai_mods["success"]

                    for frame_number in frames:
                        cur.execute("INSERT INTO AI (frame_number, ai) VALUES (?, ?);", (frame_number, "YOLOv7"))
