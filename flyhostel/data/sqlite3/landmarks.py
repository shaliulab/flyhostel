import toml
import os.path
import sqlite3
from abc import ABC

class LandmarksExporter(ABC):
    _basedir=None

    def init_landmarks_table(self, dbfile, reset=True):
        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            cur = conn.cursor()
            if reset:
                print("Dropping LANDMARKS")
                cur.execute("DROP TABLE IF EXISTS LANDMARKS;")
            cur.execute("CREATE TABLE IF NOT EXISTS LANDMARKS (id INTEGER PRIMARY KEY AUTOINCREMENT, shape char(50), specification char(100));")

    def write_landmarks_table(self, dbfile, chunks):
        
        landmarks_file=os.path.join(self._basedir, "landmarks.toml")
        if not os.path.exists(landmarks_file):
            raise FileNotFoundError(f"Please generate {landmarks_file}")
        
        with open(landmarks_file, "r") as handle:
            roi_list=toml.load(handle)["roi_list"]
            landmarks=[(line.split(" ")[1], " ".join(line.split(" ")[2:])) for line in roi_list]
            
        landmark_types={"Ellipse": "food", "Polygon": "notch"}
        landmarks=[(landmark_types[landmark[0]], landmark[1]) for landmark in landmarks]

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            conn.executemany(
                "INSERT INTO LANDMARKS (shape, specification) VALUES (?, ?);",
                landmarks
            )