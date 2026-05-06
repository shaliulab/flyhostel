import toml
import os.path
import sqlite3
import logging
from abc import ABC
logger=logging.getLogger(__name__)

def parse_landmark_line(line):
    tokens=line.split(" ")
    shape_name=tokens[1]
    coords=" ".join(tokens[2:])

    return shape_name, coords

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
            config=toml.load(handle)

        roi_list=config["roi_list"]
        notch_count=config.get("notch_count", None)
        if notch_count is None:
            logger.warning("I will assume there are 4 notches")
            notch_count=4

        landmarks=[parse_landmark_line(line) for line in roi_list]
            
        landmark_types={"Ellipse": "food", "Polygon": "notch"}
        landmarks2=[]
        count=0
        for shape_name, coords in landmarks:
            landmark_name=landmark_types[shape_name]
            if count == notch_count and shape_name == "Polygon":
                landmark_name="Polygon"

            if landmark_name=="notch":
                count+=1

            landmarks2.append((landmark_name, coords))

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            conn.executemany(
                "INSERT INTO LANDMARKS (shape, specification) VALUES (?, ?);",
                landmarks2
            )