"""
LANDMARKS table of a FlyHostel dbfile: one row per landmark, with columns shape and specification

    * food: an ellipse that delineates the food ("{'center': [x, y], 'axes': [a, b], 'angle': deg}")
    * food_excluded: an area removed from the food (same specification as food or notch).
      The food is the food ellipses without the food_excluded areas
    * notch: a tetragon that delineates a notch in the glass above the arena ("[[x, y], ...]")
    * Polygon: polygons of landmarks.toml beyond the notch_count notches
    * splitter: the line between 2 flies separated by a splitter (splitter.nf), "[[x1, y1], [x2, y2]]"

Normal experiments (FlyHostelN/NX/DATE_TIME): food (Ellipse) and notches (positive Polygon)
come from the roi_list of landmarks.toml

Experiments of one arena (FlyHostelN/NX/DATE_TIME/arenas/K): the food is everywhere in the arena,
so it is the arena itself, taken from the roi_list of DATE_TIME.toml (the ROI idtrackerai tracks in,
also in ROI_MAP): its positive Ellipse is the food and its negative ROIs are food_excluded.
Arenas have no notches. The roi_list of landmarks.toml is a copy of that of DATE_TIME.toml (not read here)

The splitter comes from landmarks.toml in both
"""
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


def arena_landmarks(toml_path):
    """(shape, specification) of the food of an arena, from the roi_list of its DATE_TIME.toml"""
    from flyhostel.data.roi_map import load_toml_rois

    if not os.path.exists(toml_path):
        raise FileNotFoundError(f"{toml_path} not found: the food of the arena cannot be written to LANDMARKS")
    rois=load_toml_rois(toml_path)
    landmarks=[]
    for sign, shape, spec in rois:
        if sign == 1 and shape != "Ellipse":
            raise ValueError(
                f"{toml_path} has a positive {shape}: the food of an arena is a positive Ellipse"
                " (positive Polygons are notches, which arenas do not have)"
            )
        landmarks.append(("food" if sign == 1 else "food_excluded", str(spec)))
    if not any(shape == "food" for shape, _ in landmarks):
        raise ValueError(f"{toml_path} has no positive Ellipse: the food of the arena cannot be written to LANDMARKS")
    return landmarks


class LandmarksExporter(ABC):
    _basedir=None
    _date_time=None
    _arena=None

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

        if self._arena is None:
            landmarks=self._food_and_notches(config, landmarks_file)
        else:
            landmarks=arena_landmarks(os.path.join(self._basedir, f"{self._date_time}.toml"))

        if config.get("splitter") is not None:
            landmarks.append(("splitter", str(config["splitter"])))

        with sqlite3.connect(dbfile, check_same_thread=False) as conn:
            conn.executemany(
                "INSERT INTO LANDMARKS (shape, specification) VALUES (?, ?);",
                landmarks
            )

    @staticmethod
    def _food_and_notches(config, landmarks_file):
        """(shape, specification) of the food and notches in the roi_list of landmarks.toml of a normal experiment"""
        if "roi_list" not in config:
            raise KeyError(f"{landmarks_file} has no roi_list (food and notches)")

        notch_count=config.get("notch_count", None)
        if notch_count is None:
            logger.warning("I will assume there are 4 notches")
            notch_count=4

        landmark_types={"Ellipse": "food", "Polygon": "notch"}
        landmarks=[]
        count=0
        for shape_name, coords in [parse_landmark_line(line) for line in config["roi_list"]]:
            landmark_name=landmark_types[shape_name]
            if count == notch_count and shape_name == "Polygon":
                landmark_name="Polygon"

            if landmark_name=="notch":
                count+=1

            landmarks.append((landmark_name, coords))
        return landmarks
