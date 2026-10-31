"""
LANDMARKS table of the dbfile (flyhostel.data.sqlite3.landmarks) and the food without its excluded areas
(flyhostel.data.pose.landmarks)
"""
import logging
import sqlite3

import numpy as np
import pandas as pd
import pytest

from flyhostel.data.pose.landmarks import distance_from_points_to_shape, scale_shape, subtract_shapes
from flyhostel.data.sqlite3.landmarks import LandmarksExporter

DATE_TIME = "2026-10-28_16-00-00"
ELLIPSE = "{'center': [475, 524], 'axes': [430, 449], 'angle': 130}"
NEGATIVE = "[[369.8, 32.1], [408.3, 143.5], [569.0, 137.1], [567.9, 27.8]]"
ARENA_TOML = f"""
number_of_animals = 2
roi_list = [
    "+ Ellipse {ELLIPSE}",
    '- Polygon {NEGATIVE}',
]
"""
SPLITTER = "splitter = [[503.7, 968.3], [498.3, 30.0]]\n"
NORMAL_LANDMARKS = """notch_count = 1
roi_list = [
    "+ Ellipse {'center': [1, 2], 'axes': [3, 4], 'angle': 0}",
    '+ Polygon [[0, 0], [1, 0], [1, 1], [0, 1]]',
]
"""


class Exporter(LandmarksExporter):
    def __init__(self, basedir, arena):
        self._basedir = str(basedir)
        self._date_time = DATE_TIME
        self._arena = arena


def write(tmp_path, landmarks, main=None):
    (tmp_path / "landmarks.toml").write_text(landmarks)
    if main is not None:
        (tmp_path / f"{DATE_TIME}.toml").write_text(main)


def landmarks_table(tmp_path, arena):
    dbfile = str(tmp_path / "test.db")
    exporter = Exporter(tmp_path, arena)
    exporter.init_landmarks_table(dbfile)
    exporter.write_landmarks_table(dbfile, chunks=[0])
    with sqlite3.connect(dbfile) as conn:
        return pd.read_sql("SELECT shape, specification FROM LANDMARKS ORDER BY id", conn)


@pytest.mark.parametrize("roi_list", ["", "roi_list = []\n", ARENA_TOML], ids=["missing", "empty", "copy"])
def test_arena_food_is_the_arena_of_the_main_toml(tmp_path, roi_list, caplog):
    write(tmp_path, roi_list + SPLITTER, ARENA_TOML)
    with caplog.at_level(logging.WARNING):
        table = landmarks_table(tmp_path, arena=2)
    assert table.values.tolist() == [
        ["food", ELLIPSE],
        ["food_excluded", NEGATIVE],
        ["splitter", "[[503.7, 968.3], [498.3, 30.0]]"],
    ]
    assert "notches" not in caplog.text


def test_arena_without_splitter(tmp_path):
    write(tmp_path, ARENA_TOML, ARENA_TOML)
    assert landmarks_table(tmp_path, arena=1)["shape"].tolist() == ["food", "food_excluded"]


def test_arena_without_main_toml_fails(tmp_path):
    write(tmp_path, SPLITTER)
    with pytest.raises(FileNotFoundError, match=f"{DATE_TIME}.toml"):
        landmarks_table(tmp_path, arena=2)


def test_arena_with_a_positive_polygon_fails(tmp_path):
    write(tmp_path, "", "roi_list = ['+ Polygon [[0, 0], [1, 0], [1, 1]]']\n")
    with pytest.raises(ValueError, match="notches"):
        landmarks_table(tmp_path, arena=2)


def test_normal_experiment_needs_roi_list(tmp_path):
    write(tmp_path, "number_of_animals = 1\n", ARENA_TOML)
    with pytest.raises(KeyError, match="roi_list"):
        landmarks_table(tmp_path, arena=None)


def test_normal_experiment_has_food_and_notches_of_landmarks_toml(tmp_path):
    write(tmp_path, NORMAL_LANDMARKS, ARENA_TOML)
    assert landmarks_table(tmp_path, arena=None)["shape"].tolist() == ["food", "notch"]


def test_food_without_excluded_area():
    food = {"center": [0, 0], "axes": [10, 10], "angle": 0}
    hole = [[-2, -2], [2, -2], [2, 2], [-2, 2]]
    points = np.array([[0.0, 0.0], [5.0, 0.0], [20.0, 0.0], [np.nan, np.nan]])
    d = subtract_shapes(distance_from_points_to_shape(points, food), points, [hole])
    # the center is in the hole: outside the food, 2 px from its edge
    assert d[0] == pytest.approx(2)
    # between the hole (3 px) and the ellipse (5 px): inside, 3 px from the nearest edge
    assert d[1] == pytest.approx(-3)
    # outside the ellipse
    assert d[2] == pytest.approx(10)
    assert np.isnan(d[3])


def test_scale_shape():
    assert scale_shape({"center": [1, 2], "axes": [3, 4], "angle": 5}, 2) == {"center": [2, 4], "axes": [6, 8], "angle": 5}
    assert scale_shape([[1, 2], [3, 4]], 0.5) == [[0.5, 1.0], [1.5, 2.0]]
