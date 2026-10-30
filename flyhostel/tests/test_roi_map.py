"""
Tests for arenas made of an ellipse with negative polygons (areas excluded from the arena) in ROI_MAP

The ROIs are those of FlyHostel4/8X/2026-10-26_18-00-00/arenas/2:
DATE_TIME.toml has an ellipse and two negative rectangles, and DATE_TIME.conf was made from it with conf-convert
"""
import sqlite3

import numpy as np
import pytest

from flyhostel.data.roi_map import (
    ROIMapMismatchError, load_roi_map, parse_conf_rois, draw_roi_mask, verify_roi_map,
)
from flyhostel.data.pose.loaders.roi import arena_calib
from flyhostel.data.sqlite3 import IdtrackeraiExporter
from flyhostel.data.sqlite3.utils import serialize_arr

DATE_TIME = "2026-10-26_18-00-00"
IMG_SHAPE = (1000, 1000)

TOML_ROIS = [
    "+ Ellipse {'center': [499, 518], 'axes': [448, 457], 'angle': 97}",
    '- Polygon [[438.3, 64.5], [431.1, 145.2], [586.4, 138.1], [591.5, 69.6]]',
    '- Polygon [[442.3, 878.8], [435.2, 972.8], [601.7, 972.8], [596.6, 874.7]]',
]

CONF_ROIS = [
    ["[473, 70], [390, 83], [346, 95], [314, 108], [287, 120], [259, 136], [221, 161], [129, 253], [106, 289], [89, 319], [77, 345], [64, 379], [52, 426], [45, 557], [57, 629], [70, 668], [82, 699], [95, 725], [113, 756], [153, 808], [237, 883], [271, 905], [298, 919], [326, 932], [361, 944], [408, 957], [557, 963], [620, 950], [660, 938], [691, 925], [720, 911], [749, 894], [798, 858], [873, 777], [894, 743], [911, 714], [923, 686], [936, 652], [948, 599], [951, 459], [939, 400], [926, 363], [914, 333], [901, 308], [883, 277], [838, 220], [758, 151], [725, 130], [698, 116], [670, 103], [635, 91], [585, 78], "],
    ["- [438, 64], [431, 145], [586, 138], [592, 70], "],
    ["- [442, 879], [435, 973], [602, 973], [597, 875], "],
]

# a point inside each negative polygon, and the center of the arena
IN_TOP_NOTCH = (100, 510)
IN_BOTTOM_NOTCH = (930, 520)
CENTER = (518, 499)


def write_toml(path, roi_list):
    with open(path, "w") as handle:
        handle.write("number_of_animals = 2\nroi_list = [\n")
        for roi in roi_list:
            handle.write(f'    "{roi}",\n')
        handle.write("]\n")


def export_roi_map(tmp_path, conf_rois, toml_rois=TOML_ROIS):
    """Write ROI_MAP with the exporter (as fh export) and verify it against DATE_TIME.toml"""
    write_toml(tmp_path / f"{DATE_TIME}.toml", toml_rois)
    exporter = IdtrackeraiExporter.__new__(IdtrackeraiExporter)
    exporter._basedir = str(tmp_path)
    exporter._date_time = DATE_TIME
    exporter._idtrackerai_conf = {"_roi": {"value": conf_rois}}
    exporter._store_metadata = {"imgshape": list(IMG_SHAPE)}
    exporter._temp_path = str(tmp_path / "mask.jpg")
    dbfile = str(tmp_path / "test.db")
    exporter.init_roi_map_table(dbfile)
    exporter.write_roi_map_table(dbfile)
    exporter.verify_roi_map_table(dbfile)
    return dbfile


def test_negative_rois_are_written_and_read_as_negative(tmp_path):
    dbfile = export_roi_map(tmp_path, CONF_ROIS)
    roi_map = load_roi_map(dbfile)

    assert [sign for sign, _ in roi_map["rois"]] == [1, -1, -1]
    for (_, stored), (_, expected) in zip(roi_map["rois"], parse_conf_rois(CONF_ROIS)):
        np.testing.assert_array_equal(stored, expected)

    mask = roi_map["mask"]
    assert mask[CENTER] == 255
    assert mask[IN_TOP_NOTCH] == 0 and mask[IN_BOTTOM_NOTCH] == 0
    assert mask[5, 5] == 0
    # the bounding box is the one of the ellipse, not changed by the negative polygons
    assert (roi_map["x"], roi_map["y"], roi_map["w"], roi_map["h"]) == (45, 70, 907, 894)


def test_verification_fails_if_the_conf_lacks_the_negative_rois(tmp_path):
    with pytest.raises(ROIMapMismatchError, match="signs"):
        export_roi_map(tmp_path, CONF_ROIS[:1])


def test_verification_fails_if_a_negative_roi_moved(tmp_path):
    moved = CONF_ROIS[:2] + [["- [42, 479], [35, 573], [202, 573], [197, 475], "]]
    with pytest.raises(ROIMapMismatchError, match="ROI 2"):
        export_roi_map(tmp_path, moved)


def test_verification_fails_if_the_ellipse_changed(tmp_path):
    toml_rois = ["+ Ellipse {'center': [499, 518], 'axes': [400, 457], 'angle': 97}"] + TOML_ROIS[1:]
    with pytest.raises(ROIMapMismatchError, match="ROI 0"):
        export_roi_map(tmp_path, CONF_ROIS, toml_rois=toml_rois)


def test_arena_calibration_ignores_the_negative_rois(tmp_path):
    dbfile = export_roi_map(tmp_path, CONF_ROIS)
    cx, cy, r = arena_calib(dbfile, px_per_mm=15)
    assert abs(cx - 499) < 3 and abs(cy - 518) < 3
    # radius of the ellipse, not of the ellipse without the notches
    assert abs(r - np.sqrt(448 * 457)) < 5


def test_roi_map_without_rois_column(tmp_path):
    """dbfiles made before negative ROIs were supported"""
    rois = parse_conf_rois(CONF_ROIS[:1])
    mask = serialize_arr(draw_roi_mask(rois, IMG_SHAPE), str(tmp_path / "mask.jpg"))
    dbfile = str(tmp_path / "old.db")
    with sqlite3.connect(dbfile) as conn:
        conn.execute("CREATE TABLE ROI_MAP (roi_idx smallint(6), roi_value smallint(6), x smallint(6), y smallint(6), w smallint(6), h smallint(6), mask longblob);")
        conn.execute("INSERT INTO ROI_MAP VALUES (0, 0, 45, 70, 907, 894, ?);", (mask,))

    roi_map = load_roi_map(dbfile)
    assert [sign for sign, _ in roi_map["rois"]] == [1]
    write_toml(tmp_path / f"{DATE_TIME}.toml", TOML_ROIS[:1])
    verify_roi_map(dbfile, str(tmp_path / f"{DATE_TIME}.toml"))
