"""
ROIs of the arena, as stored in the ROI_MAP table of a FlyHostel dbfile

The arena ROI is made of one or more signed ROIs, applied in order as in idtrackerai:
positive ROIs (+) add their area to the arena and negative ROIs (-) remove it,
e.g. an ellipse followed by negative rectangles is the ellipse without the rectangles.

They come from the _roi field of the idtrackerai .conf (DATE_TIME.conf), where rows starting with - are negative,
which is made with conf-convert from the roi_list of DATE_TIME.toml (e.g. "+ Ellipse {...}", "- Polygon [[x, y], ...]").

ROI_MAP has one row (roi_idx 0) with
    * mask: the arena (255) with the negative ROIs removed (0)
    * x, y, w, h: bounding box of the positive ROIs
    * rois: JSON list of {"sign": 1 or -1, "polygon": [[x, y], ...]}, in the order they are applied
"""
import ast
import io
import json
import logging
import re
import sqlite3

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# a ROI of the main toml and its copy in ROI_MAP must agree in at least this fraction of their pixels
MIN_ROI_AGREEMENT = 0.95
# x, y, w, h of ROI_MAP vs the bounding box of the positive ROIs of the main toml, as a fraction of the size of the box
# (conf-convert approximates ellipses with ~50 points, which cuts their extremes by ~1%)
MAX_BOUNDING_BOX_ERROR = 0.02


class ROIMapMismatchError(ValueError):
    """Raised when the ROI_MAP of a dbfile does not match the ROI of the main toml of the experiment"""


# ---- .conf (idtrackerai v4) ----

def split_roi_sign(row):
    """
    Sign (1 or -1) and points (as a string) of a row of the _roi field of a .conf file
    Rows starting with - are negative, rows starting with + or without a sign are positive
    """
    if isinstance(row, (list, tuple)) and len(row) > 0 and isinstance(row[0], str):
        row = row[0]
    row = row.strip()
    if row.startswith("-"):
        return -1, row[1:]
    if row.startswith("+"):
        return 1, row[1:]
    return 1, row


def parse_conf_rois(rows):
    """
    List of (sign, contour) of the rows of the _roi field of a .conf file
    contour is an int32 array of shape (N, 1, 2). ROIs with less than 3 points are skipped
    """
    rois = []
    for row in rows:
        sign, points = split_roi_sign(row)
        contour = np.array(ast.literal_eval("[" + points.strip().rstrip(",") + "]"), np.int32).reshape((-1, 1, 2))
        if len(contour) < 3:
            continue
        rois.append((sign, contour))
    return rois


# ---- .toml (idtrackerai v5) ----

def parse_toml_roi(line):
    """'+ Ellipse {...}' / '- Polygon [[x, y], ...]' -> (sign, shape, spec)"""
    match = re.match(r"^\s*([+-]?)\s*(Ellipse|Polygon)\s*(.*)$", line, re.DOTALL)
    if match is None:
        raise ValueError(f"Cannot parse ROI {line}. Supported ROIs are + or - followed by Ellipse or Polygon")
    sign, shape, spec = match.groups()
    return (-1 if sign == "-" else 1), shape, ast.literal_eval(spec)


def draw_toml_roi(shape, spec, img, value):
    if shape == "Ellipse":
        center = tuple(int(round(v)) for v in spec["center"])
        axes = tuple(int(round(v)) for v in spec["axes"])
        return cv2.ellipse(img, center, axes, float(spec["angle"]), 0, 360, value, -1)
    contour = np.round(np.array(spec, np.float64)).astype(np.int32).reshape((-1, 1, 2))
    return cv2.drawContours(img, [contour], -1, value, -1)


def toml_roi_points(shape, spec):
    """Outline of a ROI of a toml, not clipped to the frame (except at 0, like conf-convert)"""
    if shape == "Ellipse":
        center = tuple(int(round(v)) for v in spec["center"])
        axes = tuple(int(round(v)) for v in spec["axes"])
        points = cv2.ellipse2Poly(center, axes, int(round(spec["angle"])), 0, 360, 1)
    else:
        points = np.round(np.array(spec, np.float64)).astype(np.int32)
    return np.clip(points.reshape((-1, 1, 2)), 0, None).astype(np.int32)


def load_toml_rois(toml_path):
    """roi_list of an idtrackerai v5 .toml as a list of (sign, shape, spec)"""
    import tomli
    with open(toml_path, "rb") as handle:
        conf = tomli.load(handle)
    return [parse_toml_roi(line) for line in conf.get("roi_list", [])]


# ---- masks ----

def _start_mask(signs, shape, value):
    # if the first ROI is negative, the arena is the whole frame without it (as in idtrackerai)
    if signs and signs[0] < 0:
        return np.full(shape, value, np.uint8)
    return np.zeros(shape, np.uint8)


def draw_roi_mask(rois, shape, value=255):
    """
    Mask (value inside the arena, 0 outside) of a list of (sign, contour)
    ROIs are applied in order: positive ROIs add their area and negative ROIs remove it
    """
    mask = _start_mask([sign for sign, _ in rois], shape, value)
    for sign, contour in rois:
        mask = cv2.drawContours(mask, [contour], -1, value if sign > 0 else 0, -1)
    return mask


def draw_toml_roi_mask(toml_rois, shape, value=255):
    """Same as draw_roi_mask, for the (sign, shape, spec) of load_toml_rois"""
    mask = _start_mask([sign for sign, _, _ in toml_rois], shape, value)
    for sign, roi_shape, spec in toml_rois:
        mask = draw_toml_roi(roi_shape, spec, mask, value if sign > 0 else 0)
    return mask


def roi_bounding_rect(rois):
    """x, y, w, h of the positive ROIs"""
    positive = [contour for sign, contour in rois if sign > 0]
    if not positive:
        positive = [contour for _, contour in rois]
    return cv2.boundingRect(np.concatenate(positive))


# ---- ROI_MAP ----

def serialize_rois(rois):
    return json.dumps([{"sign": int(sign), "polygon": contour.reshape((-1, 2)).tolist()} for sign, contour in rois])


def deserialize_rois(text):
    return [
        (int(roi["sign"]), np.array(roi["polygon"], np.int32).reshape((-1, 1, 2)))
        for roi in json.loads(text)
    ]


def decode_mask_blob(b):
    b = bytes(b)
    if b[:6] == b'\x93NUMPY':                      # numpy .npy
        return np.load(io.BytesIO(b))
    try:                                            # PNG/JPEG
        a = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
        if a is not None: return a
    except Exception: pass
    try:
        from PIL import Image
        return np.array(Image.open(io.BytesIO(b)).convert('L'))
    except Exception: pass
    try:                                            # pickled array
        return np.asarray(np.load(io.BytesIO(b), allow_pickle=True))
    except Exception: pass
    raise ValueError(f"Unrecognized mask blob; first 16 bytes: {b[:16]!r}")


def load_roi_map(dbfile):
    """
    Arena of a dbfile (roi_idx 0 of ROI_MAP) as a dict with keys
        x, y, w, h: bounding box of the positive ROIs
        mask: uint8 array, 255 inside the arena and 0 outside (the negative ROIs are outside)
        rois: list of (sign, contour), sign is 1 for positive ROIs and -1 for negative ROIs

    dbfiles made before negative ROIs were supported have no rois column:
    their only ROI is positive and is recovered from the outline of the mask
    """
    with sqlite3.connect(dbfile) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM ROI_MAP WHERE roi_idx = 0 LIMIT 1").fetchone()
    if row is None:
        raise ValueError(f"ROI_MAP of {dbfile} is empty")

    mask = decode_mask_blob(row["mask"])
    mask = np.where(mask > (mask.max() / 2), 255, 0).astype(np.uint8)

    if "rois" in row.keys() and row["rois"]:
        rois = deserialize_rois(row["rois"])
    else:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        rois = [(1, max(contours, key=cv2.contourArea))] if contours else []

    return {"x": row["x"], "y": row["y"], "w": row["w"], "h": row["h"], "mask": mask, "rois": rois}


def _agreement(a, b, region):
    region = region > 0
    if not region.any():
        return 1.0
    return float(((a > 0) == (b > 0))[region].mean())


def verify_roi_map(dbfile, toml_path):
    """
    Check that the arena in ROI_MAP is the ROI in the roi_list of the main toml (DATE_TIME.toml) of the experiment:
    same number of ROIs with the same signs, and the stored polygons and mask match the toml in the area of every ROI

    Raises ROIMapMismatchError otherwise
    """
    toml_rois = load_toml_rois(toml_path)
    if not toml_rois:
        # old experiments, whose ROI was only drawn in the .conf
        logger.warning("%s has no roi_list, ROI_MAP of %s cannot be verified", toml_path, dbfile)
        return
    roi_map = load_roi_map(dbfile)
    shape = roi_map["mask"].shape
    errors = []

    toml_signs = [sign for sign, _, _ in toml_rois]
    stored_signs = [sign for sign, _ in roi_map["rois"]]
    if toml_signs != stored_signs:
        errors.append(f"ROI signs {stored_signs} in ROI_MAP != {toml_signs} in {toml_path}")
    else:
        for i, ((sign, roi_shape, spec), (_, contour)) in enumerate(zip(toml_rois, roi_map["rois"])):
            expected = draw_toml_roi(roi_shape, spec, np.zeros(shape, np.uint8), 255)
            stored = cv2.drawContours(np.zeros(shape, np.uint8), [contour], -1, 255, -1)
            agreement = _agreement(expected, stored, expected | stored)
            if agreement < MIN_ROI_AGREEMENT:
                errors.append(f"ROI {i} ({'+' if sign > 0 else '-'} {roi_shape}) of ROI_MAP overlaps only {agreement:.3f} with {toml_path}")

    expected_mask = draw_toml_roi_mask(toml_rois, shape)
    for i, (sign, roi_shape, spec) in enumerate(toml_rois):
        region = draw_toml_roi(roi_shape, spec, np.zeros(shape, np.uint8), 255)
        agreement = _agreement(expected_mask, roi_map["mask"], region)
        if agreement < MIN_ROI_AGREEMENT:
            errors.append(f"mask of ROI_MAP matches only {agreement:.3f} of the area of ROI {i} ({'+' if sign > 0 else '-'} {roi_shape}) of {toml_path}")

    positive = [toml_roi_points(roi_shape, spec) for sign, roi_shape, spec in toml_rois if sign > 0]
    if positive:
        # like x, y, w, h of ROI_MAP, the box may extend beyond the frame
        expected_box = cv2.boundingRect(np.concatenate(positive))
        stored_box = tuple(int(roi_map[k]) for k in ("x", "y", "w", "h"))
        tolerance = max(3, MAX_BOUNDING_BOX_ERROR * max(expected_box[2:]))
        if max(abs(a - b) for a, b in zip(expected_box, stored_box)) > tolerance:
            errors.append(f"x, y, w, h {stored_box} of ROI_MAP != {expected_box} (positive ROIs of {toml_path})")

    if errors:
        raise ROIMapMismatchError(f"ROI_MAP of {dbfile} does not match {toml_path}:\n" + "\n".join(errors))
    logger.info("ROI_MAP of %s matches %s (%s ROIs)", dbfile, toml_path, len(toml_rois))
