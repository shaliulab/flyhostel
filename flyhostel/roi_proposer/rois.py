"""
Parse / render / fit the ROIs used in DATE_TIME.toml (arena) and landmarks.toml (food + notches)
"""
import ast
import os
import re

import cv2
import numpy as np
import tomli

CLASSES = ["arena", "food", "notch"]
N_NOTCHES = 4


def parse_roi(line):
    """'+ Ellipse {...}' / '+ Polygon [[x, y], ...]' -> (shape, spec)"""
    _, shape, spec = line.split(" ", 2)
    return shape, ast.literal_eval(spec)


def roi_to_polygon(shape, spec):
    if shape == "Ellipse":
        return cv2.ellipse2Poly(tuple(int(v) for v in spec["center"]), tuple(int(v) for v in spec["axes"]), int(spec["angle"]), 0, 360, 5).astype(np.float32)
    return np.array(spec, dtype=np.float32)


def load_experiment_rois(basedir):
    """
    Returns {"arena": [poly], "food": [poly], "notch": [poly, ...]} from the existing toml files
    Follows flyhostel's LandmarksExporter: first ellipse is food, first 4 polygons are notches
    """
    out = {k: [] for k in CLASSES}
    date_time = os.path.basename(basedir.rstrip("/"))
    main = os.path.join(basedir, f"{date_time}.toml")
    lm = os.path.join(basedir, "landmarks.toml")
    if os.path.exists(main):
        conf = tomli.loads(open(main).read())
        for line in conf.get("roi_list", [])[:1]:
            out["arena"].append(roi_to_polygon(*parse_roi(line)))
    if os.path.exists(lm):
        conf = tomli.loads(open(lm).read())
        for line in conf.get("roi_list", []):
            shape, spec = parse_roi(line)
            if shape == "Ellipse" and not out["food"]:
                out["food"].append(roi_to_polygon(shape, spec))
            elif shape == "Polygon" and len(out["notch"]) < N_NOTCHES:
                out["notch"].append(roi_to_polygon(shape, spec))
    return out


def chunk_of(conf):
    """Chunk index used to annotate a toml (from video_paths)"""
    vp = conf.get("video_paths") or []
    if not vp:
        return None
    return int(re.sub(r"\D", "", os.path.basename(vp[0]).split(".")[0]))


# ---- mask -> toml roi string ----

def fit_ellipse(contour, scale=1.0):
    (cx, cy), (w, h), angle = cv2.fitEllipse(contour.astype(np.float32))
    # cv2 returns full axes lengths, idtrackerai wants semi-axes
    spec = {"center": [int(round(cx)), int(round(cy))], "axes": [int(round(scale * w / 2)), int(round(scale * h / 2))], "angle": int(round(angle)) % 180}
    return f"+ Ellipse {spec}"


def fit_rectangle(contour):
    box = cv2.boxPoints(cv2.minAreaRect(contour.astype(np.float32)))
    pts = ", ".join(f"[{x:.1f}, {y:.1f}]" for x, y in box)
    return f"+ Polygon [{pts}]"


def sort_notches(contours, center):
    """Clockwise from 12 o'clock, around the arena center"""
    def angle(c):
        m = c.reshape(-1, 2).mean(0)
        return np.arctan2(m[0] - center[0], -(m[1] - center[1])) % (2 * np.pi)
    return sorted(contours, key=angle)


def draw_rois(img, rois, thickness=2):
    """rois: {class: [toml roi string, ...]}"""
    colors = {"arena": (0, 0, 255), "food": (0, 255, 0), "notch": (255, 128, 0)}
    for cls, lines in rois.items():
        for line in lines:
            poly = roi_to_polygon(*parse_roi(line))
            cv2.polylines(img, [np.int32(poly)], True, colors[cls], thickness)
    return img


# arena experiments: round bright floors, found without ML
# hand-made arenas.toml ellipses (crop regions) are ~5% larger than the floor
ARENAS_TOML_SCALE = 1.05


def _gray(img):
    return img.copy() if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _hough_circles(gray, rmin, rmax, min_dist):
    g = gray.copy()
    # crops of arena experiments are padded with white, and so is the frame number label
    g[g >= 250] = 0
    g = cv2.medianBlur(g, 5)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=min_dist, param1=100, param2=0.7, minRadius=int(rmin), maxRadius=int(rmax))
    return [] if circles is None else circles[0]


def _floor_edge(gray, cx, cy, r, lo, hi, n=360, k=200):
    """
    Points on the floor edge around a rough circle: along each ray, the first point that is
    halfway between the floor brightness and the darkest value (robust to uneven light and to the wall rings behind)
    """
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rs = np.linspace(lo * r, hi * r, k)
    map_x = (cx + np.outer(np.cos(theta), rs)).astype(np.float32)
    map_y = (cy + np.outer(np.sin(theta), rs)).astype(np.float32)
    profiles = cv2.remap(g, map_x, map_y, cv2.INTER_LINEAR, borderValue=0)
    profiles[profiles >= 250] = 0
    floor = np.median(profiles[:, :k // 5], axis=1, keepdims=True)
    dark = profiles.min(axis=1, keepdims=True)
    below = profiles < (floor + dark) / 2
    below[:, :k // 5] = False
    idx = np.argmax(below, axis=1)
    found = below[np.arange(n), idx]
    if found.sum() < 5:
        return None
    radii = rs[idx]
    # flies sitting on the edge
    median = np.median(radii[found])
    mad = np.median(np.abs(radii[found] - median))
    keep = found & (np.abs(radii - median) < 3 * 1.4826 * mad + 2)
    return np.stack([cx + np.cos(theta) * radii, cy + np.sin(theta) * radii], 1)[keep].astype(np.float32)


def _robust_points(points, iterations=5):
    """
    Drop the points far from the ellipse fitted to them, iteratively
    (a shadowed stretch of floor makes a run of rays land on the wall)
    """
    for _ in range(iterations):
        if len(points) < 20:
            break
        (ex, ey), (w, h), angle = cv2.fitEllipse(points)
        t = np.deg2rad(angle)
        dx, dy = points[:, 0] - ex, points[:, 1] - ey
        u = (dx * np.cos(t) + dy * np.sin(t)) / (w / 2)
        v = (-dx * np.sin(t) + dy * np.cos(t)) / (h / 2)
        residual = np.hypot(u, v) - 1
        mad = np.median(np.abs(residual - np.median(residual))) + 1e-3
        keep = np.abs(residual - np.median(residual)) < 2.5 * 1.4826 * mad
        if keep.all():
            break
        points = points[keep]
    return points


def _refine_floor(gray, cx, cy, r):
    points = _floor_edge(gray, cx, cy, r, lo=0.5, hi=1.25)
    if points is None or len(points) < 5:
        return None
    (ex, ey), (w, h), _ = cv2.fitEllipse(_robust_points(points))
    points = _floor_edge(gray, ex, ey, (w + h) / 4, lo=0.8, hi=1.15)
    return None if points is None else _robust_points(points)


def detect_floor(img):
    """Cropped arena (arenas/N folders): the edge of the one floor filling the image"""
    gray = _gray(img)
    size = gray.shape[0]
    circles = _hough_circles(gray, 0.3 * size, 0.55 * size, size)
    if not len(circles):
        return None
    cx, cy, r = min(circles, key=lambda c: c[2])
    return _refine_floor(gray, cx, cy, r)


def detect_arenas(img):
    """
    Several arenas in one image (parent folder of an arena experiment)
    Returns the floor edge of each arena sorted top-left to bottom-right,
    or [] if the image does not look like an arena experiment
    """
    gray = _gray(img)
    size = gray.shape[0]
    circles = sorted(_hough_circles(gray, 0.12 * size, 0.3 * size, 0.4 * size), key=lambda c: c[2])
    # concentric circles are the floor and the wall rings: keep the innermost
    kept = []
    for c in circles:
        if all(np.hypot(c[0] - k[0], c[1] - k[1]) > 0.5 * k[2] for k in kept):
            kept.append(c)
    # arenas in one experiment are copies of each other
    if kept:
        median = np.median([c[2] for c in kept])
        kept = [c for c in kept if 0.8 < c[2] / median < 1.25]
    if len(kept) < 2:
        return []
    arenas = [a for a in (_refine_floor(gray, *c) for c in kept) if a is not None]
    row_height = size / 4
    return sorted(arenas, key=lambda a: (int(a[:, 1].mean() // row_height), a[:, 0].mean()))
