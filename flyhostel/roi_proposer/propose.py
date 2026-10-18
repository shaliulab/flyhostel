"""
Propose the idtrackerai tomls of an experiment

    roi-proposer propose /flyhostel_data/videos/FlyHostel2/1X/2026-10-16_13-00-00

What is written depends on the folder:
  FlyHostelN/NX/DATE_TIME            normal experiment: DATE_TIME.toml (arena) + landmarks.toml (food + 4 notches),
                                     ROIs predicted by the YOLO model trained with roi-proposer train
  FlyHostelN/NX/DATE_TIME            arena experiment (several arenas detected, or --arenas): arenas.toml,
                                     one ellipse per arena
  FlyHostelN/NX/DATE_TIME/arenas/K   one arena of an arena experiment: DATE_TIME.toml (the arena floor)
                                     + landmarks.toml (identical: no notches, everything is food)

Non-ROI settings (intensity / area thresholds, ...) are copied from the most recent annotated
experiment of the same kind recorded with the same camera, preferring the same hostel.
number_of_animals comes from the NX folder (1 for single arenas, override with --number-of-animals).
A preview (rois_proposal.jpg) is saved next to the tomls so the proposal can be checked before adjusting it.
"""
import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np
import tomli
import yaml

from flyhostel.constants import ROI_PROPOSER_DIR
from .rois import (
    ARENAS_TOML_SCALE, CLASSES, N_NOTCHES, detect_arenas, detect_floor, draw_rois,
    fit_ellipse, fit_rectangle, sort_notches,
)

DEFAULT_WEIGHTS = os.path.join(ROI_PROPOSER_DIR, "runs", "roi_proposer", "weights", "best.pt")
DEFAULT_CHUNK = 100
EXPERIMENT_PATTERN = re.compile(r"^(?P<videos>.*)/(?P<hostel>FlyHostel\d+)/(?P<n>\d+)X/(?P<date_time>[^/]+?)(/arenas/(?P<arena>[^/]+))?/?$")


def get_parser(ap=None):
    if ap is None:
        ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("basedir", nargs="+", help="Experiment folder(s), e.g. /flyhostel_data/videos/FlyHostel1/3X/2025-06-21_15-00-00")
    ap.add_argument("--chunk", type=int, default=None, help=f"Chunk to use (default: {DEFAULT_CHUNK} or the last one available)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--arenas", action="store_true", help="Force arena experiment mode for a DATE_TIME folder (detected automatically otherwise)")
    ap.add_argument("--number-of-animals", type=int, default=None, help="Default: from the NX folder, or 1 in arenas/K folders")
    ap.add_argument("--force", action="store_true", help="Overwrite existing toml files")
    ap.add_argument("--dry-run", action="store_true", help="Print the proposal and save the preview, but do not write tomls")
    ap.add_argument("--preview-dir", default=None, help="Where to save the preview (default: the experiment folder)")
    return ap


def parse_experiment(basedir):
    match = EXPERIMENT_PATTERN.match(basedir)
    if match is None:
        raise ValueError(f"{basedir} does not look like .../FlyHostelN/NX/DATE_TIME or .../FlyHostelN/NX/DATE_TIME/arenas/K")
    return match.groupdict()


def read_metadata(basedir):
    md = open(os.path.join(basedir, "metadata.yaml")).read()
    cam = re.search(r"camera-model: (\S+)", md)
    res = re.search(r"imgshape:\s*\n\s*- (\d+)", md)
    return (cam.group(1) if cam else None), (int(res.group(1)) if res else None)


def _nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


# where to find annotated experiments of each kind, and which tomls hold their settings
TEMPLATES = {
    "normal": (("FlyHostel*", "*X", "*"), lambda d, dt: {"main": f"{dt}.toml", "lm": "landmarks.toml"}),
    "arenas": (("FlyHostel*", "*X", "*"), lambda d, dt: {"main": "arenas.toml"}),
    "arena": (("FlyHostel*", "*X", "*", "arenas", "*"), lambda d, dt: {"main": f"{dt}.toml"}),
}


def find_template(basedir, kind):
    """
    Most recent annotated experiment of the same kind, recorded with the same camera
    Prefers the same experiment (other arenas), then the same hostel, then the same resolution
    """
    exp = parse_experiment(basedir)
    cam, res = read_metadata(basedir)
    pattern, files = TEMPLATES[kind]

    candidates = []
    for other in glob.glob(os.path.join(exp["videos"], *pattern)):
        if os.path.abspath(other) == basedir:
            continue
        try:
            other_exp = parse_experiment(other)
        except ValueError:
            continue
        if other_exp["date_time"] > exp["date_time"]:
            continue
        paths = {k: os.path.join(other, f) for k, f in files(other, other_exp["date_time"]).items()}
        if not all(_nonempty(p) for p in paths.values()):
            continue
        try:
            other_cam, other_res = read_metadata(other)
        except FileNotFoundError:
            continue
        same_experiment = other_exp["hostel"] == exp["hostel"] and other_exp["date_time"] == exp["date_time"]
        key = (other_cam == cam, same_experiment, other_exp["hostel"] == exp["hostel"], other_res == res, other_exp["date_time"])
        candidates.append((key, other, paths))

    if not candidates:
        raise RuntimeError(f"No annotated {kind} experiment found to copy settings from")
    key, template, paths = max(candidates, key=lambda c: c[0])
    if not key[0]:
        print(f"  WARNING: no annotated {kind} experiment recorded with camera {cam}, settings come from another camera")
    return template, {k: tomli.loads(open(p).read()) for k, p in paths.items()}


def pick_snapshot(basedir, chunk):
    """
    Returns (image, chunk, source): the chunk .png if available, else the first frame of the chunk .mp4
    (arenas/K folders get .png files only after being processed)
    """
    def chunks(ext):
        return sorted(int(os.path.basename(f).split(".")[0]) for f in glob.glob(os.path.join(basedir, "[0-9]" * 6 + ext)))

    pngs, mp4s = chunks(".png"), chunks(".mp4")
    if chunk is None:
        if DEFAULT_CHUNK in pngs or DEFAULT_CHUNK in mp4s:
            chunk = DEFAULT_CHUNK
        elif pngs:
            chunk = pngs[-1]
        elif mp4s:
            # the last one may still be being written
            chunk = mp4s[-2] if len(mp4s) > 1 else mp4s[-1]
        else:
            raise FileNotFoundError(f"No chunks (.png / .mp4) in {basedir}")

    png = os.path.join(basedir, f"{str(chunk).zfill(6)}.png")
    if os.path.exists(png):
        return cv2.imread(png), chunk, png
    mp4 = os.path.join(basedir, f"{str(chunk).zfill(6)}.mp4")
    cap = cv2.VideoCapture(mp4)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"Could not read chunk {chunk} (.png or .mp4) in {basedir}")
    return frame, chunk, mp4


def floor_from_parent(basedir, chunk):
    """
    Floor of an arenas/K folder, detected in the original (parent) recording where all arenas are visible,
    which is more robust than in the crop (uneven light makes the wall as bright as the floor)
    imgstore-crop-arenas keeps the pixel scale and saves the offset: original (x, y) = arena (x, y) + offset
    """
    with open(os.path.join(basedir, "metadata.yaml")) as handle:
        arena = yaml.safe_load(handle).get("arena")
    if not arena or "offset" not in arena or "polygon" not in arena:
        return None
    parent = os.path.dirname(os.path.dirname(basedir))
    try:
        parent_img, _, _ = pick_snapshot(parent, chunk)
    except FileNotFoundError:
        return None
    arenas = detect_arenas(parent_img)
    if not arenas:
        return None
    center = np.array(arena["polygon"], dtype=np.float32).mean(0)
    floor = min(arenas, key=lambda a: np.linalg.norm(a.mean(0) - center))
    if np.linalg.norm(floor.mean(0) - center) > 0.5 * np.ptp(floor[:, 0]):
        return None
    return floor - np.array(arena["offset"], dtype=np.float32)


def largest_contour(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return max(contours, key=cv2.contourArea) if contours else None


def predict_rois(model, img):
    """Returns ({class: [toml roi string]}, {class: [confidence]}) """
    result = model.predict(img, imgsz=1024, conf=0.1, retina_masks=True, verbose=False)[0]
    h, w = result.orig_shape
    rois = {k: [] for k in CLASSES}
    confs = {k: [] for k in CLASSES}
    if result.masks is None:
        return rois, confs
    masks = result.masks.data.cpu().numpy() > 0.5
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    order = np.argsort(-conf)

    contours = {k: [] for k in CLASSES}
    for i in order:
        name = CLASSES[cls[i]]
        limit = N_NOTCHES if name == "notch" else 1
        if len(contours[name]) >= limit:
            continue
        contour = largest_contour(masks[i][:h, :w])
        if contour is None or len(contour) < 5:
            continue
        contours[name].append(contour)
        confs[name].append(float(conf[i]))

    for c in contours["arena"]:
        rois["arena"].append(fit_ellipse(c))
    for c in contours["food"]:
        rois["food"].append(fit_ellipse(c))
    if contours["arena"]:
        center = contours["arena"][0].reshape(-1, 2).mean(0)
    else:
        center = (w / 2, h / 2)
    for c in sort_notches(contours["notch"], center):
        rois["notch"].append(fit_rectangle(c))
    return rois, confs


def format_toml(video_path, settings, number_of_animals, roi_list, landmarks):
    """Same layout as the hand-made tomls"""
    lines = ["video_paths = [", f"    '{video_path}',", "]"]
    lines.append(f"intensity_ths = {list(settings['intensity_ths'])}")
    lines.append(f"area_ths = {list(settings['area_ths'])}")
    lines.append('tracking_intervals = ""')
    lines.append(f"number_of_animals = {number_of_animals}")
    lines.append("use_bkg = false")
    lines.append("check_segmentation = false")
    if "resolution_reduction" in settings:
        lines.append(f"resolution_reduction = {float(settings['resolution_reduction'])}")
    lines.append(f"track_wo_identities = {str(bool(settings.get('track_wo_identities', False))).lower()}")
    lines.append("roi_list = [")
    for roi in roi_list:
        quote = '"' if "'" in roi else "'"
        lines.append(f"    {quote}{roi}{quote},")
    lines.append("]")
    if landmarks:
        lines.append("exclusive_rois = false")
    return "\n".join(lines) + "\n"


class LazyModel:
    """The YOLO model is only needed (and loaded) for normal experiments"""

    def __init__(self, weights):
        self._weights = weights
        self._model = None

    def get(self):
        if self._model is None:
            if not os.path.exists(self._weights):
                raise FileNotFoundError(f"Model weights not found: {self._weights}. Train it with roi-proposer train or pass --weights")
            from ultralytics import YOLO
            self._model = YOLO(self._weights)
        return self._model


def propose(model, basedir, args):
    basedir = os.path.abspath(basedir.rstrip("/"))
    exp = parse_experiment(basedir)
    date_time = exp["date_time"]
    img, chunk, source = pick_snapshot(basedir, args.chunk)
    video_path = os.path.join(os.path.realpath(basedir), f"{str(chunk).zfill(6)}.mp4")
    print(f"\n{basedir}\n  snapshot: {source}")

    problems = []
    if exp["arena"] is not None:
        kind = "arena"
        number_of_animals = args.number_of_animals or 1
        floor = floor_from_parent(basedir, chunk)
        if floor is not None:
            print("  single arena: floor found in the original recording")
        else:
            floor = detect_floor(img)
            print("  single arena: " + ("floor found in the arena video" if floor is not None else "floor NOT found"))
        rois = {"arena": [] if floor is None else [fit_ellipse(floor)]}
        if floor is None:
            problems.append("arena floor not found")
    else:
        arenas = detect_arenas(img)
        if arenas or args.arenas:
            kind = "arenas"
            rois = {"arena": [fit_ellipse(a, scale=ARENAS_TOML_SCALE) for a in arenas]}
            print(f"  arena experiment: {len(arenas)} arenas found")
            if not arenas:
                problems.append("no arenas found")
        else:
            kind = "normal"
            number_of_animals = args.number_of_animals or int(exp["n"])
            rois, confs = predict_rois(model.get(), img)
            for k in CLASSES:
                print(f"  {k}: {len(rois[k])} found, confidence {[round(c, 2) for c in confs[k]]}")
            if len(rois["arena"]) != 1:
                problems.append("arena not found")
            if len(rois["food"]) != 1:
                problems.append("food not found")
            if len(rois["notch"]) != N_NOTCHES:
                problems.append(f"{len(rois['notch'])}/{N_NOTCHES} notches found")
    if problems:
        print("  WARNING: " + ", ".join(problems) + " -> please draw the missing ROIs by hand")

    template, settings = find_template(basedir, kind)
    print(f"  settings copied from: {template}")
    if kind == "normal":
        outputs = {
            f"{date_time}.toml": format_toml(video_path, settings["main"], number_of_animals, rois["arena"], landmarks=False),
            "landmarks.toml": format_toml(video_path, settings["lm"], number_of_animals, rois["food"] + rois["notch"], landmarks=True),
        }
    elif kind == "arena":
        main_txt = format_toml(video_path, settings["main"], number_of_animals, rois["arena"], landmarks=False)
        outputs = {f"{date_time}.toml": main_txt, "landmarks.toml": main_txt}
    else:
        # the arenas are cropped out of this video: the template's number_of_animals (0) is kept
        number_of_animals = args.number_of_animals if args.number_of_animals is not None else settings["main"].get("number_of_animals", 0)
        outputs = {"arenas.toml": format_toml(video_path, settings["main"], number_of_animals, rois["arena"], landmarks=True)}

    preview = draw_rois(img.copy(), rois, thickness=max(2, img.shape[0] // 500))
    if args.preview_dir:
        os.makedirs(args.preview_dir, exist_ok=True)
        preview_path = os.path.join(args.preview_dir, basedir.replace("/", "_").strip("_") + "_rois_proposal.jpg")
    else:
        preview_path = os.path.join(basedir, "rois_proposal.jpg")
    cv2.imwrite(preview_path, preview)
    print(f"  preview: {preview_path}")

    for filename, txt in outputs.items():
        path = os.path.join(basedir, filename)
        if args.dry_run:
            print(f"--- {path}\n{txt}")
            continue
        if _nonempty(path) and not args.force:
            print(f"  {path} exists, not overwriting (use --force)")
            continue
        with open(path, "w") as handle:
            handle.write(txt)
        print(f"  wrote {path}")


def main(args=None):
    if args is None:
        args = get_parser().parse_args()
    model = LazyModel(args.weights)
    failed = False
    for basedir in args.basedir:
        try:
            propose(model, basedir, args)
        except Exception as error:
            print(f"{basedir}: {error}", file=sys.stderr)
            failed = True
    sys.exit(int(failed))


if __name__ == "__main__":
    main()
