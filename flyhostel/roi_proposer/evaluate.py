"""
Compare proposed ROIs with the hand-made ones on the validation experiments (the most recent ones)
Saves side by side previews (hand-made in white, proposal in color) under runs/eval/
"""
import argparse
import glob
import os

import cv2
import numpy as np

from flyhostel.constants import ROI_PROPOSER_DIR
from .propose import DEFAULT_WEIGHTS, predict_rois
from .rois import CLASSES, draw_rois, load_experiment_rois, parse_roi, roi_to_polygon


def get_parser(ap=None):
    if ap is None:
        ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", default=os.path.join(ROI_PROPOSER_DIR, "data", "images", "val"))
    ap.add_argument("--videos", default="/flyhostel_data/videos")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--output", default=os.path.join(ROI_PROPOSER_DIR, "runs", "eval"))
    return ap


def iou(poly_a, poly_b, shape):
    a = np.zeros(shape, np.uint8)
    b = np.zeros(shape, np.uint8)
    cv2.fillPoly(a, [np.int32(poly_a)], 1)
    cv2.fillPoly(b, [np.int32(poly_b)], 1)
    return (a & b).sum() / max((a | b).sum(), 1)


def main(args=None):
    if args is None:
        args = get_parser().parse_args()
    from ultralytics import YOLO
    model = YOLO(args.weights)
    os.makedirs(args.output, exist_ok=True)

    stats = {k: {"center_px": [], "iou": []} for k in CLASSES}
    missing = {k: 0 for k in CLASSES}
    for png in sorted(glob.glob(os.path.join(args.images, "*.png"))):
        # images are named FlyHostelN__NX__DATE_TIME after their experiment
        basedir = os.path.join(args.videos, os.path.basename(png)[:-len(".png")].replace("__", "/"))
        if not os.path.isdir(basedir):
            print(f"{basedir} not found, skipping")
            continue
        img = cv2.imread(png)
        truth = load_experiment_rois(basedir)
        rois, _ = predict_rois(model, png)
        pred = {k: [roi_to_polygon(*parse_roi(r)) for r in v] for k, v in rois.items()}

        for k in CLASSES:
            used = set()
            for t in truth[k]:
                tc = t.mean(0)
                candidates = [(np.linalg.norm(p.mean(0) - tc), j) for j, p in enumerate(pred[k]) if j not in used]
                if not candidates:
                    missing[k] += 1
                    continue
                dist, j = min(candidates)
                used.add(j)
                stats[k]["center_px"].append(dist)
                stats[k]["iou"].append(iou(t, pred[k][j], img.shape[:2]))

        for k in CLASSES:
            for t in truth[k]:
                cv2.polylines(img, [np.int32(t)], True, (255, 255, 255), max(2, img.shape[0] // 400))
        draw_rois(img, rois, thickness=max(2, img.shape[0] // 700))
        cv2.imwrite(os.path.join(args.output, os.path.basename(png).replace(".png", ".jpg")), cv2.resize(img, (900, 900)))

    print(f"{'roi':6s} {'n':>4s} {'missed':>6s} {'center err px (median / 90%)':>30s} {'IoU (median / 10%)':>22s}")
    for k in CLASSES:
        c = np.array(stats[k]["center_px"])
        i = np.array(stats[k]["iou"])
        print(f"{k:6s} {len(c):4d} {missing[k]:6d} {np.median(c):14.1f} / {np.percentile(c, 90):6.1f} {np.median(i):14.3f} / {np.percentile(i, 10):5.3f}")


if __name__ == "__main__":
    main()
