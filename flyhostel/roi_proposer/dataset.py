"""
Build a YOLO segmentation dataset from all experiments that already have DATE_TIME.toml + landmarks.toml

One image per experiment: the chunk snapshot (.png) the landmarks were annotated on.
The most recent experiments (by date) are held out for validation, to mimic proposing ROIs for new experiments.
"""
import argparse
import glob
import os
import shutil

import cv2
import numpy as np
import tomli

from flyhostel.constants import ROI_PROPOSER_DIR
from .rois import CLASSES, chunk_of, load_experiment_rois


def get_parser(ap=None):
    if ap is None:
        ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--videos", default="/flyhostel_data/videos")
    ap.add_argument("--output", default=os.path.join(ROI_PROPOSER_DIR, "data"))
    ap.add_argument("--n-val", type=int, default=60)
    return ap


def snapshot_of(basedir):
    date_time = os.path.basename(basedir)
    for f in ["landmarks.toml", f"{date_time}.toml"]:
        path = os.path.join(basedir, f)
        if os.path.exists(path):
            chunk = chunk_of(tomli.loads(open(path).read()))
            if chunk is not None:
                png = os.path.join(basedir, f"{str(chunk).zfill(6)}.png")
                if os.path.exists(png):
                    return png
    return None


def main(args=None):
    if args is None:
        args = get_parser().parse_args()
    experiments = []
    for basedir in glob.glob(os.path.join(args.videos, "FlyHostel*", "*X", "*")):
        date_time = os.path.basename(basedir)
        if not (os.path.exists(os.path.join(basedir, f"{date_time}.toml")) and os.path.exists(os.path.join(basedir, "landmarks.toml"))):
            continue
        experiments.append(basedir)

    experiments.sort(key=os.path.basename)
    val = set(experiments[-args.n_val:])

    n = {"train": 0, "val": 0}
    for basedir in experiments:
        png = snapshot_of(basedir)
        if png is None:
            print(f"No snapshot for {basedir}")
            continue
        rois = load_experiment_rois(basedir)
        if not rois["arena"] or not rois["food"] or len(rois["notch"]) != 4:
            print(f"Skipping {basedir}: {[(k, len(v)) for k, v in rois.items()]}")
            continue
        h, w = cv2.imread(png, cv2.IMREAD_GRAYSCALE).shape

        split = "val" if basedir in val else "train"
        name = basedir.replace(args.videos, "").strip("/").replace("/", "__")
        os.makedirs(os.path.join(args.output, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.output, "labels", split), exist_ok=True)
        # copies, not links: experiments get moved / cleaned up and training must be reproducible
        dest = os.path.join(args.output, "images", split, name + ".png")
        if os.path.lexists(dest):
            os.remove(dest)
        shutil.copy2(png, dest)

        lines = []
        for cls_id, cls in enumerate(CLASSES):
            for poly in rois[cls]:
                poly = np.clip(poly, 0, [w - 1, h - 1]) / [w, h]
                lines.append(f"{cls_id} " + " ".join(f"{v:.5f}" for v in poly.flatten()))
        with open(os.path.join(args.output, "labels", split, name + ".txt"), "w") as handle:
            handle.write("\n".join(lines) + "\n")
        n[split] += 1

    with open(os.path.join(args.output, "dataset.yaml"), "w") as handle:
        handle.write(f"path: {os.path.abspath(args.output)}\ntrain: images/train\nval: images/val\nnames:\n")
        for i, cls in enumerate(CLASSES):
            handle.write(f"  {i}: {cls}\n")
    print(n)


if __name__ == "__main__":
    main()
