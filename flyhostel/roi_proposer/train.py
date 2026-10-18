"""
Train the ROI proposer (YOLO instance segmentation: arena, food, notch)
"""
import argparse
import os

from flyhostel.constants import ROI_PROPOSER_DIR


def get_parser(ap=None):
    if ap is None:
        ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=os.path.join(ROI_PROPOSER_DIR, "data", "dataset.yaml"))
    ap.add_argument("--model", default="yolo11m-seg.pt")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--name", default="roi_proposer")
    return ap


def main(args=None):
    if args is None:
        args = get_parser().parse_args()
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=8,
        project=os.path.join(ROI_PROPOSER_DIR, "runs"), name=args.name, exist_ok=True,
        # one arena per image: mosaic would break the scene layout
        mosaic=0.0, close_mosaic=0,
        # notches can be anywhere around the arena
        degrees=180, fliplr=0.5, flipud=0.5, scale=0.2, translate=0.1,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        patience=50, plots=True,
    )


if __name__ == "__main__":
    main()
