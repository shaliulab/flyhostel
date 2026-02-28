import argparse
import math
from flyhostel.utils import get_basedir
from flyhostel.data.human_validation.cvat.main import (
    integrate_human_annotations,
    save_human_annotations
)

def get_parser():

    ap=argparse.ArgumentParser(conflict_handler="resolve")
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--folder", type=str, default=None)
    ap.add_argument("--fn-interval", type=int, nargs=2, required=False, default=[0, math.inf])
    return ap

def main():

    ap=get_parser()
    ap.add_argument("--frames-from-annotation", dest="frames_from_annotation", action="store_true", help="Infer frame window from annotations")
    ap.add_argument("--number-of-rows", type=int, default=1, help="If images in cvat are a grid, how many rows the grid has")
    ap.add_argument("--number-of-cols", type=int, default=1, help="If images in cvat are a grid, how many rows the grid has")
    ap.add_argument("--redownload", action="store_true", default=False, required=False)

    args=ap.parse_args()

    if args.folder is None:
        folder = f"{get_basedir(args.experiment)}/flyhostel/validation"
    else:
        folder=args.folder

    integrate_human_annotations(
        args.experiment,
        folder=folder,
        first_frame_number=args.fn_interval[0],
        last_frame_number=args.fn_interval[1],
        redownload=args.redownload,
        number_of_rows=args.number_of_rows,
        number_of_cols=args.number_of_cols,
        frames_from_annotation=args.frames_from_annotation,
    )


def save():

    ap=get_parser()
    args=ap.parse_args()

    if args.folder is None:
        folder = f"{get_basedir(args.experiment)}/flyhostel/validation"
    else:
        folder=args.folder

    save_human_annotations(
        args.experiment, folder,
        first_frame_number=args.fn_interval[0],
        last_frame_number=args.fn_interval[1],
    )

