import argparse
import math
from flyhostel.data.human_validation.cvat.main import integrate_human_annotations, save_human_annotations

def get_parser():

    ap=argparse.ArgumentParser(conflict_handler="resolve")
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--folder", type=str, required=True)
    ap.add_argument("--fn-interval", type=int, nargs=2, required=False, default=[0, math.inf])
    return ap

def main():

    ap=get_parser()
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    ap.add_argument("--number-of-rows", type=int, required=True)
    ap.add_argument("--number-of-cols", type=int, required=True)
    ap.add_argument("--redownload", action="store_true", default=False, required=False)

    args=ap.parse_args()
    integrate_human_annotations(
        args.experiment, args.folder, args.tasks,
        first_frame_number=args.fn_interval[0],
        last_frame_number=args.fn_interval[1],
        redownload=args.redownload,
        number_of_rows=args.number_of_rows,
        number_of_cols=args.number_of_cols, 
    )


def save():

    ap=get_parser()
    args=ap.parse_args()
    save_human_annotations(
        args.experiment, args.folder,
        first_frame_number=args.fn_interval[0],
        last_frame_number=args.fn_interval[1],
    )

