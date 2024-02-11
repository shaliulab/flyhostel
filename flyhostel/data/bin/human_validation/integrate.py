import argparse
from flyhostel.data.human_validation.cvat.main import integrate_human_annotations

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--folder", type=str, required=True)
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    return ap

def main():

    ap=get_parser()
    args=ap.parse_args()
    integrate_human_annotations(args.experiment, args.folder, args.tasks)

