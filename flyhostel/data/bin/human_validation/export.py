import argparse
from flyhostel.data.human_validation.export import export_images

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--folder", type=str, required=True)
    ap.add_argument("--n-jobs", dest="n_jobs", type=int, default=-2)
    return ap


def main():

    ap = get_parser()
    args=ap.parse_args()
    export_images(
        args.folder, args.experiment,
        n_jobs=args.n_jobs
    )