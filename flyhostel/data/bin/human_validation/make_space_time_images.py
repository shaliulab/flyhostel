import argparse
from flyhostel.data.human_validation.space_time_images import make_space_time_images

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--folder", type=str, required=True)
    ap.add_argument("--time-window-length", type=float, required=True)
    ap.add_argument("--n-jobs", dest="n_jobs", type=int, default=-2)
    return ap


def main():
    
    ap = get_parser()
    args=ap.parse_args()
    make_space_time_images(
        args.folder, args.experiment,
        time_window_length=args.time_window_length,
        n_jobs=args.n_jobs
    )