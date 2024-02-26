
import argparse
from flyhostel.data.human_validation.scene_qc import annotate_scene_quality
def get_parser():

    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--folder", type=str, required=True)
    ap.add_argument("--n-jobs", default=-2, type=int)
    ap.add_argument("--sample-size", default=None, type=int, required=False)
    return ap


def main():

    ap=get_parser()
    args=ap.parse_args()
    annotate_scene_quality(args.experiment, args.folder, args.n_jobs, sample_size=args.sample_size)


