import argparse
from flyhostel.data.interactions.main import compute_experiment_interactions
from flyhostel.data.pose.constants import bodyparts_wo_joints as BODYPARTS

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--number-of-animals", type=int, required=True)
    ap.add_argument("--output", type=str, required=True, help="Output csv with interaction table")
    ap.add_argument("--dist-thresh", type=float, dest="dist_max_mm", help="Max distance between interacting flies, in mm", required=True)
    ap.add_argument("--time-thresh", type=float, dest="min_interaction_duration", help="Min time spent interacting, in seconds", required=True)
    return ap


def main():
    ap=get_parser()
    args=ap.parse_args()
    raise NotImplementedError
    compute_experiment_interactions(
        group, bodyparts=BODYPARTS, number_of_animals=args.number_of_animals,
        output=args.output, dist_max_mm=args.dist_max_mm, min_interaction_duration=args.min_interaction_duration
    )