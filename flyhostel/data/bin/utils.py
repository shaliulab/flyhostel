import argparse
from flyhostel.utils import (
    get_framerate,
    get_chunksize,
    get_wavelet_profile,
    get_number_of_animals,
)

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str)
    return ap

def main_get_wavelet_profile():
    ap = get_parser()

    args=ap.parse_args()
    experiment=args.experiment
    profile=get_wavelet_profile(experiment)
    print(profile)
    return 0


def main_get_chunksize():
    ap = get_parser()

    args=ap.parse_args()
    experiment=args.experiment
    chunksize=get_chunksize(experiment)
    print(chunksize)
    return 0


def main_get_framerate():
    ap = get_parser()

    args=ap.parse_args()
    experiment=args.experiment
    framerate=get_framerate(experiment)
    print(framerate)
    return 0

def main_get_number_of_animals():
    ap = get_parser()

    args=ap.parse_args()
    experiment=args.experiment
    number_of_animals=get_number_of_animals(experiment)
    print(number_of_animals)
    return 0

