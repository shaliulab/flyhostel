import argparse


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("-n", "--number-of-animals", type=int)
    ap.add_argument("-t", "--time-steps", type=int)
    ap.add_argument("-o", "--output", type=str, default=None)
    ap.add_argument("-j", "--n-jobs", type=int, default=1)
    return ap  
