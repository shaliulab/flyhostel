import argparse
def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument(
        "--experiment-folder", "--input", dest="input", type=str, required=True
    )
    ap.add_argument(
        "--output", dest="output", type=str, default="."
    )
    ap.add_argument(
        "--reference_hour",
        "--zt0",
        dest="reference_hour",
        type=float,
        required=True,
    )

    ap.add_argument(
        "--light-threshold",
        dest="light_threshold",
        type=int,
        default=None,
    )
    return ap
