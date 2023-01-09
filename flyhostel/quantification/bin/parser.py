import argparse
import os.path

from flyhostel.constants import ANALYSIS_FOLDER
from flyhostel.constants import OUTPUT_FOLDER

def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    
    ap.add_argument(
        "--imgstore-folder", dest="imgstore_folder", default=os.getcwd(), type=str
    )

    ap.add_argument("--interval", nargs="+", type=int, required=False, default=None)
    ap.add_argument("--n-jobs", dest="n_jobs", type=int, default=1, required=False)

    ap.add_argument("--output", dest="output", default=os.path.join(os.getcwd(), OUTPUT_FOLDER), type=str)
    ap.add_argument(
        "--ld-annotation",
        dest="ld_annotation",
        action="store_true",
        default=True,
    )
    ap.add_argument(
        "--no-ld-annotation",
        dest="ld_annotation",
        action="store_false",
        default=True,
    )

    ap.add_argument("--source", default="trajectories", choices=["trajectories", "blobs", "csv"])
    ap.add_argument("--interpolate-nans", action="store_true", default=False)   
    return ap
