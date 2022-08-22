import argparse
import os.path
from flyhostel.data.idtrackerai import copy_idtrackerai_data
from flyhostel.constants import ANALYSIS_FOLDER

def get_parser(ap=None):
    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument(
        "--imgstore-folder", dest="imgstore_folder", type=str, default=os.getcwd()
    )

    ap.add_argument(
        "--analysis-folder", dest="analysis_folder", type=str, default=os.path.join(os.getcwd(), ANALYSIS_FOLDER)
    )
    ap.add_argument(
        "--overwrite", action="store_true", default=False,
        help="Makes a new copy of the idtrackerai's trajectory file"
        " even if a copy already exists, overwriting it"
    )

    ap.add_argument("--interval", nargs="+", type=int)
    
    return ap


def main(args=None, ap=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    copy_idtrackerai_data(args.imgstore_folder, args.analysis_folder, interval=args.interval, overwrite=args.overwrite)
