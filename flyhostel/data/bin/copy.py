import argparse
from flyhostel.data.idtrackerai import copy_idtrackerai_data

def get_parser(ap=None):
    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument(
        "--imgstore-folder", dest="imgstore_folder", required=True, type=str
    )

    ap.add_argument(
        "--analysis-folder", dest="analysis_folder", default=None, type=str
    )
    ap.add_argument(
        "--overwrite", action="store_true", default=True,
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
