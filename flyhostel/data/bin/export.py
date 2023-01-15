import argparse
from flyhostel.data.sqlite3 import export_dataset


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, type=str, help="path to metadata.yaml")
    ap.add_argument("--chunks", nargs="+", type=int, required=True, help="chunks to export")
    return ap


def main(args=None, ap=None):
    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    export_dataset(args.metadata, args.chunks)
