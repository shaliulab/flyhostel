import argparse
from flyhostel.data.sqlite3 import export_dataset


def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, type=str, help="path to metadata.yaml")
    ap.add_argument("--tables", required=False, type=str, nargs="+", default="all", help="List of tables to be written to sqlite file, 'all' means all of them")
    # ap.add_argument("--tables", required=False, type=str, default="all", help="Comma separated values tables to be written to sqlite file, 'all' means all of them")
    ap.add_argument("--chunks", nargs="+", type=int, required=True, help="chunks to export")
    ap.add_argument("--reset", action="store_true", default=False, help="If an existing dbfile is found, remove and make it from scratch")
    return ap


def main(args=None, ap=None):
    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()


    # if args.tables == "all":
    #     tables=args.tables
    # else:
    #     tables = args.tables.split(",")

    if args.tables[0] == "all":
        tables="all"
    else:
        tables=args.tables




    export_dataset(args.metadata, args.chunks, reset=args.reset, tables=tables)
