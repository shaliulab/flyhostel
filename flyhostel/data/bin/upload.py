import argparse
from flyhostel.data.upload import upload_chunks

def get_parser(ap=None):

    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", nargs="+", type=int)
    ap.add_argument("--chunks", nargs="+", type=int)
    ap.add_argument("--experiment-folder", dest="experiment_folder", type=str, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    return ap


def main(ap=None, args=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()


    return upload_chunks(
        experiment_folder=args.experiment_folder,
        interval=args.interval, chunks=args.chunks,
        jobs=args.jobs
    )