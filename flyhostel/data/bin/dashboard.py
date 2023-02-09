import argparse
import re
import os.path
import warnings

import joblib

from flyhostel.data.dashboard import validate_experiment


def parse_metadata_path(metadata):
    flyhostel_id=int(re.search("FlyHostel([0-9])", metadata).group(1))
    number_of_animals=int(re.search("([0-9])X", metadata).group(1))
    date_time=re.search("[0-9]X/([0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]_[0-9][0-9]-[0-9][0-9]-[0-9][0-9])", metadata).group(1)

    return flyhostel_id, number_of_animals, date_time
 

def process_entry(entry, chunk_start, chunk_end):
    if os.path.exists(entry):
        print(f"Processing {entry}")
        flyhostel_id, number_of_animals, date_time = parse_metadata_path(entry)
        validate_experiment(flyhostel_id=flyhostel_id, number_of_animals=number_of_animals, date_time=date_time, chunk_start=chunk_start, chunk_end=chunk_end)
    else:
        warnings.warn(f"{entry} not found")


def main():

    ap=argparse.ArgumentParser()

    ap.add_argument("--flyhostel-id")
    ap.add_argument("--number-of-animals")
    ap.add_argument("--date-time")
    ap.add_argument("--metadata")
    ap.add_argument("--index")
    ap.add_argument("--chunk-start", default=50, type=int)
    ap.add_argument("--chunk-end", default=349, type=int)
    ap.add_argument("--n-jobs", type=int, default=1)
    args=ap.parse_args()

    if args.index is not None:

        with open(args.index, "r") as filehandle:
            lines=filehandle.readlines()

        lines=[line.strip() for line in lines]

        joblib.Parallel(n_jobs=args.n_jobs)(joblib.delayed(process_entry)(line, args.chunk_start, args.chunk_end)
            for line in lines
        )

        return

    elif args.metadata is not None:
        flyhostel_id, number_of_animals, date_time = parse_metadata_path(args.metadata)
     
    else:
        flyhostel_id=args.flyhostel_id
        number_of_animals=args.number_of_animals
        date_time=args.date_time

    validate_experiment(flyhostel_id=flyhostel_id, number_of_animals=number_of_animals, date_time=date_time, chunk_start=args.chunk_start, chunk_end=args.chunk_end)
