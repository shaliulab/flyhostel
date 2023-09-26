"""
Interface between flyhostel outputs abd B-SOID
"""

import argparse
import sqlite3
import os.path
import joblib
from flyhostel.data.pose import pipeline, load_concatenation_table, parse_number_of_animals

def main():
    """
    Concatenate the .h5 files produced in the analysis Nextflow process
    (which reformats existing .slp files into .h5 files)
    into a single file using the concatenation information

    .h5 files must be available under basedir/flyhostel/single_animal/id/
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--dbfile", type=str, required=True)
    ap.add_argument("--chunks", type=int, nargs="+", required=False, default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    args=ap.parse_args()


    experiment_name=os.path.splitext(os.path.basename(args.dbfile))[0]
    basedir = os.path.dirname(args.dbfile)

    with sqlite3.connect(args.dbfile) as conn:
        cur=conn.cursor()
        number_of_animals=parse_number_of_animals(cur)
        concatenation=load_concatenation_table(cur, basedir)

    joblib.Parallel(n_jobs=args.n_jobs)(
        joblib.delayed(
            pipeline
        )(
            experiment_name, identity, concatenation, args.chunks
        )
        for identity in range(1, number_of_animals+1)
    )