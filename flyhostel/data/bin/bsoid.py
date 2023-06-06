import argparse
import sqlite3
import os.path
import joblib
from flyhostel.data.bsoid import pipeline, load_concatenation_table, parse_number_of_animals

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=False)
    ap.add_argument("--dbfile", type=str, required=True)
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
            experiment_name, identity, concatenation
        )
        for identity in range(1, number_of_animals+1)
    )
