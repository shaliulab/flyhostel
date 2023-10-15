"""
Interface between flyhostel outputs abd B-SOID
"""

import argparse
import sqlite3
import os.path
import joblib
from flyhostel.data.pose import pipeline, load_concatenation_table, parse_number_of_animals
from flyhostel.data.pose.preprocess import main as preprocess

def main():
    """
    Concatenate the .h5 files produced in the analysis Nextflow process
    (which reformats existing .slp files into .h5 files)
    into a single file using the concatenation information

    No imputation is performed
    .h5 files must be available under basedir/flyhostel/single_animal/id/
    files are saved to whatever $POSE_DATA points to
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=False)
    group=ap.add_mutually_exclusive_group()
    group.add_argument("--experiment", type=str, help="Experiment key (FlyHostelX_XX_XXXXX)")
    group.add_argument("--dbfile", type=str, help="Path to .db file")
    ap.add_argument("--chunks", type=int, nargs="+", required=False, default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    args=ap.parse_args()


    if args.experiment is not None:
        experiment_name=args.experiment
        tokens = experiment_name.split("_")
        flyhostel=tokens[0]
        number_of_animals=tokens[1]
        date_time = "_".join(tokens[2:4])

        basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], flyhostel, number_of_animals, date_time)
        dbfile = os.path.join(basedir, experiment_name + ".db")
    else:
        assert args.dbfile is not None
        dbfile = args.dbfile
        tokens=dbfile.split(os.path.sep)
        experiment_name = "_".join(tokens[-4:]).rstrip(".db")
        number_of_animals=int(tokens[-4:][1].rstrip("X"))

    print(dbfile)

    with sqlite3.connect(dbfile) as conn:
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