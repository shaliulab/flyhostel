"""
Interface between SLEAP and downstream behavior pipelines
"""

import argparse
import glob
import sqlite3
import os.path
import joblib
from flyhostel.data.pose.export import pipeline, load_concatenation_table, parse_number_of_animals


def main():
    """
    Concatenate the .h5 files produced in the analysis Nextflow process
    (which reformats existing .slp files into .h5 files)
    into a single file using the concatenation information

    No imputation is performed
    .h5 files must be available under basedir/flyhostel/single_animal/id/
    files are saved to the --output folder
    """

    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", default=False)
    group=ap.add_mutually_exclusive_group()
    group.add_argument("--experiment", type=str, help="Experiment key (FlyHostelX_XX_XXXXX)")
    ap.add_argument("--identity", type=str, help="00 or 01 or 02...", default=None)
    group.add_argument("--dbfile", type=str, help="Path to .db file")
    ap.add_argument("--chunks", type=int, nargs="+", required=False, default=None)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--write-only", action="store_true", default=False, help="If passed, detected cache files are ignored, the computation is performed and the cache file is overwritten")
    ap.add_argument("--output", default=None, required=True, type=str)
    args=ap.parse_args()


    if args.experiment is not None:
        experiment_name=args.experiment
        tokens = experiment_name.split("_")
        basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], tokens[0], tokens[1], "_".join(tokens[2:4]))
        if not os.path.exists(basedir):
            basedirs = glob.glob(basedir + "*")
            if len(basedirs)==1:
                basedir=basedirs[0]
                experiment_name="_".join(basedir.split(os.path.sep)[-3:])
                print(experiment_name)
                tokens=experiment_name.split("_")

        print(f"Selected basedir {basedir}")
        dbfile = os.path.join(basedir, experiment_name + ".db")
    else:
        assert args.dbfile is not None
        dbfile = args.dbfile
        experiment_name = os.path.basename(dbfile).rstrip(".db")
        tokens=experiment_name.split("_")

    number_of_animals_x=tokens[1]
    flyhostel=tokens[0]
    date_time = "_".join(tokens[2:4])
    basedir = os.path.join(os.environ["FLYHOSTEL_VIDEOS"], flyhostel, number_of_animals_x, date_time)

    with sqlite3.connect(dbfile) as conn:
        cur=conn.cursor()
        number_of_animals=parse_number_of_animals(cur)
        if number_of_animals==1:
            concatenation=load_concatenation_table(cur, basedir, concatenation_table="CONCATENATION")
        else:
            concatenation=load_concatenation_table(cur, basedir, concatenation_table="CONCATENATION_VAL")


    if args.identity is None:
        identities=range(1, number_of_animals+1)
    else:
        identities=[int(args.identity)]

    joblib.Parallel(n_jobs=args.n_jobs)(
    # joblib.Parallel(n_jobs=1)(
        joblib.delayed(
            pipeline
        )(
            experiment_name, identity, concatenation, args.chunks, output=args.output, strict=False #, write_only=args.write_only
        )
        for identity in identities
    )