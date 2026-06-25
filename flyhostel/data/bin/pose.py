"""
Interface between SLEAP and downstream behavior pipelines
"""

import argparse
from flyhostel.utils.pose_export import recreate_pose_file

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

    compile(args.experiment, args.chunks, args.output, identity=args.identity, n_jobs=args.n_jobs)


def compile(experiment, chunks, output, identity=None, n_jobs=1):
    recreate_pose_file(experiment, chunks=chunks, output=output, identity=identity, n_jobs=n_jobs)
