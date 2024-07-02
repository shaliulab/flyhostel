import argparse
import logging
import os

import pandas as pd
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.constants import bodyparts as BODYPARTS
from flyhostel.data.pose.constants import (
    bodyparts_speed,
    bodyparts_xy,
    MIN_TIME,
    MAX_TIME,
)
from flyhostel.data.pose.distances import compute_distance_features_pairs

logger=logging.getLogger(__name__)

def filter_experiment(experiment, identity, stride, min_time=MIN_TIME, max_time=MAX_TIME, output=".", n_jobs=1, **kwargs):

    loader = FlyHostelLoader(experiment, identity=identity, chunks=range(0, 400))
    loader.load_and_process_data(
        min_time=min_time, max_time=max_time,
        stride=stride,
        cache="/flyhostel_data/cache",
        useGPU=0,
        speed=False,
        sleep=False,
        n_jobs=n_jobs,
        load_behavior=False,
        **kwargs
    )
    pose=loader.pose_boxcar.copy()
    columns=bodyparts_xy

    pose=compute_distance_features_pairs(pose, [("head", "proboscis"),])
    loader.pose_interpolated=[]
    for id, df in pose.groupby("id"):
        logger.debug("Exporting %s", id)
        df_single_animal=loader.full_interpolation(df, columns)
        loader.export(pose=df_single_animal, dest_folder=output)
        loader.pose_interpolated.append(df_single_animal)
    loader.pose_interpolated=pd.concat(loader.pose_interpolated, axis=0)


def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--identity", type=str, required=True)
    ap.add_argument("--output", type=str, required=False, default=".")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min-time", type=int, default=MIN_TIME)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--max-time", type=int, default=MAX_TIME)
    ap.add_argument("--write-only", action="store_true", default=False, help="If passed, detected cache files are ignored, the computation is performed and the cache file is overwritten")
    ap.add_argument("--filters", default=None, type=str, nargs="+")
    ap.add_argument("--files", type=str, nargs="+", required=False, default=None, help="Path to pre-compiled pose files. Will be sorted alphabetically")
    return ap

def main():

    ap=get_parser()
    args=ap.parse_args()

    if args.files is not None:
        files=sorted(args.files)
    else:
        files=None

    filters_order=args.filters
    if len(filters_order)==1:
        filters_order=filters_order[0].split("-")


    filter_experiment(
        experiment=args.experiment, identity=args.identity, min_time=args.min_time, max_time=args.max_time, stride=args.stride,
        output=args.output, files=files, n_jobs=args.n_jobs, write_only=args.write_only, filters_order=filters_order

    )
