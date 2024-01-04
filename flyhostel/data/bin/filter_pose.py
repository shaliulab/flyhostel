import argparse
import itertools

import pandas as pd
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.pose.constants import bodyparts as BODYPARTS
from flyhostel.data.pose.constants import (
    min_score,
    JUMP_WINDOW_SIZE_SECONDS,
    MAX_JUMP_MM,
    interpolate_seconds,
    framerate,
    bodyparts_speed,
    MIN_TIME,
    MAX_TIME,
)


def filter_experiment(experiment, min_time, max_time, stride):

    loader = FlyHostelLoader(experiment, chunks=range(0, 400))
    loader.load_and_process_data(
        min_time=min_time, max_time=max_time,
        stride=stride, bodyparts=BODYPARTS,
        cache="/flyhostel_data/cache",
        # filters is ignored
        filters=None,
        min_score=min_score,
        window_size_seconds=JUMP_WINDOW_SIZE_SECONDS,
        max_jump_mm=MAX_JUMP_MM,
        interpolate_seconds=interpolate_seconds,
        framerate=framerate,
        useGPU=0
    )

    to_export=loader.pose_speed_boxcar.copy()
    to_export=loader.obtain_umap_input(to_export, to_export, dt0_features=["head_proboscis_distance"])
    loader.pose_interpolated=[]
    for id, df in to_export.groupby("id"):
        df_single_animal=loader.full_interpolation(df, bodyparts_speed)
        loader.export(df_single_animal)
        loader.pose_interpolated.append(df_single_animal)
    loader.pose_interpolated=pd.concat(loader.pose_interpolated, axis=0)


def get_parser():    
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min-time", type=int, default=MIN_TIME)
    ap.add_argument("--max-time", type=int, default=MAX_TIME)
    return ap

def main():

    ap=get_parser()
    args=ap.parse_args()

    filter_experiment(experiment=args.experiment, min_time=args.min_time, max_time=args.max_time, stride=args.stride)
