import argparse
import itertools
from flyhostel.data.pose.main import FlyHostelLoader
from flyhostel.data.bodyparts import bodyparts as BODYPARTS



# for documentation purpose
filters={"nanmedian": {"window_size": 0.2, "min_window_size": 40, "order": 0}, "nanmean": {"window_size": 0.2, "min_window_size":10, "order": 1}}

interpolate_seconds={bp: 30 for bp in BODYPARTS}
interpolate_seconds["proboscis"]=0.5
ZT_START=14
ZT_END=15
min_score={bp: 0.5 for bp in BODYPARTS}
min_score["proboscis"]=0.8
bodyparts_xy=list(itertools.chain(*[[bp + "_x", bp + "_y"] for bp in BODYPARTS]))
bodyparts_speed=list(itertools.chain(*[[bp + "_speed"] for bp in BODYPARTS]))
min_time=ZT_START*3600
max_time=ZT_END*3600
stride=75


def filter_experiment(experiment, min_time, max_time):


    loader = FlyHostelLoader(experiment, chunks=range(0, 400))
    loader.load_and_process_data(
        min_time=min_time, max_time=max_time,
        stride=stride, bodyparts=BODYPARTS,
        cache="/flyhostel_data/cache",
        filters=None, min_score=min_score,
        window_size_seconds=0.5, max_jump_mm=1,
        interpolate_seconds=interpolate_seconds,
        useGPU=0
    )

    to_export=loader.pose_speed_boxcar.copy()
    to_export=loader.obtain_umap_input(to_export, to_export, dt0_features=["head_proboscis_distance"])
    loader.pose_interpolated=loader.full_interpolation(to_export, bodyparts_speed)
    loader.export(to_export)


def get_parser():    
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    return ap

def main():

    ap=get_parser()
    args=ap.parse_args()

    filter_experiment(experiment=args.experiment, min_time=max_time,_max_time=max_time)
