import argparse
from tqdm.auto import tqdm

from flyhostel.data.pose.constants import bodyparts_xy, framerate
from flyhostel.data.pose.movie import annotate_chunk
from flyhostel.data.pose.main import FlyHostelLoader


def get_parser():
    
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--identity", type=str, required=True)
    ap.add_argument("--input", type=str, required=False, default=None)
    ap.add_argument("--output", type=str, required=False, default="./output.mp4")
    ap.add_argument("--stride", type=int, required=False, default=1)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--verbose", action="store_true", default=False)
    ap.add_argument("--with-pose", action="store_true", default=False, dest="with_pose")
    ap.add_argument("--frame-numbers", type=int, nargs="+", default=None)
    return ap


def annotate_by_chunk():

    cache="/flyhostel_data/cache"

    ap = get_parser()
    args=ap.parse_args()

    stride=args.stride
    experiment=args.experiment
    identity=args.identity
    

    chunks=range(0, 400)
    loader = FlyHostelLoader(experiment, chunks=chunks)
    loader.load_pose_data(
        cache=cache,
        stride=stride,
    )

    loader.process_data(
        cache=cache,
        filters=None, useGPU=0,
        stride=stride,
        identity=identity
    )
    
    loader.load_behavior_data(loader.experiment, identity=0, pose=loader.pose_boxcar, interpolate_frames=25)

    df=loader.behavior
    del loader
    
    if df is None:
        return
    
    df.loc[df["behavior"].isna(), "behavior"]="unknown"
    fps=framerate//stride

    df=df.reset_index(drop=True).drop("index", axis=1, errors="ignore")
    annotate_chunk(
        experiment, df, args.chunk, identity,
        input_video=args.input, output_video=args.output,
        fps=fps,
        with_pose=args.with_pose,
        gui_progress=args.verbose,
        frame_numbers=args.frame_numbers
    )
