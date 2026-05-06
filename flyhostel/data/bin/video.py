import argparse
import os.path
from flyhostel.data.video import SingleVideoMaker
from flyhostel.utils import (
    get_square_width,
    get_square_height,
)

def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True, type=str)
    ap.add_argument("--dataset", required=True, type=str, help="path to FlyHostel.db")
    ap.add_argument("--stacked", action="store_true", default=False, help="Whether to output single fly data from the same chunk in the same video, stacked horizontally, or not (separate videos). Default False (separate videos)")
    ap.add_argument("--frame-number", default=None, type=int, nargs="+", help="If not passed, all frames will be used")
    ap.add_argument("--n-jobs", type=int, default=-2, help="Number of parallel processes to make videos")
    ap.add_argument("--width", type=int, default=None, help="Expected width of the images taken from the video data or the cached segmentation_data")
    ap.add_argument("--height", type=int, default=None, help="Expected height of the images taken from the video data or the cached segmentation_data")
    ap.add_argument("--resolution", type=str, default=None,
                    help="""
                    Resolution of the resulting video. If not identical to width and height, the frames are resized accordingly.
                    If the dataset contains data for more than one animal, the width of the video will be scaled for each animal.
                    Example, 200x200 for two animals produces a video with resolution 400x200. Format is widthxheight
                    """
                )
    ap.add_argument("--output", type=str, required=False, default=None, help="Folder where the output will be generated")
    ap.add_argument("--chunksize", type=int, default=None, help="If provided, chunksize of the output")
    ap.add_argument("--chunks", type=int, nargs="+", help="Chunks to be processed")
    ap.add_argument("--identifiers", type=int, nargs="+", help="Local identities", default=[-1])
    return ap


def main(args=None, ap=None):
    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()


    video_maker=SingleVideoMaker(experiment = args.experiment, flyhostel_dataset=args.dataset, value=args.frame_number, stacked=args.stacked, identifiers=args.identifiers)

    if args.height is None:
        height=get_square_height(args.experiment)
    else:
        height=args.height

    if args.width is None:
        width=get_square_width(args.experiment)
    else:
        width=args.width

    
    frame_size=(width, height)


    if args.resolution is None:
        resolution=frame_size
    else:
        resolution=tuple([int(e) for e in args.resolution.split("x")])


    if args.n_jobs == 1:
        video_maker.make_single_video_single_process(
            output=args.output, frame_size=frame_size,
            resolution=resolution, chunks=args.chunks, chunksize=args.chunksize,
        )
    else:
        video_maker.make_single_video_multi_process(
            n_jobs=args.n_jobs, output=args.output, frame_size=frame_size,
            resolution=resolution, chunks=args.chunks, chunksize=args.chunksize,
        )


def save_csv(args=None, ap=None):
    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    video_maker=SingleVideoMaker(experiment = args.experiment, flyhostel_dataset=args.dataset, value=args.frame_number, stacked=args.stacked, identifiers=args.identifiers)
    output=args.output
    video_maker.save_coords_to_csv(output=output, chunks=args.chunks)