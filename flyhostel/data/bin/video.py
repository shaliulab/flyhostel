import argparse
import os.path
from flyhostel.data.video import SingleVideoMaker


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str, help="path to FlyHostel.db")
    ap.add_argument("--frame-number", default=None, type=int, nargs="+", help="If not passed, all frames will be used")
    ap.add_argument("--n-jobs", type=int, default=-2, help="Number of parallel processes to make videos")
    ap.add_argument("--width", type=int, default=200, help="Expected width of the images taken from the video data or the cached segmentation_data")
    ap.add_argument("--height", type=int, default=200, help="Expected height of the images taken from the video data or the cached segmentation_data")
    ap.add_argument("--resolution", type=str, default="200x200", help="Resolution of the resulting video. If not identical to width and height, the frames are resized accordingly. If the dataset contains data for more than one animal, the width of the video will be scaled for each animal. Example, 200x200 for two animals produces a video with resolution 400x200. Format is widthxheight")
    ap.add_argument("--basedir", type=str, required=False, default=None, help="Folder where the output will be generated")
    ap.add_argument("--chunksize", type=int, default=None, help="If provided, chunksize of the output")
    ap.add_argument("--chunks", type=int, nargs="+", help="Chunks to be processed")
    return ap


def main(args=None, ap=None):
    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    video_maker=SingleVideoMaker(flyhostel_dataset=args.dataset, value=args.frame_number)
    resolution=tuple([int(e) for e in args.resolution.split("x")])

    if args.n_jobs == 1:
        video_maker.make_single_video_single_process(output=args.basedir, frameSize=(args.width, args.height), resolution=resolution, chunks=args.chunks)
    else:
        video_maker.make_single_video_multi_process(n_jobs=args.n_jobs, output=args.basedir, frameSize=(args.width, args.height), resolution=resolution, chunks=args.chunks, chunksize=args.chunksize)
