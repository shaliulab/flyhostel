import argparse
from flyhostel.data.video import SingleVideoMaker


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str, help="path to FlyHostel.db")
    ap.add_argument("--frame-number", default=None, type=int, nargs="+", help="If not passed, all frames will be used")
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--height", type=int, default=200)
    ap.add_argument("--basedir", type=str, required=True, help="Folder where the output will be generated")
    return ap


def main(args=None, ap=None):
    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    video_maker=SingleVideoMaker(flyhostel_dataset=args.dataset, value=args.frame_number)
    video_maker.make_single_video(basedir=args.basedir, frameSize=(args.width, args.height))
