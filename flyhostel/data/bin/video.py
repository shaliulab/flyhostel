import argparse
from flyhostel.data.video import SingleVideoMaker


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str, help="path to FlyHostel.db")
    ap.add_argument("--frame-number", default=None, type=int, nargs="+")
    return ap


def main(args=None, ap=None):
    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    video_maker=SingleVideoMaker(flyhostel_dataset=args.dataset, value=args.frame_number)
    video_maker.make_single_video()