import argparse
from flyhostel.data.pose.ethogram.ethogram import make_ethogram
from flyhostel.data.pose.ethogram.plot import main as draw_ethogram

def get_parser():
    
    ap=argparse.ArgumentParser()
    # group=ap.add_mutually_exclusive_group()
    # group.add_argument("--experiment", required=False, default=None)
    ap.add_argument("--experiment", required=False, default=None)
    ap.add_argument("--identity", required=False, default=None)
    ap.add_argument("--files", nargs="+", type=str, default=None, help="Processed pose files (.h5)")
    ap.add_argument("--raw-files", nargs="+", type=str, default=None, help="Processed pose files (.h5)")
    ap.add_argument("--wavelets", type=str, default=None)
    # group.add_argument("--input", required=False, type=str, default=None,
    #                    help="path to input video on which annotations will be drawn. If ")
    ap.add_argument("--output", required=False, type=str, default=".")
    ap.add_argument("--model-path")
    ap.add_argument("--frame-number", type=int, nargs="+", default=None)
    ap.add_argument("--t0", type=int, default=None)
    ap.add_argument("--postprocess", action="store_true", default=False)
    ap.add_argument("--correct-by-all-inactive", dest="correct_by_all_inactive", action="store_true", default=False)
    ap.add_argument("--correct-groom-behaviors", action="store_true", default=False, help="If True, groom bouts shorter than 5 seconds are set to background")
    return ap

def main():
    
    ap=get_parser()
    args=ap.parse_args()
    make_ethogram(
        args.experiment, str(args.identity).zfill(2),
        model_path=args.model_path,
        files=args.files,
        raw_files=args.raw_files,
        wavelet_file=args.wavelets,
        output=args.output,
        frame_numbers=args.frame_number,
        postprocess=args.postprocess,
        # t0=args.t0,
        correct_by_all_inactive=args.correct_by_all_inactive,
        cache=None,
        correct_groom_behaviors=args.correct_groom_behaviors,
    )

    return None