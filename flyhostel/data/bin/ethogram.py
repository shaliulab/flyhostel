import argparse
from flyhostel.data.pose.ethogram import make_ethogram, MODEL_PATH

def get_parser():
    
    ap=argparse.ArgumentParser()
    group=ap.add_mutually_exclusive_group()
    group.add_argument("--experiment", required=False, default=None)
    ap.add_argument("--identity", required=False, default=None)
    group.add_argument("--input", required=False, type=str, default=None,
                       help="path to input video on which annotations will be drawn. If ")
    ap.add_argument("--output", required=False, type=str, default=".")
    ap.add_argument("--model-path", default=MODEL_PATH, required=False)
    ap.add_argument("--frame-number", type=int, nargs="+", default=None)
    ap.add_argument("--postprocess", action="store_true", default=False)
    return ap

def main():
    
    ap=get_parser()
    args=ap.parse_args()
    fig=make_ethogram(args.experiment, str(args.identity).zfill(2), model_path=args.model_path, input=args.input, output=args.output, frame_numbers=args.frame_number, postprocess=args.postprocess)

    return None
