import argparse
import os.path
from flyhostel.data.pose.behavior import predict_behavior

LTA_DATA=os.environ["LTA_DATA"]

def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--identity", type=str, default=None)
    ap.add_argument("--output", type=str, default=None)
    ap.add_argument("--wavelets", type=str, default=None, help=
                    f"""
                    Path to pre-computed wavelets.mat. If not provided, will be assumed to be in {LTA_DATA}/Wavelets/experiment__identity-pcaModes-wavelets.mat
                    """)
    ap.add_argument("--files", type=str, nargs="+", required=False, default=None, help="Paht to pre-compiled pose files. Will be sorted alphabetically")
    ap.add_argument("--model-path", required=True, type=str, help="Path to .pkl serialized model")
    return ap


def main():

    ap = get_parser()
    args=ap.parse_args()

    assert os.path.exists(args.model_path)
    if args.wavelets is not None:
        assert os.path.exists(args.wavelets)
    
    predict_behavior(args.experiment, model_path=args.model_path, identity=args.identity, output=args.output, files=args.files, wavelets=args.wavelets)

