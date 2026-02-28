import argparse
import logging
import shutil
import os.path

from flyhostel.data.human_validation.main import annotate_for_validation
logger=logging.getLogger(__name__)
logging.getLogger("flyhostel.data.human_validation.main").setLevel(logging.DEBUG)

def get_parser():
    ap=argparse.ArgumentParser()
    ap.add_argument("--experiment", type=str, required=True)
    ap.add_argument("--folder", type=str, required=True)
    ap.add_argument("--time-window-length", type=float, required=True)
    ap.add_argument("--min-frame-number", default=None, type=int)
    ap.add_argument("--max-frame-number", default=None, type=int)
    ap.add_argument("--n-jobs", dest="n_jobs", type=int, default=-2)
    ap.add_argument("--cache", action="store_true", default=False)
    return ap

def main():

    ap=get_parser()
    args=ap.parse_args()
    
    time_window_length=round(1/150, 3)
    os.makedirs(args.folder, exist_ok=True)
    try:
        shutil.rmtree(os.path.join(args.folder, "movies"))
    except:
        pass
    
    df, df_bin, qc_fail=annotate_for_validation(
        args.experiment, args.folder,
        time_window_length=args.time_window_length,
        format=".mp4",
        n_jobs=args.n_jobs,
        cache=args.cache,
        min_frame_number=args.min_frame_number,
        max_frame_number=args.max_frame_number
    )

if __name__ == "__main__":
    main()