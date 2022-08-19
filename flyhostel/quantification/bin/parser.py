import argparse

def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    
    ap.add_argument(
        "--imgstore-folder", dest="imgstore_folder", required=True, type=str
    )

    ap.add_argument("--interval", nargs="+", type=int, required=False, default=None)

    ap.add_argument("--output", dest="output", default=None, type=str)
    ap.add_argument(
        "--ld-annotation",
        dest="ld_annotation",
        action="store_true",
        default=True,
    )
    ap.add_argument(
        "--no-ld-annotation",
        dest="ld_annotation",
        action="store_false",
        default=True,
    )

    ap.add_argument("--source", default="trajectories", choices=["trajectories", "blobs", "csv"])
    ap.add_argument("--interpolate-nans", action="store_true", default=False)   
    return ap
