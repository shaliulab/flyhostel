import argparse
import os.path
from flyhostel.data.download import PATTERNS
import subprocess

def get_parser(ap=None):

    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, required=True, help="Absolute path to root of flyhostel videos in local computer")
    ap.add_argument("--store-path", type=str, required=True, help="Path to metadata.yaml relative to the root (should be identical regardless of whether the local or remote path is specified")
    ap.add_argument("--imgstore", action="store_true", default=False)
    ap.add_argument("--idtrackerai", action="store_true", default=False)
    ap.add_argument("--flyhostel", action="store_true", default=False)
    ap.add_argument("--chunk", type=int, required=True)
    return ap

def main(args=None, ap=None):
    
    if args is None:
        ap = get_parser(ap=ap)
        args= ap.parse_args()
    DROPBOX_VIDEOS_ROOT="/Data/flyhostel_data/videos"


    session = str(args.chunk).zfill(6)
    basedir = os.path.dirname(args.store_path)
    folder = basedir


    if args.imgstore:
        imgstore_files=PATTERNS["imgstore"](folder, session=session)
    else:
        imgstore_files=[]

    if args.idtrackerai:
        idtrackerai_files=PATTERNS["idtrackerai"](folder, session=session, version=2)       
    else:
        idtrackerai_files=[]

    if args.flyhostel:
        raise NotImplementedError
    else:
        flyhostel_files = []


    files = imgstore_files + idtrackerai_files + flyhostel_files

    for file in files:
        src_file = f"Dropbox:{DROPBOX_VIDEOS_ROOT}/{file}"
        dst_file = os.path.join(args.root, file)
        cmd = ["dropy", src_file, dst_file]
        print(cmd)
        
        p = subprocess.Popen(cmd)

