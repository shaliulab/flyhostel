import argparse
import os.path
import logging
import re
import joblib
from dropy.web_utils import sync, list_folder
from dropy import DropboxDownloader

logger = logging.getLogger(__name__)

def match_files_to_patterns(files, patterns):
    keep_files = []
    for file in files:
        for pattern in patterns:
            if re.match(pattern, file):
                keep_files.append(file)
                break
    
    return keep_files


def download_analysis_results(rootdir, folder, version=2, ncores=-2):
    """
    Downloads the idtrackerai results stored in Dropbox
    """

    assert rootdir.startswith("/")
    assert folder.startswith("/")

    res = list_folder(folder)
    files = res["files"]
    # dirs = res["dirs"]

    if version == 1:
        analysis_folder = folder
    elif version == 2:
        analysis_folder = os.path.join(folder, "idtrackerai") 

    patterns = [
        os.path.join(
            analysis_folder,
            "session_[0-9]{6}_error.txt"
        ),
        os.path.join(
            analysis_folder,
            "session_[0-9]{6}/video_object.npy"
        ),
        os.path.join(
            analysis_folder,
            "session_[0-9]{6}/preprocessing/blobs_collection*.npy"
        ),

        os.path.join(
            analysis_folder,
            "session_[0-9]{6}/preprocessing/fragments.npy"
        ),
    ]

    keep_files = match_files_to_patterns(files, patterns)
    
    if len(keep_files) == 0:
        logger.warning(f"No files matching patterns in {folder}")
        return

    logger.debug(f"Files to be downloaded: {keep_files}")

    joblib.Parallel(n_jobs=ncores)(
        joblib.delayed(sync)(
            f"Dropbox:{folder}/{file}", os.path.join(rootdir, file)
        )
            for file in keep_files
    )


def get_parser(ap=None):
    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("--rootdir", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--version", default=2)
    ap.add_argument("--ncores", default=-2)
    return ap

def main(ap=None, args=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()

    download_analysis_results(
        rootdir = args.rootdir,
        folder = args.folder,
        version=args.version,
        ncores=args.ncores
    )
