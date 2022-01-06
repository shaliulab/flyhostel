import argparse
import os.path
import logging
import re
import joblib
from dropy.web_utils import sync as sync_
from dropy.web_utils import list_folder
from dropy import DropboxDownloader

logger = logging.getLogger(__name__)

def match_files_to_patterns(folder, files, patterns):
    keep_files = []
    for file in files:
        for pattern in patterns:
            if re.match(pattern, file):
                filename = file.replace(folder, "")
                logger.debug(f"{file} -> {filename}")
                keep_files.append(
                    filename
                )
                break

    return keep_files

def sync(src, dst, *args, **kwargs):
    logger.info(f"{src} -> {dst}")
    return sync_(src, dst, *args, **kwargs)


def download_analysis_results(rootdir, folder, version=2, ncores=-2):
    """
    Downloads the idtrackerai results stored in Dropbox
    """

    assert rootdir.startswith("/")
    assert folder.startswith("/")

    folder_display = folder.replace("/./", "/")
    subfolder = folder.split("/./")
    if len(subfolder) == 1:
        subfolder = ""
    else:
        subfolder = subfolder[1]

    assert "/./" not in folder_display
    res = list_folder(folder_display)
    files = res["files"]
    # dirs = res["dirs"]

    if version == 1:
        analysis_folder = folder_display
    elif version == 2:
        analysis_folder = os.path.join(folder_display, "idtrackerai")

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

    keep_files = match_files_to_patterns(folder_display, files, patterns)

    if len(keep_files) == 0:
        logger.warning(f"No files matching patterns in {folder_display}")
        return

    logger.debug(f"Files to be downloaded: {keep_files}")

    args = [
        (f"Dropbox:{folder_display}/{file}", os.path.join(rootdir, subfolder, file))
        for file in keep_files
    ]

    print(args)

    joblib.Parallel(n_jobs=ncores)(
        joblib.delayed(sync)(
            *arg
        )
            for arg in args
    )


def get_parser(ap=None):
    if ap is None:
        ap = argparse.ArgumentParser()

    ap.add_argument("--rootdir", required=True)
    ap.add_argument("--folder", required=True)
    ap.add_argument("--version", default=2, type=int)
    ap.add_argument("--ncores", default=-2, type=int)
    return ap


def main(args=None):

    if args is None:
        ap = get_parser()
        args = ap.parse_args()

    download_analysis_results(
        rootdir = args.rootdir,
        folder = args.folder,
        version=args.version,
        ncores=args.ncores
    )
