#! /usr/bin/python

import argparse
import warnings
import os.path
import pathlib
import http.server
import socketserver
import time
import os
import signal
import sys
PORT = 8000
VIDEO_DATABASE="/flyhostel_data/videos"

import joblib
from imgstore.interface import VideoCapture

#import logging
#import logging.config


#logging.basicConfig(level=logging.INFO)
#handler = logging.StreamHandler(sys.stdout)
#stderr_hdlr = logging.StreamHandler(sys.stderr)
#rootLogger = logging.getLogger()

#rootLogger.addHandler(handler)
#rootLogger.addHandler(stderr_hdlr)
#logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

# local_settings.py
local_settings="""
SETTINGS_PRIORITY=1
COLOR=False
READ_FORMAT="imgstore"
MULTI_STORE_ENABLED=False
NUMBER_OF_JOBS_FOR_BACKGROUND_SUBTRACTION=6
NUMBER_OF_JOBS_FOR_CONNECTING_BLOBS=6
NUMBER_OF_JOBS_FOR_SEGMENTATION=6
NUMBER_OF_JOBS_FOR_SETTING_ID_IMAGES=1
RECONNECT_BLOBS_FROM_CACHE=True
POSTPROCESS_IMPOSSIBLE_JUMPS=False
DISABLE_PROTOCOL_3=True
IDTRACKERAI_IS_TESTING=False
SKIP_SAVING_IDENTIFICATION_IMAGES=False
TIME_FORMAT="H|CF"
DATA_POLICY="remove_segmentation_data"
SKIP_EVERY_FRAME=1
CHUNK=0
"""


Handler = http.server.SimpleHTTPRequestHandler

def generate_index(database):
    n_jobs=10
    INDEX_FILE = os.path.join(database, "index.txt")
    print(f"Writing index to {INDEX_FILE}")
    videos = pathlib.Path(database)
    entries = videos.rglob("metadata.yaml")
    entries = sorted(list(entries))

    try:
        with open(INDEX_FILE, "w") as filehandle:
            for entry in entries:
                filehandle.write(f"{entry}\n")
    except:
        warnings.warn(f"Cannot generate {INDEX_FILE}")

    print(f"Wrote {len(entries)} entries to index")
    prepare_experiment(entries, n_jobs=n_jobs)

    return entries 


def prepare_experiment(entries, n_jobs):
    
    
    def prepare_experiment_one_file(metadata):
        if "highspeed" in metadata and "2022" in metadata:
            return
        elif "2022" in metadata:
            return

        else:
            print(metadata)
            idtrackerai_folder = os.path.join(os.path.dirname(metadata), "idtrackerai")
            video_annotator_folder=os.path.join(os.path.dirname(metadata), "video-annotator")
            
            os.makedirs(idtrackerai_folder, exist_ok=True)
            os.makedirs(video_annotator_folder, exist_ok=True)
            local_settings_path=os.path.join(idtrackerai_folder, "local_settings.py")
            if not os.path.exists(local_settings_path):
                with open(local_settings_path, "w") as filehandle:
                    filehandle.write(local_settings)

            return VideoCapture(metadata, 0)


    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            prepare_experiment_one_file
            )(str(metadata))
        for metadata in entries
    )


def setup_http_server(database):
    
    os.chdir(database)
    running=False
    port=PORT
    
    while not running:
        try:
            with socketserver.TCPServer(("", port), Handler) as httpd:
                running=True
                print(f"serving at port {port}")
                httpd.serve_forever()
                return 0
        except OSError as error:
            print(f"Fail to set up http server")
            print(error)
        if not running: port+=1


def generate_index_main(database):

    entries = generate_index(database)
    print("Done")
    now = time.time()
    last_time = now
    entries=None

    while True:
        now = time.time()
        if now % 3600 == 0 and (now - last_time) > 3590:
            last_time = now
            entries = generate_index(database)
        else:
            time.sleep(1)
    return entries

 
def exitHandler(signalnumb, frame):
    print("Quitting")
    sys.exit(0)

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--http-server", action="store_true", default=False)
    ap.add_argument("--index", action="store_true", default=False)
    args = ap.parse_args()
    signal.signal(signal.SIGINT, exitHandler)

    if args.http_server:
        setup_http_server(VIDEO_DATABASE)

    if args.index:
        generate_index_main(VIDEO_DATABASE)
    else:
        return


print(__name__)

if __name__ == "__main__":
    main()
