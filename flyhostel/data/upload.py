import argparse
import tqdm
import glob
import os.path
import subprocess
import joblib

def get_parser(ap=None):

    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", nargs="+", type=int)
    ap.add_argument("--chunks", nargs="+", type=int)
    ap.add_argument("--experiment-folder", dest="experiment_folder", type=str, required=True)
    ap.add_argument("--jobs", type=int, default=1)
    return ap


def upload_chunk(experiment_folder, chunk):

    files = sorted(glob.glob(os.path.join(experiment_folder, f"{str(chunk).zfill(6)}*")))
    files.append(f"session_{str(chunk).zfill(6)}-local_settings.py")
    experiment_name = os.path.basename(experiment_folder.rstrip("/"))

    for fullname in tqdm.tqdm(files):
        fullname = fullname.replace(experiment_name, f"{experiment_name}/./")
        name = os.path.basename(fullname)
        p = subprocess.Popen([
            "dropy",
            fullname,
            f"Dropbox:/Data/flyhostel_data/videos/{experiment_name}"
        ])

        p.communicate()

    
    folder = os.path.join(experiment_folder, ".", "idtrackerai", f"session_{str(chunk).zfill(6)}")
    subprocess.Popen([
        "dropy",
        folder,
        f"Dropbox:/Data/flyhostel_data/videos/{experiment_name}"
    ]).communicate()
    
    

def upload_chunks(experiment_folder, interval=None, chunks=None, jobs=5):

    if chunks is None:
        chunks = []

    for chunk in range(*interval):
        chunks.append(chunk)

    if jobs == 1:
        for chunk in chunks:
            upload_chunk(experiment_folder, chunk)
    else:
        joblib.Parallel(n_jobs=jobs)(
            joblib.delayed(upload_chunk)(experiment_folder, chunk)
            for chunk in chunks
        )

    return


def main(ap=None, args=None):

    if args is None:
        ap = get_parser(ap)
        args = ap.parse_args()


    return upload_chunks(
        experiment_folder=args.experiment_folder,
        interval=args.interval, chunks=args.chunks,
        jobs=args.jobs
    )

if __name__ == "__main__":
    main()
