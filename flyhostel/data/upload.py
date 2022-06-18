import argparse
import tqdm
import glob
import os.path
import subprocess
import joblib

def upload_chunk(experiment_folder, chunk):

    files = sorted(glob.glob(os.path.join(experiment_folder.rstrip("/"), f"{str(chunk).zfill(6)}*")), reverse=True)
    files.append(os.path.join(experiment_folder.rstrip("/"), f"session_{str(chunk).zfill(6)}-local_settings.py"))
    experiment_name = os.path.basename(experiment_folder.rstrip("/"))

    for fullname in tqdm.tqdm(files):
        fullname = fullname.replace(experiment_name, f"{experiment_name}/.")
        name = os.path.basename(fullname)
        cmd = [
            "dropy",
            fullname,
            f"Dropbox:/Data/flyhostel_data/videos/{experiment_name}/"
        ]
        print(cmd)
        p = subprocess.Popen(cmd)
        p.communicate()


    folder = os.path.join(experiment_folder, ".", "idtrackerai", f"session_{str(chunk).zfill(6)}")
    cmd = [
        "dropy",
        folder,
        f"Dropbox:/Data/flyhostel_data/videos/{experiment_name}/"
    ]

    print(cmd)
    subprocess.Popen(cmd).communicate()


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



if __name__ == "__main__":
    main()
