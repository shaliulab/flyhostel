import logging
import argparse
import os.path

import joblib
import numpy as np
import h5py
from tqdm.auto import tqdm

import sleap_io as sio
# from movement.io import load_poses

# TODO Fix these imports
#from flyhostel.data.pose.constants import chunksize, framerate

logger=logging.getLogger(__name__)

def generate_filtered_labels(basedir, pose_tracks, keypoints, local_identity, chunk):
    labels_in=f"{basedir}/flyhostel/single_animal/{str(local_identity).zfill(3)}/{str(chunk).zfill(6)}.mp4.predictions.slp"
    labels = sio.load_slp(labels_in)
    
    # Create skeleton.
    skeleton = labels.skeleton
    
    # Create video.
    video = labels.videos[0]

    # Sort the keypoints as specified by the skeleton
    target=[node.name for node in skeleton.nodes]
    index_map = {value: i for i, value in enumerate(keypoints)}
    sorted_indices = [index_map[x] for x in target]

    instances=[]
    
    
    # Create instance.
    for i in range(pose_tracks.shape[0]):
        instances.append(
            sio.Instance.from_numpy(
                points=pose_tracks[i, 0, sorted_indices, :],
                skeleton=skeleton
            )
        )
    
    # Create labeled frame.
    labeled_frames = [sio.LabeledFrame(video=video, frame_idx=i, instances=[instances[i]]) for i in range(len(instances))]
    
    # Create labels.
    new_labels = sio.Labels(videos=[video], skeletons=[skeleton], labeled_frames=labeled_frames)
    labels_out = f"single_animal_filter/{str(local_identity).zfill(3)}/{str(chunk).zfill(6)}.mp4.predictions.slp"
    # labels_out = f"{basedir}/flyhostel/single_animal_filter/{str(local_identity).zfill(3)}/{str(chunk).zfill(6)}.mp4.predictions.slp"

    out_folder=os.path.dirname(labels_out)
    os.makedirs(out_folder, exist_ok=True)
    sio.save_slp(labels=new_labels, filename=labels_out)
    logger.info("Saved %s", labels_out)

    return labels_out


def export_filtered_pose(basedir, file_path, n_jobs=1):
    # ds=load_poses.from_sleap_file(file_path, fps=framerate)
    with h5py.File(file_path, "r") as f:
        files=[e.decode() for e in f["files"]]
        chunks=[int(os.path.basename(file).split(".")[0]) for file in files]
        local_identities=[int(os.path.basename(os.path.dirname(file)).split(".")[0]) for file in files]
        first_chunk=chunks[0]
        pose_tracks=f["tracks"][:].transpose(3, 0, 2, 1)
        keypoints=[e.decode() for e in f["node_names"][:]]

    
    pose_tracks_list=[
        pose_tracks[(chunksize*(chunk-first_chunk)):(chunksize*(chunk-first_chunk+1)), ...]
        for chunk in chunks
    ]

    out=joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(
            generate_filtered_labels
        )(
            basedir, pose_tracks_list[i], keypoints=keypoints, local_identity=local_identity, chunk=chunk
        )
        for i, (local_identity, chunk) in enumerate(tqdm(zip(local_identities, chunks)))
    )


def get_parser():

    ap = argparse.ArgumentParser()
    ap.add_argument("--basedir")
    ap.add_argument("--file-path")
    ap.add_argument("--n-jobs", default=1, type=int)
    return ap

def main():
    ap = get_parser()
    args=ap.parse_args()
    export_filtered_pose(args.basedir, args.file_path, n_jobs=args.n_jobs)
