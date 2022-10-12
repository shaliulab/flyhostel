import logging
import glob
import os.path
import numpy as np
import trajectorytools.trajectories.concatenate as concatenate

logger = logging.getLogger(__name__)

def get_trajectory_files(experiment_folder):

    trajectories = concatenate.get_trajectories(
        experiment_folder
    )  # , allow_human=False)#, ignore_corrupt_chunks=True)
    trajectories_paths = [(k, v) for k, v in trajectories.items()]
    trajectories_paths = sorted(trajectories_paths, key=lambda x: x[0])
    trajectories_paths = [t[1] for t in trajectories_paths]
    trajectories_paths = [e for e in trajectories_paths if not "original" in e]

    return trajectories_paths


def load_trajectories(trajectories_paths, interval, **kwargs):


    chunks = [int(os.path.basename(p).replace(
        ".npy", ""
    )) for p in trajectories_paths]


    if interval is not None:
        assert len(interval) == 2
        indices = (chunks.index(interval[0]), chunks.index(interval[1]-1))
        chunks = chunks[indices[0]:indices[1]+1]
        trajectories_paths = trajectories_paths[indices[0]:indices[1]+1]
    
    
    timestamps_paths = [
        os.path.join(
            os.path.dirname(trajectories_paths[i]),
            f"{str(chunk).zfill(6)}.npz"
        )
        for i, chunk in enumerate(chunks)
    ]
    
    missing_timestamps_paths = [
        os.path.join(
            os.path.dirname(trajectories_paths[0]),
            f"{str(chunk).zfill(6)}.npz"
        )
        for chunk in list(range(0, chunks[0]))
    ]

    timestamps = []
    missing_timestamps = []
    
    for path in timestamps_paths:
        timestamps.extend(np.load(path, allow_pickle=True)["frame_time"] / 1000)

    for path in missing_timestamps_paths:
        missing_timestamps.extend(np.load(path, allow_pickle=True)["frame_time"] / 1000)
        

    timestamps=np.array(timestamps)

    status, tr = concatenate.from_several_idtracker_files(
        trajectories_paths,
        timestamps=timestamps,
        strict=False,
        zero_index=interval[0],
        **kwargs
    )

    # Since we have the frames per second stored int the tr.params dictionary we will use
    # tr.new_time_unit(tr.params["frame_rate"], "s")

    return (status, chunks, tr), (timestamps, missing_timestamps)

def pad_beginning_so_always_referenced_to_record_start(tr, missing_timestamps):
    
    number_of_points_missing = len(missing_timestamps)
    tr.pad_at_beginning(number_of_points_missing)
    tr._number_of_points_missing = number_of_points_missing
    return tr

