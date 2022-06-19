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


def load_trajectories(trajectories_paths, interval, timestamps_paths=None, **kwargs):


    chunks = [int(os.path.basename(p).replace(
        ".npy", ""
    )) for p in trajectories_paths]


    if interval is not None:
        assert len(interval) == 2
        indices = (chunks.index(interval[0]), chunks.index(interval[1]-1))
        chunks = chunks[indices[0]:indices[1]+1]
        trajectories_paths = trajectories_paths[indices[0]:indices[1]+1]
        if timestamps_paths is not None:
            timestamps_paths = timestamps_paths[indices[0]:indices[1]+1]


    if timestamps_paths is not None:
        timestamps = []
        for path in timestamps_paths:
            timestamps.extend(np.load(path, allow_pickle=True)["frame_time"] / 1000)

    status, tr = concatenate.from_several_idtracker_files(
        trajectories_paths,
        timestamps=np.array(timestamps),
        strict=False,
        zero_index=interval[0],
        **kwargs
    )
    logger.info(
        "flyhostel has loaded" \
        f"{round((tr._s.shape[0]+2)/3600/12, 2)} hours of data successfully",
    )  # / seconds in hour and frames in second

    # Since we have the frames per second stored int the tr.params dictionary we will use
    # tr.new_time_unit(tr.params["frame_rate"], "s")

    return (status, chunks, tr)
