import logging

import trajectorytools.trajectories.concatenate as concatenate

logger = logging.getLogger(__name__)

def get_trajectory_files(experiment_folder):

    trajectories = concatenate.get_trajectories(
        experiment_folder
    )  # , allow_human=False)#, ignore_corrupt_chunks=True)
    chunks = [int(k.replace("session_", "")) for k in trajectories]
    trajectories_paths = [(k, v) for k, v in trajectories.items()]
    trajectories_paths = sorted(trajectories_paths, key=lambda x: x[0])
    trajectories_paths = [t[1] for t in trajectories_paths]
    trajectories_paths = [e for e in trajectories_paths if not "original" in e]
    return chunks, trajectories_paths


def load_trajectories(experiment_folder, interval):

    chunks, trajectories_paths = get_trajectory_files(experiment_folder)
    
    if interval is not None:
        assert len(interval) == 2
        indices = (chunks.index(interval[0]), chunks.index(interval[1]-1))
        chunks = chunks[indices[0]:indices[1]]
        trajectories_paths = trajectories_paths[indices[0]:indices[1]-1]

    status, tr = concatenate.from_several_idtracker_files(
        trajectories_paths, strict=False,
        zero_index=interval[0]
    )
    logger.info(
        "flyhostel has loaded",
        f"{(tr._s.shape[0]+2)/3600/12} hours of data successfully",
    )  # / seconds in hour and frames in second

    # Since we have the frames per second stored int the tr.params dictionary we will use
    # tr.new_time_unit(tr.params["frame_rate"], "s")

    return status, chunks, tr
