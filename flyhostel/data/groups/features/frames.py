from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from .data_structures import GroupFrame
from .utils import (
    _build_frame,
    _preextract_arrays,
)

def build_frames_parallel(
    pose_features,
    fps: float,
    step_seconds: float = 5.0,
    n_workers: int | None = None,
    chunksize: int = 64,
) -> list[GroupFrame]:
    """
    Fast, parallel replacement for the original loop.

    Parameters
    ----------
    pose_features : xarray.Dataset with dims (time, individuals, keypoints, space)
    fps           : frames per second of the recording
    step_seconds  : temporal subsampling in seconds (default 5 s)
    n_workers     : number of parallel workers (default = all CPUs)
    chunksize     : tasks per worker batch (tune for your frame count)
    """
    step_frames = max(1, int(fps * step_seconds))
    n_workers   = n_workers or cpu_count()

    print("Extracting arrays from xarray...", flush=True)
    positions, times, frame_numbers, individuals, kp_index, n_legs = _preextract_arrays(
        pose_features, step_frames
    )

    # positions: (T, 2, K, N) — already a plain numpy array, cheap to slice

    # Build argument tuples — each worker gets one (T, 2, K, N) → (2, K, N) slice
    try:
        args = [
            (t_idx, times[t_idx], frame_numbers[t_idx], positions[t_idx], n_legs, kp_index)
            for t_idx in range(len(times))
        ]
    except:
        args=[]


    print(f"Building {len(args)} frames across {n_workers} workers...", flush=True)
    if n_workers==1:
        frames=[]
        for arg in args:
            frames.append(_build_frame(arg))

    else:
        with Pool(processes=n_workers) as pool:
            frames = list(
                tqdm(
                    pool.imap(_build_frame, args, chunksize=chunksize),
                    total=len(args),
                    desc="Frames",
                )
            )

    return frames
