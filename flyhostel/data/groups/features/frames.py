import logging
import traceback
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from .data_structures import GroupFrame
from .utils import (
    _build_frame,
    _preextract_arrays,
)

logger=logging.getLogger(__name__)

def build_frames_parallel(
    pose_features,
    fps: float,
    step_seconds: float = 5.0,
    n_workers: int | None = None,
    chunksize: int = 64,
) -> list[GroupFrame]:
    """
    Fast, parallel replacement for the original loop.
    Now also handles distance features.
    """
    step_frames = max(1, int(fps * step_seconds))
    n_workers = n_workers or cpu_count()

    print("Extracting arrays from xarray...", flush=True)
    (positions, times, frame_numbers,
     food_distance, notch_distance, edge_distance,
     n_individuals, kp_index, n_legs) = _preextract_arrays(pose_features, step_frames)

    # Build argument tuples
    try:
        args = [
            (t_idx, times[t_idx], frame_numbers[t_idx], positions[t_idx],
             food_distance[t_idx], notch_distance[t_idx], edge_distance[t_idx],
             n_individuals, kp_index, n_legs)
            for t_idx in range(len(times))
        ]
    except Exception as e:
        logger.error(f"Error building args: {e}")
        logger.error(traceback.print_exc())
        args = []

    print(f"Building {len(args)} frames across {n_workers} workers...", flush=True)
    if n_workers == 1:
        frames = [_build_frame(arg) for arg in args]
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