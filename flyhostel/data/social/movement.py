"""
Per-fly movement state on the GroupTracks grid: velocity, moving/immobile, time since the
fly last moved, and sleep bouts (immobility >= min_sleep_s, the 5-min rule).

One definition of "immobile" is used everywhere (approach attribution, the recipient's
pre-contact immobility, sleep latency), so the numbers in the output tables agree with
each other. ``asleep`` can be replaced by the lab's behaviour-based sleep annotation
with ``asleep_from_loaders``.
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import centered_velocity
from .runs import apply_columns, drop_short, runs

logger = logging.getLogger(__name__)


@dataclass
class Movement:
    velocity: np.ndarray        # (T, N, 2) mm/s
    speed: np.ndarray           # (T, N) mm/s, NaN where untracked
    moving: np.ndarray          # (T, N) bool. Untracked frames count as moving: we cannot
                                #   claim immobility we did not observe
    since_moving_s: np.ndarray  # (T, N) seconds since the last moving frame (causal)
    asleep: np.ndarray          # (T, N) bool, whole immobility bout >= min_sleep_s
    sleep_bouts: pd.DataFrame   # fly, start_idx, end_idx (exclusive), duration_s

    def next_sleep_onset(self, fly, idx):
        """Grid index where the sleep bout containing or following ``idx`` starts.

        Returns -1 if the fly never sleeps again. The onset is the START of the immobility
        bout, not the moment it reaches min_sleep_s, so it can precede ``idx`` when the fly
        stopped moving before ``idx`` and stayed still.
        """
        bouts = self.sleep_bouts[self.sleep_bouts["fly"] == fly]
        k = np.searchsorted(bouts["end_idx"].to_numpy(), idx, side="right")
        return int(bouts["start_idx"].iloc[k]) if k < len(bouts) else -1


def _since(flag, fps):
    """Seconds since the last True of ``flag`` along axis 0; counts from the start if never."""
    n = flag.shape[0]
    pos = np.where(flag, np.arange(n)[:, None], -1)
    last = np.maximum.accumulate(pos, axis=0)
    return ((np.arange(n)[:, None] - last) / fps).astype(np.float32)


def _sleep_bouts(asleep, fps):
    rows = []
    for fly in range(asleep.shape[1]):
        starts, ends = runs(asleep[:, fly])
        rows.append(pd.DataFrame({"fly": fly, "start_idx": starts, "end_idx": ends}))
    bouts = pd.concat(rows, ignore_index=True)
    bouts["duration_s"] = (bouts["end_idx"] - bouts["start_idx"]) / fps
    return bouts


def compute_movement(tracks, cfg, asleep=None):
    """
    Arguments:
        tracks (GroupTracks)
        cfg (MovementConfig)
        asleep (np.ndarray): optional (T, N) bool sleep annotation on the tracks grid
            (e.g. from ``asleep_from_loaders``). By default sleep = immobility >= min_sleep_s.
    """
    fps = tracks.fps
    velocity = centered_velocity(tracks.xy, fps, cfg.smooth_s)
    speed = np.linalg.norm(velocity, axis=-1)
    with np.errstate(invalid="ignore"):
        moving = ~(speed <= cfg.move_speed_mm_s)          # NaN speed -> moving
    moving = apply_columns(drop_short, moving, tracks.frames(cfg.ignore_move_s, 1))

    if asleep is None:
        min_len = tracks.frames(cfg.min_sleep_s, 1)
        asleep = apply_columns(drop_short, ~moving, min_len)

    return Movement(
        velocity=velocity, speed=speed, moving=moving,
        since_moving_s=_since(moving, fps),
        asleep=asleep, sleep_bouts=_sleep_bouts(asleep, fps),
    )


def asleep_from_loaders(loaders, tracks, min_time_immobile=300):
    """Lab-standard sleep (behaviour-based, 1 Hz) projected onto the tracks grid.

    Uses ``loader.load_sleep_data(bin_size=None)``; flies without a sleep file fall back
    to all-False with a warning.
    """
    by_id = {loader.ids[0]: loader for loader in loaders}
    asleep = np.zeros((tracks.n_frames, tracks.n_flies), dtype=bool)
    grid = pd.DataFrame({"frame_number": tracks.frame_number})
    for i, fly_id in enumerate(tracks.ids):
        loader = by_id[fly_id]
        try:
            loader.load_sleep_data(bin_size=None, min_time_immobile=min_time_immobile)
        except FileNotFoundError as error:
            logger.warning("No sleep annotation for %s (%s)", loader, error)
            continue
        sleep = loader.sleep[["frame_number", "asleep"]].sort_values("frame_number")
        sleep["frame_number"] = sleep["frame_number"].astype(np.int64)
        hit = pd.merge_asof(grid, sleep, on="frame_number", direction="backward",
                            tolerance=int(2 * tracks.fps * tracks.stride))
        asleep[:, i] = hit["asleep"].fillna(False).astype(bool).to_numpy()
    return asleep
