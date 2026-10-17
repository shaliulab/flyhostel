"""
Data layer: turn a list of FlyHostelLoaders (one experiment) into arrays the pipelines
can vectorise over.

* GroupTracks  dense centroid positions of all N flies on ONE regular frame grid,
               shape (T, N, 2), arena mm. Built once per experiment; everything
               centroid-based (proximity, movement, sleep, approach attribution) uses it.
* PoseReader   random access to one fly's RAW SLEAP pose in arena mm, on the same grid.
               Pose is only ever read around candidate events, never for the whole
               recording, so memory stays bounded however long the experiment is.
"""
import itertools
import logging
import os
from dataclasses import dataclass

import h5py
import joblib
import numpy as np

from flyhostel.utils.pose_export import get_first_frame_number
from .runs import ffill_short_gaps, frames

logger = logging.getLogger(__name__)


@dataclass
class GroupTracks:
    experiment: str
    ids: list                  # N fly ids as in loader.ids[0], e.g. "FlyHostel1_6X_2025-10-02_1|01"
    animals: list              # N dataset names, e.g. "FlyHostel1_6X_2025-10-02_16-00-00__01"
    identities: list           # N int identities
    frame_number: np.ndarray   # (T,) int64, regular grid with step `stride`
    t: np.ndarray              # (T,) float64 ZT seconds
    xy: np.ndarray             # (T, N, 2) float32 centroid, arena mm, NaN = not tracked
    fps: float                 # sampling rate of the grid
    stride: int                # native frames per grid step
    chunksize: int
    pixels_per_mm: float

    @property
    def n_frames(self):
        return len(self.frame_number)

    @property
    def n_flies(self):
        return len(self.ids)

    @property
    def pairs(self):
        return list(itertools.combinations(range(self.n_flies), 2))

    def to_index(self, frame_number):
        """Grid index of frame_number(s), floored to the grid and clipped to its range."""
        idx = (np.asarray(frame_number, np.int64) - self.frame_number[0]) // self.stride
        return np.clip(idx, 0, self.n_frames - 1)

    def end_frame(self, end_idx):
        """Exclusive frame_number matching an exclusive grid index."""
        return self.frame_number[np.asarray(end_idx) - 1] + self.stride

    def frames(self, seconds, minimum=0):
        return frames(seconds, self.fps, minimum)


def _fly_centroids(loader, stride, scale_attr, min_time=None, max_time=None):
    """Centroid of one fly on frames that are multiples of ``stride``, in arena mm."""
    dt = loader.dt
    if dt is None:
        loader.load_centroid_data(min_time=min_time, max_time=max_time)
        dt, loader.dt = loader.dt, None   # the arrays below are all we keep
    if dt is None:
        raise ValueError(f"No centroid data for {loader}")

    fn = dt["frame_number"].to_numpy(np.int64)
    t = dt["t"].to_numpy(np.float64)
    keep = (fn % stride) == 0
    if min_time is not None:
        keep &= t >= min_time
    if max_time is not None:
        keep &= t < max_time
    ppm = float(getattr(loader, scale_attr))
    return dict(
        frame_number=fn[keep],
        t=t[keep],
        x=(dt["center_x"].to_numpy(np.float64)[keep] / ppm).astype(np.float32),
        y=(dt["center_y"].to_numpy(np.float64)[keep] / ppm).astype(np.float32),
    )


def load_group_tracks(loaders, cfg, min_time=None, max_time=None, n_jobs=1):
    """Build GroupTracks for flies that share one arena.

    Arguments:
        loaders (list of FlyHostelLoader): flies of ONE experiment. Loaders that already
            have ``dt`` loaded are used as is; the rest are loaded, in parallel over flies
            when n_jobs > 1.
        cfg (TrackConfig)
        min_time, max_time (float): ZT seconds, optional
    """
    experiments = sorted({loader.experiment for loader in loaders})
    if len(experiments) != 1:
        raise ValueError(f"GroupTracks needs flies of one experiment, got {experiments}")
    loaders = sorted(loaders, key=lambda loader: loader.identity)

    framerate = float(loaders[0].framerate)
    stride = max(1, int(round(framerate / cfg.fps)))
    fps = framerate / stride

    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(_fly_centroids)(loader, stride, cfg.scale_attr, min_time, max_time)
        for loader in loaders
    )

    fn0 = min(r["frame_number"].min() for r in results)
    fn1 = max(r["frame_number"].max() for r in results)
    grid = np.arange(fn0, fn1 + 1, stride, dtype=np.int64)

    xy = np.full((len(grid), len(loaders), 2), np.nan, dtype=np.float32)
    t = np.full(len(grid), np.nan)
    for i, r in enumerate(results):
        idx = (r["frame_number"] - fn0) // stride
        xy[idx, i, 0] = r["x"]
        xy[idx, i, 1] = r["y"]
        t[idx] = np.where(np.isnan(t[idx]), r["t"], t[idx])

    xy = ffill_short_gaps(xy, frames(cfg.max_gap_s, fps))
    ok = np.isfinite(t)
    t = np.interp(grid, grid[ok], t[ok])

    tracks = GroupTracks(
        experiment=experiments[0],
        ids=[loader.ids[0] for loader in loaders],
        animals=[loader.datasetnames[0] for loader in loaders],
        identities=[int(loader.identity) for loader in loaders],
        frame_number=grid, t=t, xy=xy, fps=fps, stride=stride,
        chunksize=int(loaders[0].chunksize),
        pixels_per_mm=float(getattr(loaders[0], cfg.scale_attr)),
    )
    missing = np.isnan(xy[..., 0]).mean(axis=0)
    logger.info("%s: %d frames at %.1f fps, missing fraction per fly %s",
                tracks.experiment, tracks.n_frames, fps, np.round(missing, 3).tolist())
    return tracks


class PoseReader:
    """Windowed access to one fly's raw pose .h5 (the file the PE pipeline reads).

    read(start_fn, end_fn) returns keypoints on the analysis grid in ARENA mm:
    crop coordinates + the per-frame ``anchor`` (crop top-left in arena pixels).
    Keypoints whose SLEAP point score is below ``min_score[node]`` are NaN.

    The .h5 tracks are gzip-chunked by video chunk; a large chunk cache means windows
    read in time order decompress each chunk once. The file handle is opened lazily
    and dropped on pickling, so readers can be shipped to joblib workers.
    """

    def __init__(self, path, chunksize, pixels_per_mm, stride, nodes=None, min_score=None, cache_mb=256):
        self.path = path
        self.pixels_per_mm = float(pixels_per_mm)
        self.stride = int(stride)
        self.cache_mb = cache_mb
        self._file = None
        with h5py.File(path, "r") as f:
            names = [n.decode() for n in f["node_names"][:]]
            if "anchor" not in f:
                raise KeyError(
                    f"{path} has no 'anchor' dataset. Regenerate it with "
                    "loader.get_pose_file_h5py('raw', dt=loader.dt)"
                )
            self.n_rows = f["tracks"].shape[3]
        self.first_fn = get_first_frame_number(path, chunksize)
        self.nodes = [n for n in (nodes or names) if n in names]
        self._node_idx = np.array([names.index(n) for n in self.nodes])
        min_score = min_score or {}
        self._min_score = np.array([min_score.get(n, 0.0) for n in self.nodes], dtype=np.float32)

    @classmethod
    def from_loader(cls, loader, stride, scale_attr="pixels_per_mm", **kwargs):
        path = loader.get_pose_file_path("raw")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return cls(path, loader.chunksize, getattr(loader, scale_attr), stride, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    @property
    def file(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r", rdcc_nbytes=self.cache_mb * 1024 ** 2, rdcc_nslots=10007)
        return self._file

    def read(self, start_fn, end_fn):
        """Pose on the grid frames start_fn, start_fn + stride, ... < end_fn.

        Returns (frame_number (L,), xy (L, K, 2) float32 arena mm). Frames outside the
        file are NaN, so two flies read over the same range are always aligned.
        """
        fns = np.arange(start_fn, end_fn, self.stride, dtype=np.int64)
        out = np.full((len(fns), len(self.nodes), 2), np.nan, dtype=np.float32)
        rows = fns - self.first_fn
        ok = (rows >= 0) & (rows < self.n_rows)
        if ok.any():
            a, b = int(rows[ok][0]), int(rows[ok][-1]) + 1
            tracks = self.file["tracks"][0, :, :, a:b:self.stride]          # (2, nodes, L)
            scores = self.file["point_scores"][0, :, a:b:self.stride]       # (nodes, L)
            anchor = self.file["anchor"][a:b:self.stride]                   # (L, 2)
            xy = tracks[:, self._node_idx, :].transpose(2, 1, 0) + anchor[:, None, :]
            xy[scores[self._node_idx, :].T < self._min_score] = np.nan
            out[ok] = xy / self.pixels_per_mm
        return fns, out


def make_pose_readers(loaders, tracks, cfg_tracks, cfg_contact, nodes=None):
    """One PoseReader per fly id. Flies without a raw pose file are skipped with a warning."""
    readers = {}
    for loader in loaders:
        try:
            readers[loader.ids[0]] = PoseReader.from_loader(
                loader, tracks.stride, scale_attr=cfg_tracks.scale_attr,
                nodes=nodes, min_score=cfg_contact.min_score,
            )
        except (FileNotFoundError, KeyError) as error:
            logger.warning("No usable pose for %s: %s", loader, error)
    return readers
