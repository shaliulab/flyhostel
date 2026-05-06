"""
Merge spuriously split contact intervals in FlyHostel tracking data.

Intervals are loaded from a CSV produced by an upstream pipeline (which
detects breaks of the idtracker.ai heuristics). This module then:

  1. For each interval, identifies the candidate pair of flies by position
     at the bracketing clean frames (entry / exit).
  2. Merges consecutive intervals that are:
        - temporally close (gap_frames <= delta_time)
        - spatially continuous: the pair at interval i's exit can be
          forward-tracked through the gap to the pair at interval i+1's
          entry (within a position tolerance)
        - and stayed close together throughout the gap
          (max pairwise distance <= delta_space)

Identity continuity is checked by spatial propagation (nearest-neighbor
across gap frames where all N flies are resolved), NOT by the IDENTITY
table, because identity labels can swap across contact events.

CSV format
----------
Expected columns: at minimum `scene_start` (first failure frame) and
`length` (number of contiguous failure frames). Any other columns are
preserved as `Interval.metadata` and survive merging (with values from
the most recent merged child).

Performance notes
-----------------
ROI_0 can have millions of rows. Per-frame position lookups (needed only
at interval boundaries and within candidate merge gaps) are done lazily
via an indexed query, with an LRU cache. The raw table is never bulk-loaded.

A tqdm progress bar is shown on the two loops that drive per-frame
queries. tqdm is optional — if not installed, a no-op fallback is used.
"""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Interval:
    """A contiguous run of frames flagged as problematic by the upstream pipeline."""
    start_frame: int                    # first failure frame (== scene_start)
    end_frame: int                      # last failure frame  (== scene_start + length - 2)
    entry_frame: int                    # last clean frame before (start_frame - 1)
    exit_frame: int                     # first clean frame after  (end_frame + 1)
    entry_positions: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    exit_positions: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    children: list[tuple[int, int]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)   # other CSV columns

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame + 1


# ---------------------------------------------------------------------------
# Loading intervals from the upstream CSV
# ---------------------------------------------------------------------------

def load_intervals_from_csv(csv_path: str,
                            metadata_cols: Optional[list[str]] = None
                            ) -> list[Interval]:
    """
    Read intervals from the CSV produced by the upstream pipeline.

    Parameters
    ----------
    csv_path : str
        Path to the CSV.
    metadata_cols : list[str] or None
        Names of additional columns to attach to each interval's `metadata`
        dict. If None, all columns other than `scene_start` and `length`
        are preserved.
    """
    df = pd.read_csv(csv_path)
    df=df.loc[df["pass"]==False]
    required = {"scene_start", "length"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    if metadata_cols is None:
        metadata_cols = [c for c in df.columns
                         if c not in required and not c.startswith("Unnamed")]

    df = df.sort_values("scene_start").reset_index(drop=True)

    intervals: list[Interval] = []
    for _, row in df.iterrows():
        start = int(row["scene_start"])
        length = int(row["length"])
        # Convention: number of failure frames is (length - 1),
        # so the last failure frame is scene_start + length - 2.
        end = start + length - 2
        meta = {col: row[col] for col in metadata_cols if col in df.columns}
        intervals.append(Interval(
            start_frame=start,
            end_frame=end,
            entry_frame=start - 1,
            exit_frame=end + 1,
            children=[(start, end)],
            metadata=meta,
        ))
    return intervals


# ---------------------------------------------------------------------------
# RoiStore — lazy, cached access to ROI_0
# ---------------------------------------------------------------------------

class RoiStore:
    """
    Wraps a SQLite connection and provides per-frame (x, y) lookups on
    ROI_0 with an LRU cache. Ensures an index on frame_number exists.
    """

    def __init__(self,
                 db_path: str,
                 cache_size: int = 4096):
        self.db_path = db_path
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = cache_size
        self._conn = sqlite3.connect(db_path)
        self._ensure_index()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "RoiStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _ensure_index(self) -> None:
        cur = self._conn.cursor()
        cur.execute("PRAGMA index_list('ROI_0')")
        existing = cur.fetchall()
        has_frame_idx = False
        for row in existing:
            idx_name = row[1]
            cur.execute(f"PRAGMA index_info('{idx_name}')")
            cols = cur.fetchall()
            if cols and cols[0][2] == "frame_number":
                has_frame_idx = True
                break
        if not has_frame_idx:
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_roi0_frame "
                "ON ROI_0(frame_number)"
            )
            self._conn.commit()

    def positions_with_local_identity(self, frame: int) -> pd.DataFrame:
        """Returns a DataFrame with columns [local_identity, x, y] for a single frame,
        joining ROI_0 and IDENTITY on (frame_number, in_frame_index)."""
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT i.local_identity, r.x, r.y
            FROM ROI_0 r
            INNER JOIN IDENTITY i
            ON r.in_frame_index = i.in_frame_index
            AND i.frame_number = r.frame_number
            WHERE r.frame_number = ?
            AND i.frame_number = ?
            ORDER BY r.in_frame_index
            """,
            (frame, frame),
        )
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=["local_identity", "x", "y"])

        
    def positions(self, frame: int) -> np.ndarray:
        """(M, 2) array of (x, y) positions at a single frame."""
        cached = self._cache.get(frame)
        if cached is not None:
            self._cache.move_to_end(frame)
            return cached
        cur = self._conn.cursor()
        cur.execute(
            "SELECT x, y FROM ROI_0 WHERE frame_number = ? "
            "ORDER BY in_frame_index",
            (frame,),
        )
        rows = cur.fetchall()
        arr = np.array(rows, dtype=float) if rows else np.empty((0, 2))
        self._cache[frame] = arr
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return arr

    def positions_range(self, start_frame: int, end_frame: int) -> dict[int, np.ndarray]:
        """Bulk fetch positions for frames in [start_frame, end_frame] inclusive."""
        if end_frame < start_frame:
            return {}
        cur = self._conn.cursor()
        cur.execute(
            "SELECT frame_number, x, y FROM ROI_0 "
            "WHERE frame_number BETWEEN ? AND ? "
            "ORDER BY frame_number, in_frame_index",
            (start_frame, end_frame),
        )
        raw: dict[int, list[tuple[float, float]]] = {}
        for f, x, y in cur.fetchall():
            raw.setdefault(f, []).append((x, y))
        arrays: dict[int, np.ndarray] = {}
        for f in range(start_frame, end_frame + 1):
            arr = np.array(raw[f], dtype=float) if f in raw else np.empty((0, 2))
            arrays[f] = arr
            self._cache[f] = arr
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return arrays


# ---------------------------------------------------------------------------
# Candidate pair identification
# ---------------------------------------------------------------------------

def _closest_pair(positions: np.ndarray) -> np.ndarray:
    if positions.shape[0] < 2:
        raise ValueError("Need at least 2 positions to find a pair.")
    diffs = positions[:, None, :] - positions[None, :, :]
    d2 = (diffs ** 2).sum(axis=-1)
    np.fill_diagonal(d2, np.inf)
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    return positions[[i, j]]


def annotate_candidate_pairs(intervals: list[Interval],
                             store: RoiStore,
                             show_progress: bool = True) -> list[Interval]:
    """Populate entry_positions and exit_positions for each interval."""
    it = tqdm(intervals, desc="Annotating pairs",
              disable=not show_progress, unit="iv")
    for iv in it:
        entry_pos = store.positions(iv.entry_frame)
        exit_pos = store.positions(iv.exit_frame)
        try:
            iv.entry_positions = _closest_pair(entry_pos)
            iv.exit_positions = _closest_pair(exit_pos)
        except ValueError:
            pass  # <2 flies at boundary; will fail downstream matching
    return intervals


# ---------------------------------------------------------------------------
# Forward propagation across a gap
# ---------------------------------------------------------------------------

def _hungarian_step(prev_positions: np.ndarray,
                    current_positions: np.ndarray) -> np.ndarray:
    diffs = prev_positions[:, None, :] - current_positions[None, :, :]
    cost = (diffs ** 2).sum(axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    return current_positions[col_ind]


def propagate_pair(store: RoiStore,
                   seed_positions: np.ndarray,
                   seed_frame: int,
                   target_frame: int,
                   max_step_px: Optional[float] = None
                   ) -> Optional[np.ndarray]:
    """
    Forward-track two flies from seed_frame to target_frame. Returns a
    (T, 2, 2) array of per-frame positions, or None if propagation was
    ambiguous (missing blobs or implausibly large per-frame step).
    """
    assert seed_positions.shape == (2, 2)
    if target_frame <= seed_frame:
        return np.empty((0, 2, 2))

    frames_dict = store.positions_range(seed_frame + 1, target_frame)

    tracks = np.empty((target_frame - seed_frame, 2, 2))
    current = seed_positions
    for offset, frame in enumerate(range(seed_frame + 1, target_frame + 1)):
        pos = frames_dict[frame]
        if pos.shape[0] < 2:
            return None
        matched = _hungarian_step(current, pos)
        if max_step_px is not None:
            step = np.linalg.norm(matched - current, axis=-1)
            if (step > max_step_px).any():
                return None
        tracks[offset] = matched
        current = matched
    return tracks


# ---------------------------------------------------------------------------
# Merge predicate
# ---------------------------------------------------------------------------

def _positions_match(a: np.ndarray, b: np.ndarray, tol_px: float) -> bool:
    diffs = a[:, None, :] - b[None, :, :]
    cost = (diffs ** 2).sum(axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_dists = np.sqrt(cost[row_ind, col_ind])
    return bool((matched_dists <= tol_px).all())


def _max_pairwise_distance(tracks: np.ndarray) -> float:
    if tracks.shape[0] == 0:
        return 0.0
    d = np.linalg.norm(tracks[:, 0, :] - tracks[:, 1, :], axis=-1)
    return float(d.max())


def should_merge(prev: Interval,
                 curr: Interval,
                 store: RoiStore,
                 delta_time: int,
                 delta_space_px: float,
                 match_tol_px: float,
                 max_step_px: Optional[float] = None,
                 metadata_predicate: Optional[Callable[[Interval, Interval], bool]] = None,
                 ) -> tuple[bool, dict]:
    """
    Decide whether two consecutive intervals belong to the same episode.

    `metadata_predicate(prev, curr) -> bool` is an optional hook to veto a
    merge based on CSV metadata (e.g. don't merge across an interval where
    `pass=True`). Return False to forbid the merge.
    """
    diag: dict = {
        "gap_frames": None,
        "max_dist_during_gap": None,
        "propagation_ok": None,
        "pair_match": None,
        "reason": None,
    }
    gap_frames = curr.start_frame - prev.end_frame - 1
    diag["gap_frames"] = gap_frames

    if gap_frames < 0:
        diag["reason"] = "overlap (should not happen)"
        return False, diag
    if gap_frames > delta_time:
        diag["reason"] = "gap too long"
        return False, diag

    if metadata_predicate is not None and not metadata_predicate(prev, curr):
        diag["reason"] = "vetoed by metadata predicate"
        return False, diag

    tracks = propagate_pair(
        store,
        seed_positions=prev.exit_positions,
        seed_frame=prev.exit_frame,
        target_frame=curr.entry_frame,
        max_step_px=max_step_px,
    )
    if tracks is None:
        diag["propagation_ok"] = False
        diag["reason"] = "propagation ambiguous"
        return False, diag
    diag["propagation_ok"] = True

    if tracks.shape[0] > 0:
        full_tracks = np.concatenate([prev.exit_positions[None, :, :], tracks], axis=0)
    else:
        full_tracks = prev.exit_positions[None, :, :]
    max_d = _max_pairwise_distance(full_tracks)
    diag["max_dist_during_gap"] = max_d

    if max_d > delta_space_px:
        diag["reason"] = "pair drifted apart during gap"
        return False, diag

    final_positions = tracks[-1] if tracks.shape[0] > 0 else prev.exit_positions
    if not _positions_match(final_positions, curr.entry_positions, match_tol_px):
        diag["pair_match"] = False
        diag["reason"] = "different pair at entry of next interval"
        return False, diag
    diag["pair_match"] = True
    diag["reason"] = "merge"
    return True, diag


# ---------------------------------------------------------------------------
# Merging pass
# ---------------------------------------------------------------------------

def _merge_two(prev: Interval, curr: Interval) -> Interval:
    merged_meta = dict(prev.metadata)
    merged_meta.update(curr.metadata)   # last-write-wins
    return Interval(
        start_frame=prev.start_frame,
        end_frame=curr.end_frame,
        entry_frame=prev.entry_frame,
        exit_frame=curr.exit_frame,
        entry_positions=prev.entry_positions,
        exit_positions=curr.exit_positions,
        children=prev.children + curr.children,
        metadata=merged_meta,
    )


def merge_intervals(intervals: list[Interval],
                    store: RoiStore,
                    delta_time: int,
                    delta_space_px: float,
                    match_tol_px: float,
                    max_step_px: Optional[float] = None,
                    metadata_predicate: Optional[Callable[[Interval, Interval], bool]] = None,
                    collect_diagnostics: bool = False,
                    show_progress: bool = True
                    ) -> tuple[list[Interval], list[dict]]:
    if not intervals:
        return [], []

    merged: list[Interval] = [intervals[0]]
    diagnostics: list[dict] = []

    it = tqdm(intervals[1:], desc="Merging intervals",
              disable=not show_progress, unit="iv")
    for curr in it:
        prev = merged[-1]
        decision, diag = should_merge(
            prev, curr, store,
            delta_time=delta_time,
            delta_space_px=delta_space_px,
            match_tol_px=match_tol_px,
            max_step_px=max_step_px,
            metadata_predicate=metadata_predicate,
        )
        if collect_diagnostics:
            diag["prev_end"] = prev.end_frame
            diag["curr_start"] = curr.start_frame
            diagnostics.append(diag)
        if decision:
            merged[-1] = _merge_two(prev, curr)
        else:
            merged.append(curr)
    return merged, diagnostics


# ---------------------------------------------------------------------------
# Opt-in preprocessing
# ---------------------------------------------------------------------------

def drop_short_intervals(intervals: list[Interval], min_duration: int) -> list[Interval]:
    """Opt-in filter. Default pipeline does NOT call this with min_duration > 1."""
    return [iv for iv in intervals if iv.duration >= min_duration]


# ---------------------------------------------------------------------------
# Biological-scale helpers
# ---------------------------------------------------------------------------

DROSOPHILA_MAX_SPEED_MM_S = 40.0  # walking ceiling with margin; not flight


def compute_max_step_px(px_per_mm: float,
                        fps: float,
                        max_speed_mm_s: float = DROSOPHILA_MAX_SPEED_MM_S,
                        safety_factor: float = 2.0) -> float:
    return max_speed_mm_s * px_per_mm / fps * safety_factor



# ---------------------------------------------------------------------------
# Pre-processing: collapse overlapping / abutting raw intervals
# ---------------------------------------------------------------------------

def _collapse_overlapping(intervals: list[Interval]) -> list[Interval]:
    """
    Merge intervals that overlap or are contiguous in frame space.

    This handles upstream CSV artefacts where two rows cover overlapping
    frame ranges (e.g. interval A ends at frame 831273 while interval B
    starts at frame 831270).  Such pairs are collapsed into a single
    Interval whose span covers both, with children lists concatenated and
    metadata merged (last-writer-wins, same convention as _merge_two).

    Assumes the input list is already sorted by start_frame (guaranteed by
    load_intervals_from_csv).  Positions are left at their zero defaults;
    annotate_candidate_pairs will fill them in afterwards.
    """
    if not intervals:
        return []

    collapsed: list[Interval] = [intervals[0]]
    for curr in intervals[1:]:
        prev = collapsed[-1]
        # Overlap  : curr.start_frame <= prev.end_frame
        # Abutting : curr.start_frame == prev.end_frame + 1  (optional — remove
        #            the "- 1" below to treat abutting as separate)
        if curr.start_frame <= prev.end_frame + 1:
            # Extend prev to cover curr
            if curr.end_frame > prev.end_frame:
                merged_meta = dict(prev.metadata)
                merged_meta.update(curr.metadata)
                collapsed[-1] = Interval(
                    start_frame=prev.start_frame,
                    end_frame=curr.end_frame,
                    entry_frame=prev.entry_frame,
                    exit_frame=curr.exit_frame,
                    entry_positions=prev.entry_positions,   # zeros; fixed later
                    exit_positions=curr.exit_positions,     # zeros; fixed later
                    children=prev.children + curr.children,
                    metadata=merged_meta,
                )
            # else: curr is fully contained within prev — just absorb children
            else:
                merged_meta = dict(prev.metadata)
                merged_meta.update(curr.metadata)
                collapsed[-1] = Interval(
                    start_frame=prev.start_frame,
                    end_frame=prev.end_frame,
                    entry_frame=prev.entry_frame,
                    exit_frame=prev.exit_frame,
                    entry_positions=prev.entry_positions,
                    exit_positions=prev.exit_positions,
                    children=prev.children + curr.children,
                    metadata=merged_meta,
                )
        else:
            collapsed.append(curr)
    return collapsed

# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_pipeline(csv_path: str,
                 db_path: str,
                 fps: float,
                 px_per_mm: float,
                 delta_time_s: float = 1.0,
                 delta_space_mm: float = 3.0,
                 match_tol_mm: float = 1.0,
                 max_step_px: Optional[float] = None,
                 min_duration: int = 1,
                 metadata_cols: Optional[list[str]] = None,
                 metadata_predicate: Optional[Callable[[Interval, Interval], bool]] = None,
                 show_progress: bool = True,
                 cache_size: int = 4096,
                 store: "RoiStore | None" = None,
                 ) -> tuple[list[Interval], list[dict]]:
    """
    End-to-end: load intervals from CSV, annotate pairs from ROI_0, merge stutters.

    Parameters
    ----------
    csv_path : str
        Upstream CSV with at least `scene_start` and `length` columns.
    db_path : str
        SQLite DB containing ROI_0.
    fps : float
        Recording frame rate (Hz).
    px_per_mm : float
        Imaging scale.
    delta_time_s : float
        Max gap (seconds) to bridge when merging.

        This is the maximum gap, in seconds of real time, that the algorithm will try to bridge when merging two intervals.
        If two intervals are separated by more than delta_time_s seconds of clean tracking in between,
        they're treated as separate events no matter how close the flies were.
        The reason this is in seconds rather than frames is that you record at different frame rates (150 Hz and 47 Hz).
        At 150 Hz, 1 second is 150 frames; at 47 Hz, 1 second is 47 frames. By specifying the threshold in seconds, 
        the same biological criterion ("up to 1 s of brief separation still counts as the same courtship bout")
        translates correctly across recordings without you having to remember which fps you used.
        Internally the function converts to frames via delta_time_frames = round(delta_time_s * fps).

    delta_space_mm : float
        Max pairwise separation (mm) allowed across the gap.
        While bridging a gap, the algorithm forward-tracks the candidate pair through every frame in the gap and computes the maximum distance between the two flies during that span.
        If that maximum exceeds delta_space_mm millimeters at any point, the algorithm refuses to merge — interpreting the excursion as evidence the flies actually separated and re-engaged independently.
        This is in millimeters rather than pixels because pixel scale changes with imaging hardware, but a fly doesn't change.
        A 2.5-mm threshold is roughly 1 body length (Drosophila body length ≈ 2.5 mm), which means "they stayed within touching distance throughout the gap."
        Internally it's converted via delta_space_px = delta_space_mm * px_per_mm.
        
    match_tol_mm : float
        Position-match tolerance (mm) at gap boundaries.
        After forward-tracking the pair from interval i's exit through the gap,
        the algorithm asks: "do the propagated positions at the end of the gap match the positions of interval i+1's candidate pair,
        treated as a set?" If the two propagated points are within match_tol_mm of the two newly-detected interval-entry points, this is the same physical pair. Otherwise, something different is happening — perhaps a different pair of flies converged, or an identity got lost in the propagation.
        This guards against a specific failure mode: imagine four flies in the arena, fly 1 and fly 2 collide producing interval i,
        then later fly 3 and fly 4 collide producing interval i+1, with a short gap between them.
        Without this check, "short gap" alone might falsely merge them. The position-match check confirms that the same physical pair carried through.
    
        match_tol_mm = 1.0 (about half a body length) is a reasonable default — it's tight enough to reject different-pair scenarios, but loose enough to absorb small drift in the nearest-neighbor propagation and minor centroid jitter. If you find legitimate merges getting rejected because positions just barely miss the tolerance, raise it; if you find different-pair events getting merged, lower it.
        
        The distinction between delta_space_mm and match_tol_mm is worth being clear on, because they sound similar:

            delta_space_mm measures how far apart the two flies in the same pair are during the gap (intra-pair distance).
            match_tol_mm measures how far the propagated pair has drifted from the newly-detected pair at the end of the gap (cross-method registration error).


    max_step_px : float or None
        Per-frame displacement cap for propagation; auto-derived if None.
    min_duration : int
        Drop intervals shorter than this before merging. Default 1 keeps all.
    metadata_cols : list[str] or None
        CSV columns to preserve as Interval.metadata. None = all extras.
    metadata_predicate : callable or None
        Optional hook (prev, curr) -> bool to veto merges based on metadata.
    show_progress : bool
        Show tqdm progress bars.
    cache_size : int
        Frames to retain in the per-frame LRU cache.
    """
    delta_time_frames = int(round(delta_time_s * fps))
    delta_space_px = delta_space_mm * px_per_mm
    match_tol_px = match_tol_mm * px_per_mm
    if max_step_px is None:
        max_step_px = compute_max_step_px(px_per_mm=px_per_mm, fps=fps)

    intervals = load_intervals_from_csv(csv_path, metadata_cols=metadata_cols)
    intervals = _collapse_overlapping(intervals)
    if min_duration > 1:
        intervals = drop_short_intervals(intervals, min_duration=min_duration)

    intervals = annotate_candidate_pairs(
        intervals, store,
        show_progress=show_progress
    )
    merged, diagnostics = merge_intervals(
        intervals, store,
        delta_time=delta_time_frames,
        delta_space_px=delta_space_px,
        match_tol_px=match_tol_px,
        max_step_px=max_step_px,
        metadata_predicate=metadata_predicate,
        collect_diagnostics=True,
        show_progress=show_progress,
    )
    return merged, diagnostics


# ---------------------------------------------------------------------------
# Summary DataFrame
# ---------------------------------------------------------------------------

def intervals_to_dataframe(intervals: list[Interval]) -> pd.DataFrame:
    rows = []
    for iv in intervals:
        row = {
            "start_frame": iv.start_frame,
            "end_frame": iv.end_frame,
            "entry_frame": iv.entry_frame,
            "exit_frame": iv.exit_frame,
            "duration": iv.duration,
            "n_children": len(iv.children),
        }
        # Flatten metadata into the row (prefixed to avoid collisions)
        for k, v in iv.metadata.items():
            row[f"meta_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    merged, diag = run_pipeline(
        csv_path="intervals.csv",
        db_path="foo.db",
        fps=150.0,             # or 47.0
        px_per_mm=30.0,        # replace with your calibration
        delta_time_s=1.0,
        delta_space_mm=3.0,
        match_tol_mm=1.0,
        min_duration=1,
    )
    df = intervals_to_dataframe(merged)
    print(f"Merged down to {len(merged)} episodes "
          f"(from {len(diag) + len(merged)} candidate intervals).")
    print(df.describe())