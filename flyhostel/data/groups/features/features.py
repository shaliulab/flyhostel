from multiprocessing import Pool, cpu_count
from itertools import combinations
import warnings
import numpy as np
from tqdm import tqdm

from .pairwise_features import _aggregate_pairwise, pairwise_features
from .context_features import _aggregate_individual_context, _individual_context_features
from .group_features import group_level_features
from .data_structures import GroupFrame

def extract_group_features(
        frame: GroupFrame,
        contact_radius: float = 5.0,
        local_radius:   float = 20.0,
) -> dict:
    """
    Extract the full social pose feature vector for one frame.
    Returns a flat dict of floats, ready to be stacked into a
    (n_frames, n_features) matrix for HMM or other downstream models.

    Works for any group size n >= 2.
    """
    flies = frame.flies
    n = len(flies)
    assert n >= 2, "Need at least 2 flies for social features."

    features = {}

    # 1. Pairwise features (both directed orientations for each pair)
    all_pw = []
    for i, j in combinations(range(n), 2):
        all_pw.append(pairwise_features(flies[i], flies[j]))
        all_pw.append(pairwise_features(flies[j], flies[i]))
    features.update(_aggregate_pairwise(all_pw))

    # 2. Individual-in-context features (aggregated over all flies)
    per_fly_ctx = []
    for i, fly in enumerate(flies):
        others = [flies[j] for j in range(n) if j != i]
        per_fly_ctx.append(
            _individual_context_features(fly, others,
                                         contact_radius, local_radius)
        )
    features.update(_aggregate_individual_context(per_fly_ctx))

    # 3. Group-level features
    features.update(group_level_features(flies, contact_radius))

    return features


def _extract_frame_features(args) -> list[float] | None:
    """
    Top-level worker (must be module-level for pickling).
    Returns an ordered list of floats, or None on failure.
    args = (frame, keys, contact_radius, local_radius)
    """
    frame, keys, contact_radius, local_radius = args

    if len(frame.flies) < 2:
        return None

    try:
        feat = extract_group_features(frame, contact_radius, local_radius)
        return [feat[k] for k in keys]
    except Exception as e:
        warnings.warn(f"Frame {frame.frame_id}: {e}")
        return None


def _discover_keys(
    frames: list[GroupFrame],
    contact_radius: float,
    local_radius: float,
) -> list[str]:
    """Run on the first valid frame in the main process to discover feature keys."""
    for frame in frames:
        if len(frame.flies) < 2:
            continue
        try:
            return list(extract_group_features(frame, contact_radius, local_radius).keys())
        except Exception:
            continue
    raise ValueError("No valid frames found — cannot determine feature keys.")


def extract_features_timeseries(
    frames: list[GroupFrame],
    contact_radius: float = 5.0,
    local_radius: float = 20.0,
    n_workers: int | None = None,
    chunksize: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """
    Process all frames in parallel and return:
        X    : np.ndarray of shape (n_frames, n_features)  — NaN for failed rows
        keys : list of feature names (column labels)

    Parameters
    ----------
    frames        : list of GroupFrame objects
    contact_radius: passed through to extract_group_features
    local_radius  : passed through to extract_group_features
    n_workers     : parallel workers (default = all CPUs)
    chunksize     : tasks per worker batch
    """
    n_workers = n_workers or cpu_count()

    keys   = _discover_keys(frames, contact_radius, local_radius)
    n_feat = len(keys)

    args = [
        (frame, keys, contact_radius, local_radius)
        for frame in frames
    ]

    print(f"Extracting features from {len(frames)} frames across {n_workers} workers...", flush=True)
    with Pool(processes=n_workers) as pool:
        results = list(
            tqdm(
                pool.imap(_extract_frame_features, args, chunksize=chunksize),
                total=len(frames),
                desc="Extracting features",
            )
        )

    X = np.full((len(frames), n_feat), np.nan)
    for i, row in enumerate(results):
        if row is not None:
            X[i] = row

    return X, keys


# ---------------------------------------------------------------------------
# Convenience: process a full recording
# ---------------------------------------------------------------------------

def extract_features_timeseries_non_parallel(
        frames: list[GroupFrame],
        contact_radius: float = 5.0,
        local_radius:   float = 20.0,
) -> tuple[np.ndarray, list[str]]:
    """
    Process all frames and return:
        X      : np.ndarray of shape (n_frames, n_features)
        keys   : list of feature names (column labels)
    Frames with fewer than 2 detected flies are skipped (row of NaNs).
    """
    rows = []
    keys = None

    for frame in tqdm(frames, desc="Extracting features"):
        if len(frame.flies) < 2:
            rows.append(None)
            continue
        try:
            feat = extract_group_features(frame, contact_radius, local_radius)
        except Exception as e:
            warnings.warn(f"Frame {frame.frame_id}: {e}")
            rows.append(None)
            continue

        if keys is None:
            keys = list(feat.keys())
        rows.append([feat[k] for k in keys])

    if keys is None:
        raise ValueError("No valid frames found.")

    n_feat = len(keys)
    X = np.full((len(rows), n_feat), np.nan)
    for i, row in enumerate(rows):
        if row is not None:
            X[i] = row

    return X, keys
