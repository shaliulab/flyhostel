"""
The one run-length toolkit of the social package.

All intervals are index intervals with the end EXCLUSIVE, so mask[start:end] is exactly
the run (same convention as proboscis_extension.segmentation).
"""
import numpy as np


def runs(mask):
    """Contiguous True runs of a 1D boolean mask.

    Returns
    -------
    (starts, ends) : two int arrays, ends exclusive.

    Example
    -------
    >>> runs(np.array([0, 1, 1, 0, 1], bool))
    (array([1, 4]), array([3, 5]))
    """
    m = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[False], m, [False]]).astype(np.int8))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def bridge(mask, max_gap):
    """Fill False gaps of at most ``max_gap`` frames that lie BETWEEN two True runs.

    Leading and trailing False stretches are never filled.

    >>> bridge(np.array([1, 0, 0, 1, 0, 0, 0, 1], bool), 2).astype(int)
    array([1, 1, 1, 1, 0, 0, 0, 1])
    """
    m = np.asarray(mask, bool).copy()
    if max_gap <= 0:
        return m
    starts, ends = runs(m)
    gaps = starts[1:] - ends[:-1]
    for g0, g1 in zip(ends[:-1][gaps <= max_gap], starts[1:][gaps <= max_gap]):
        m[g0:g1] = True
    return m


def drop_short(mask, min_len):
    """Set True runs shorter than ``min_len`` frames to False."""
    m = np.asarray(mask, bool).copy()
    if min_len <= 1:
        return m
    starts, ends = runs(m)
    for s, e in zip(starts, ends):
        if e - s < min_len:
            m[s:e] = False
    return m


def apply_columns(func, mask, *args):
    """Apply a 1D run operation to every column of a (T, N) mask."""
    out = np.empty_like(np.asarray(mask, bool))
    for j in range(out.shape[1]):
        out[:, j] = func(mask[:, j], *args)
    return out


def ffill_short_gaps(x, max_gap):
    """Forward-fill interior NaN gaps of at most ``max_gap`` rows, along axis 0.

    Unlike ``DataFrame.ffill(limit=...)``, a longer gap is left entirely NaN instead of
    having its first ``max_gap`` rows filled, because a long absence is real missing data.
    Works on any array whose first axis is time; each trailing column is treated separately.
    """
    x = np.array(x, copy=True)
    flat = x.reshape(x.shape[0], -1)
    for j in range(flat.shape[1]):
        starts, ends = runs(np.isnan(flat[:, j]))
        for s, e in zip(starts, ends):
            if s > 0 and e < flat.shape[0] and e - s <= max_gap:
                flat[s:e, j] = flat[s - 1, j]
    return flat.reshape(x.shape)


def state_runs(values):
    """Runs of equal values covering the whole series: (starts, ends), ends exclusive.

    >>> state_runs(np.array([1, 1, 0, 0, 0, 1], bool))
    (array([0, 2, 5]), array([2, 5, 6]))
    """
    v = np.asarray(values)
    change = np.flatnonzero(v[1:] != v[:-1]) + 1
    return np.concatenate([[0], change]), np.concatenate([change, [len(v)]])


def frames(seconds, fps, minimum=0):
    """Convert a duration in seconds to a whole number of frames at ``fps``."""
    return max(minimum, int(round(seconds * fps)))
