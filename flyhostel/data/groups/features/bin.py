import numpy as np


def bin_features(
    X: np.ndarray,
    keys: list[str],
    frame_ids,           # the time or frame index corresponding to each row
    fps: float,
    bin_seconds: float = 10.0,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Temporally bin a (n_frames, n_features) feature matrix.

    For each bin:
      - contact / binary features  → fraction of frames where flag == 1
      - velocity / distance features → mean + max (two output columns)
      - all other features          → mean + std (two output columns)

    Returns
    -------
    X_binned  : (n_bins, n_output_features)
    keys_out  : list of output column names
    bin_times : centre time of each bin (in the same units as frame_ids)
    """
    bin_frames = int(fps * bin_seconds)

    # Classify features by type
    CONTACT_KEYWORDS  = {"contact", "frac", "mutual"}
    VELOCITY_KEYWORDS = {"velocity", "speed", "dist"}

    def _ftype(key: str) -> str:
        k = key.lower()
        if any(kw in k for kw in CONTACT_KEYWORDS):
            return "binary"
        if any(kw in k for kw in VELOCITY_KEYWORDS):
            return "velocity"
        return "default"

    ftypes = [_ftype(k) for k in keys]

    # Build output column names
    keys_out = []
    for k, ft in zip(keys, ftypes):
        if ft == "binary":
            keys_out.append(f"{k}_frac")
        elif ft == "velocity":
            keys_out.extend([f"{k}_mean", f"{k}_max"])
        else:
            keys_out.extend([f"{k}_mean", f"{k}_std"])

    n_frames = len(X)
    n_bins   = n_frames // bin_frames
    rows, bin_times = [], []

    for b in range(n_bins):
        start = b * bin_frames
        end   = start + bin_frames
        chunk = X[start:end]                    # (bin_frames, n_features)
        valid = chunk[~np.isnan(chunk).any(axis=1)]  # drop NaN rows

        if len(valid) == 0:
            rows.append(np.full(len(keys_out), np.nan))
        else:
            row = []
            for j, (k, ft) in enumerate(zip(keys, ftypes)):
                col = valid[:, j]
                if ft == "binary":
                    row.append(col.mean())                   # fraction
                elif ft == "velocity":
                    row.extend([col.mean(), col.max()])      # mean + peak
                else:
                    row.extend([col.mean(), col.std()])      # mean + spread
            rows.append(row)

        # bin centre time
        # mid = (start + end) / 2
        # bin start time
        pointer = start
        entry=[]
        for time_axis in frame_ids:
            entry.append(
                time_axis[min(int(pointer), n_frames - 1)]
            )
        bin_times.append(tuple(entry))

    X_binned  = np.array(rows)
    bin_times = np.array(bin_times) # n_bins x len(frame_ids)
    return X_binned, keys_out, bin_times


# --- usage ---

# X, keys came from extract_features_timeseries()
# frame_ids are the time values from your xarray (pose_features.time.values[::step])

# X_binned, keys_binned, bin_times = bin_features(
#     X,
#     keys,
#     frame_ids = times,       # the subsampled time array from build_frames_parallel
#     fps       = fps / int(fps * 5),   # effective fps after 5-second subsampling
#     bin_seconds = 10.0,
# )