import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def downsample(X, downsample_factor):
    # 2. Downsample
    if len(X.shape)==1:
        X_ds = X[::downsample_factor]
    else:
        X_ds=X[::downsample_factor, :]
    return X_ds


def preprocess_single(X, aggregate_to_fps, fps, scales_s, base_keys=None):
    """Downsample + multiscale, for ONE fly's array. Same chain as preprocess_pooled,
    minus the pooling and the scaler fit."""
    downsample_factor = int(fps / aggregate_to_fps)
    X_ds = downsample(X, downsample_factor)
    X_ms = extract_multiscale_features(X_ds, fps=aggregate_to_fps, scales_s=scales_s, base_keys=base_keys)
    return X_ms


def preprocess_pooled(
    X_by_fly: dict[int, np.ndarray],
    aggregate_to_fps: float,
    fps: float,
    scales_s: list,
    base_keys=None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Pool all flies, downsample, and extract multiscale features.
    
    Parameters
    ----------
    X_by_fly : dict of fly_id → (n_frames, n_features)
    
    Returns
    -------
    X_norm : (n_frames_pooled, n_features_multiscale)
    fly_ids : (n_frames_pooled,) — which fly each frame came from
    scaler : fitted StandardScaler
    """
    # 1. Pool all flies
    all_X = []
    all_fly_ids = []
    
    for fly_id, X in X_by_fly.items():
        all_X.append(X)
        all_fly_ids.extend([fly_id] * len(X))
    
    X_pooled = np.vstack(all_X)
    fly_ids = np.array(all_fly_ids)
    downsample_factor = int(fps / aggregate_to_fps)
    X_ds= downsample(X_pooled, downsample_factor)
    fly_ids_ds=downsample(fly_ids, downsample_factor)

    
    if verbose:
        print(f"Pooled {len(X_by_fly)} flies, {len(X_pooled)} total frames")
        print(f"Downsampling {len(X_pooled)} frames @ {fps} fps → {len(X_ds)} frames @ {aggregate_to_fps} fps")
    
    # 3. Extract multiscale features
    X_multiscale = extract_multiscale_features(
        X_ds, fps=aggregate_to_fps, scales_s=scales_s, base_keys=base_keys
    )
    multiscale_keys = list(X_multiscale.columns)        # grab BEFORE coercing
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_multiscale)         # returns ndarray, fine
    

    # 4. Normalize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_multiscale)


    if verbose:
        print(f"Extracted {X_norm.shape[1]} multiscale features")
    
    return X_norm, fly_ids_ds, scaler, multiscale_keys



def extract_multiscale_features(
    X: np.ndarray,
    fps: float = 1.0,
    scales_s: list = [1, 5, 30, 300, 3600],
    base_keys: list = None,
) -> pd.DataFrame:
    """
    Hierarchical features at multiple timescales.

    X         : (n_frames, n_features)
    base_keys : names of the n_features input columns. If None, uses feat_0..feat_{n-1}.
    Returns   : DataFrame (n_frames, n_features * n_scales * 2), columns named
                f"{feature}__mean_{window}" / f"{feature}__std_{window}".
    """
    scales_frames = [max(1, int(fps * s)) for s in scales_s]
    if len(set(scales_frames)) < len(scales_frames):
        scales_frames = sorted(set(scales_frames))   # dedup on capping

    n_frames, n_feat = X.shape
    n_scales = len(scales_frames)

    if base_keys is None:
        base_keys = [f"feat_{i}" for i in range(n_feat)]
    assert len(base_keys) == n_feat, f"{len(base_keys)} keys vs {n_feat} columns"

    X_multiscale = np.zeros((n_frames, n_feat * n_scales * 2))
    names = []

    for feat_idx in range(n_feat):
        col_idx = 0
        for window_frames in scales_frames:
            series = pd.Series(X[:, feat_idx])

            rolling_mean = series.rolling(window=window_frames, center=False, min_periods=1).mean()
            X_multiscale[:, feat_idx * n_scales * 2 + col_idx] = rolling_mean.values
            names.append(f"{base_keys[feat_idx]}__mean_{window_frames}")
            col_idx += 1

            rolling_std = series.rolling(window=window_frames, center=False, min_periods=1).std().fillna(0)
            X_multiscale[:, feat_idx * n_scales * 2 + col_idx] = rolling_std.values
            names.append(f"{base_keys[feat_idx]}__std_{window_frames}")
            col_idx += 1

    return pd.DataFrame(X_multiscale, columns=names)