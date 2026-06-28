import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_pooled(
    X_by_fly: dict[int, np.ndarray],
    aggregate_to_fps: float,
    fps: float,
    scales_s: list,
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
    
    # 2. Downsample
    downsample_factor = int(fps / aggregate_to_fps)
    X_ds = X_pooled[::downsample_factor, :]
    fly_ids_ds = fly_ids[::downsample_factor]
    
    if verbose:
        print(f"Pooled {len(X_by_fly)} flies, {len(X_pooled)} total frames")
        print(f"Downsampling {len(X_pooled)} frames @ {fps} fps → {len(X_ds)} frames @ {aggregate_to_fps} fps")
    
    # 3. Extract multiscale features
    X_multiscale = extract_multiscale_features(X_ds, fps=aggregate_to_fps, scales_s=scales_s)
    
    # 4. Normalize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_multiscale)
    
    if verbose:
        print(f"Extracted {X_norm.shape[1]} multiscale features")
    
    return X_norm, fly_ids_ds, scaler


def extract_multiscale_features(
    X: np.ndarray,
    fps: float = 1.0,
    scales_s: list = [1, 5, 30, 300, 3600]
) -> np.ndarray:
    """
    Create hierarchical features at multiple timescales.
    
    X: (n_frames, n_features)
    Returns: (n_frames, n_features * n_scales * 2)
    
    For each original feature, compute rolling mean and std at each timescale.
    """
    scales_frames = [max(1, int(fps * s)) for s in scales_s]
    
    if len(set(scales_frames)) < len(scales_frames):
        # Remove duplicates caused by capping
        scales_frames = sorted(list(set(scales_frames)))
    
    n_frames, n_feat = X.shape
    n_scales = len(scales_frames)
    
    # Multi-scale features: (mean, std) at each scale
    X_multiscale = np.zeros((n_frames, n_feat * n_scales * 2))
    
    for feat_idx in range(n_feat):
        col_idx = 0
        for scale_idx, window_frames in enumerate(scales_frames):
            series = pd.Series(X[:, feat_idx])
            
            # Rolling mean
            rolling_mean = series.rolling(
                window=window_frames, 
                center=False, 
                min_periods=1
            ).mean()
            X_multiscale[:, feat_idx * n_scales * 2 + col_idx] = rolling_mean.values
            col_idx += 1
            
            # Rolling std (fill NaN with 0 for constant features)
            rolling_std = series.rolling(
                window=window_frames,
                center=False,
                min_periods=1
            ).std()
            rolling_std = rolling_std.fillna(0)
            X_multiscale[:, feat_idx * n_scales * 2 + col_idx] = rolling_std.values
            col_idx += 1
    
    return X_multiscale
