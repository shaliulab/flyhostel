import numpy as np
import pandas as pd

def aggregate_features_temporal(
    X: np.ndarray,
    bin_duration_s: float = 0.5,
    fps: float = 25.0,
) -> np.ndarray:
    """
    Aggregate features into time bins.
    
    X: (n_frames, n_features)
    Returns: (n_bins, n_features)
    
    E.g., if bin_duration=0.5s at 25fps, each bin averages 12-13 frames.
    """
    frames_per_bin = max(1, int(fps * bin_duration_s))
    n_bins = len(X) // frames_per_bin
    
    X_binned = np.zeros((n_bins, X.shape[1]))
    for i in range(n_bins):
        start = i * frames_per_bin
        end = start + frames_per_bin
        # Average within bin
        X_binned[i] = np.nanmean(X[start:end], axis=0)
    
    return X_binned



def extract_multiscale_features(
    X: np.ndarray,
    fps: float = 47.0,
    scales_s: list = [1, 5, 30, 300, 3600]
) -> np.ndarray:
    """
    Create hierarchical features at multiple timescales.
    
    X: (n_frames, n_features)
    Returns: (n_frames, n_features * n_scales)
    
    For each original feature, compute rolling statistics at 5 timescales.
    """
    scales_frames = [int(fps * s) for s in scales_s]

    print(scales_frames)
    
    n_frames, n_feat = X.shape
    n_scales = len(scales_s)
    
    # Multi-scale features: (mean, std) at each scale
    X_multiscale = np.zeros((n_frames, n_feat * n_scales * 2))
    
    for feat_idx in range(n_feat):
        col_idx = 0
        for scale_idx, window_frames in enumerate(scales_frames):
            if window_frames > 1:
                # Rolling mean
                X_multiscale[:, feat_idx * n_scales * 2 + col_idx] = pd.Series(
                    X[:, feat_idx]
                ).rolling(window=window_frames, center=False).mean().fillna(method='bfill')
                col_idx += 1
                
                # Rolling std
                X_multiscale[:, feat_idx * n_scales * 2 + col_idx] = pd.Series(
                    X[:, feat_idx]
                ).rolling(window=window_frames, center=False).std().fillna(method='bfill')
                col_idx += 1
    
    return X_multiscale



def preprocess(X, aggregate_to_fps, fps, scales_s, verbose=True):
    # 1. Downsample to reduce memory (e.g., 25 fps → 5 fps)
    downsample_factor = int(fps / aggregate_to_fps)
    X_ds = X[::downsample_factor, :]
    
    if verbose:
        # print(np.isnan(X_ds).any(1).mean())
        print(f"Downsampling {len(X)} frames @ {fps} fps → {len(X_ds)} frames @ {aggregate_to_fps} fps")
    
    # 2. Extract multi-scale features
    X_multiscale = extract_multiscale_features(X_ds, fps=aggregate_to_fps, scales_s=scales_s)
    # print(np.where(np.isnan(X_ds).all(0)))
    return X_multiscale
