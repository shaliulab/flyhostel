import numpy as np
import torch
from torch.utils.data import Dataset

# ============================================================================
# Dataset
# ============================================================================

class BehaviorSequenceDataset(Dataset):
    """
    Sliding-window dataset for sequence prediction.
    
    Given X of shape (n_frames, n_features), creates sequences of length seq_len
    where each sequence predicts the next frame.
    """
    def __init__(self, X: np.ndarray, seq_len: int = 10, stride: int = 1):
        """
        Parameters
        ----------
        X       : (n_frames, n_features)
        seq_len : length of input sequence (default 10 frames = ~0.4 sec at 25 fps)
        stride  : step between sequences (default 1)
        """
        self.X = X.astype(np.float32)
        self.original_length = len(self.X)
        self.seq_len = seq_len
        self.stride = stride
        
        # Remove NaN rows
        valid_idx = ~np.isnan(self.X).any(axis=1)
        self.X = self.X[valid_idx]
        assert valid_idx.any()
        assert valid_idx.sum() > (seq_len+1)
        
        if len(self.X) < seq_len + 1:
            raise ValueError(
                f"Not enough frames ({len(self.X)}) for seq_len={seq_len}. length of data passed = {self.original_length} "
            )
    
    def __len__(self):
        return (len(self.X) - self.seq_len - 1) // self.stride
    
    def __getitem__(self, idx: int):
        start = idx * self.stride
        end = start + self.seq_len
        
        x_seq = torch.from_numpy(self.X[start:end])  # (seq_len, n_features)
        y_true = torch.from_numpy(self.X[end])       # (n_features,)
        
        return x_seq, y_true


        
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
