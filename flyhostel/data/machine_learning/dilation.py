import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import numpy as np
from sklearn.preprocessing import StandardScaler
from .dataset import BehaviorSequenceDataset
from .utils import aggregate_features_temporal


class DilatedConvNet(nn.Module):
    """
    Uses dilated convolutions to capture multi-scale temporal dependencies.
    
    Dilation = spacing between kernel elements.
    Dilation=1: looks at consecutive frames
    Dilation=2: looks at every 2nd frame
    Dilation=4: looks at every 4th frame
    
    This lets you look far back without high memory.
    """
    def __init__(self, n_features: int, kernel_size: int = 3):
        super().__init__()
        
        # Stack dilated convolutions: dilation = 1, 2, 4, 8, 16
        self.convs = nn.ModuleList([
            nn.Conv1d(n_features, 64, kernel_size=kernel_size, padding='same', dilation=1),
            nn.Conv1d(64, 64, kernel_size=kernel_size, padding='same', dilation=2),
            nn.Conv1d(64, 64, kernel_size=kernel_size, padding='same', dilation=4),
            nn.Conv1d(64, 64, kernel_size=kernel_size, padding='same', dilation=8),
            nn.Conv1d(64, 64, kernel_size=kernel_size, padding='same', dilation=16),
        ])
        self.relu = nn.ReLU()
        self.output = nn.Linear(64, n_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, n_features)
        returns: (batch, n_features) — predicted next frame
        """
        # Conv1d expects (batch, channels, length)
        x = x.transpose(1, 2)  # (batch, n_features, seq_len)
        
        for conv in self.convs:
            x = self.relu(conv(x))
        
        # Take last timestep and project
        x = x[:, :, -1]  # (batch, 64)
        x = self.output(x)  # (batch, n_features)
        
        return x
    

def train_dilated_conv_predictor(
    X: np.ndarray,
    bin_duration_s: float = 0.5,
    fps: float = 25.0,
    seq_len: int = 20,  # 20 bins × 0.5s = 10 seconds lookback
    n_epochs: int = 100,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
) -> dict:
    """
    Dilated conv model for long-timescale behavioral prediction.
    
    With bin_duration=0.5s and seq_len=20, you see 10 seconds of history.
    """
    # 1. Aggregate to 0.5s bins
    X_agg = aggregate_features_temporal(X, bin_duration_s=bin_duration_s, fps=fps)
    
    # 2. Create dataset
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_agg)
    dataset = BehaviorSequenceDataset(X_norm, seq_len=seq_len, stride=1)
    
    n_train = int(0.8 * len(dataset))
    train_indices = np.arange(0, n_train)
    val_indices = np.arange(n_train, len(dataset))
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # 3. Model
    model = DilatedConvNet(n_features=X.shape[1]).to(device)
    
    # 4. Training (similar to before)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = np.inf
    patience = 20
    no_improve_count = 0
    
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        for x_seq, y_true in train_loader:
            x_seq, y_true = x_seq.to(device), y_true.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(x_seq)
        
        train_loss /= len(train_set)
        history['train_loss'].append(train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_seq, y_true in val_loader:
                x_seq, y_true = x_seq.to(device), y_true.to(device)
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_true)
                val_loss += loss.item() * len(x_seq)
        
        val_loss /= len(val_set)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_state = model.state_dict().copy()
        else:
            no_improve_count += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")
        
        if no_improve_count >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            model.load_state_dict(best_model_state)
            break
    
    if verbose:
        print(f"\nBinned {len(X)} frames into {len(X_agg)} bins ({bin_duration_s}s each)")
        print(f"Lookback: {seq_len} bins = {seq_len * bin_duration_s:.1f} seconds")
    
    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'device': device,
        'bin_duration_s': bin_duration_s,
        'seq_len': seq_len,
    }

