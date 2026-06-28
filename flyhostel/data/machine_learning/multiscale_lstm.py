import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
from .utils import preprocess
from .dataset import BehaviorSequenceDataset
from .lstm import LSTMPredictor
    
def train_multiscale_lstm_predictor(
    X: np.ndarray,
    fps: float = 47.0,
    seq_len: int = 1000,  # ~40 seconds at 25 fps OR 200 sec at 5fps aggregated
    aggregate_to_fps: float = 5.0,  # downsample to 5 fps
    n_epochs: int = 100,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
    scales_s: list = [1, 5, 30, 300, 3600],
) -> dict:
    """
    Multi-scale LSTM predictor.
    
    Parameters
    ----------
    X                : (n_frames, n_features)
    fps              : original sampling rate
    seq_len          : sequence length in aggregated frames
    aggregate_to_fps : downsample X to this fps (e.g., 5 fps)
    
    With aggregate_to_fps=5 and seq_len=1000, you look back 200 seconds (~3 minutes).
    To go hours, increase seq_len or aggregate more aggressively.
    """
    
    X_multiscale=preprocess(X, aggregate_to_fps, fps, scales_s=scales_s, verbose=verbose)

    # 3. Normalize
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_multiscale)
    # print(np.isnan(X_norm).mean())
    # print(np.isnan(X_norm).all(axis=0))

    # 4. Create sequences
    dataset = BehaviorSequenceDataset(X_norm, seq_len=seq_len, stride=1)
    
    n_train = int(0.8 * len(dataset))
    train_indices = np.arange(0, n_train)
    val_indices = np.arange(n_train, len(dataset))
    
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # 5. Model
    model = LSTMPredictor(
        n_features=X_multiscale.shape[1],
        n_hidden=256,
        n_layers=2,
        dropout=0.2,
    ).to(device)
    
    # 6. Training
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15
    )
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = np.inf
    patience = 25
    no_improve_count = 0
    best_model_state = None
    last_t=time.time()
    
    for epoch in range(n_epochs):
        # Train
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
        
        # Validate
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
            now=time.time()
            print(f"Epoch {epoch+1}/{n_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f}. Time = {round(last_t-now)} s")
            last_t=now

        
        if no_improve_count >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    lookback_sec = seq_len / aggregate_to_fps
    if verbose:
        print(f"\nLookback: {seq_len} steps @ {aggregate_to_fps} fps = {lookback_sec:.1f} seconds = {lookback_sec/60:.1f} minutes")
    
    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'device': device,
        'seq_len': seq_len,
        'aggregate_to_fps': aggregate_to_fps,
        'lookback_sec': lookback_sec,
    }