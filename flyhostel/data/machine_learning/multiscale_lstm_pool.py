import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
import numpy as np
from .dataset import BehaviorSequenceDataset
from .utils_pool import preprocess_pooled
from .lstm import LSTMPredictor

def train_multiscale_lstm_predictor_pooled_no_log(
    X_by_fly: dict[int, np.ndarray],
    fps: float = 47.0,
    seq_len: int = 1000,
    aggregate_to_fps: float = 5.0,
    n_epochs: int = 100,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
    scales_s: list = [1, 5, 30, 300, 3600],
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    random_seed: int = 42,
) -> dict:
    """
    Train on pooled data from all flies, with random CV (samples from all over).
    
    Parameters
    ----------
    X_by_fly : dict of fly_id → (n_frames, n_features)
        All flies' data pooled together for training
    fps : original sampling rate
    seq_len : sequence length in aggregated frames
    aggregate_to_fps : downsample to this fps
    test_fraction : fraction of all data for final test set
    val_fraction : fraction of remaining for validation
    random_seed : for reproducibility
    
    Returns
    -------
    model_dict with:
        - model: trained LSTM
        - scaler: fitted StandardScaler
        - metadata for per-fly evaluation
    """
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # 1. Preprocess: pool all flies, downsample, multiscale features
    X_norm, fly_ids_ds, scaler = preprocess_pooled(
        X_by_fly,
        aggregate_to_fps,
        fps,
        scales_s,
        verbose=verbose,
    )
    
    # 2. Random CV split (not temporal — samples from all over the experiment)
    n_frames = len(X_norm)
    indices = np.arange(n_frames)
    np.random.shuffle(indices)
    
    n_test = int(n_frames * test_fraction)
    n_val = int((n_frames - n_test) * val_fraction)
    
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    
    X_train = X_norm[train_idx]
    X_val = X_norm[val_idx]
    X_test = X_norm[test_idx]
    
    if verbose:
        print(f"Random CV split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    
    # 3. Create datasets
    dataset_train = BehaviorSequenceDataset(X_train, seq_len=seq_len, stride=1)
    dataset_val = BehaviorSequenceDataset(X_val, seq_len=seq_len, stride=1)
    dataset_test = BehaviorSequenceDataset(X_test, seq_len=seq_len, stride=1)
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    # 4. Model
    model = LSTMPredictor(
        n_features=X_norm.shape[1],
        n_hidden=256,
        n_layers=2,
        dropout=0.2,
    ).to(device)
    
    # 5. Training
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
    last_t = time.time()
    
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
        
        train_loss /= len(dataset_train)
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
        
        val_loss /= len(dataset_val)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_state = model.state_dict().copy()
        else:
            no_improve_count += 1
        
        if verbose and (epoch + 1) % 10 == 0:
            now = time.time()
            elapsed = round(now - last_t)
            print(f"Epoch {epoch+1}/{n_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | Time: {elapsed}s")
            last_t = now
        
        if no_improve_count >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    # Test on held-out test set
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x_seq, y_true in test_loader:
            x_seq, y_true = x_seq.to(device), y_true.to(device)
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_true)
            test_loss += loss.item() * len(x_seq)
    
    test_loss /= len(dataset_test)
    
    lookback_sec = seq_len / aggregate_to_fps
    if verbose:
        print(f"\nFinal Test Loss: {test_loss:.4f}")
        print(f"Lookback: {seq_len} steps @ {aggregate_to_fps} fps = {lookback_sec:.1f} seconds = {lookback_sec/60:.1f} minutes")
    
    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'device': device,
        'seq_len': seq_len,
        'aggregate_to_fps': aggregate_to_fps,
        'lookback_sec': lookback_sec,
        'fps': fps,
        'scales_s': scales_s,
        'test_loss': test_loss,
        'X_by_fly': X_by_fly,  # Keep for per-fly evaluation
    }



def train_multiscale_lstm_predictor_pooled_async(
    X_by_fly: dict[int, np.ndarray],
    fps: float = 47.0,
    seq_len: int = 1000,
    aggregate_to_fps: float = 5.0,
    n_epochs: int = 100,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
    scales_s: list = [1, 5, 30, 300, 3600],
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    random_seed: int = 42,
) -> dict:
    """
    Train on pooled data from all flies, with random CV (samples from all over).
    With progress bars for monitoring long training runs.
    """
    import psutil
    import os
    from tqdm import tqdm
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Get memory info at start
    process = psutil.Process(os.getpid())
    
    # 1. Preprocess
    if verbose:
        print("Preprocessing data...")
    X_norm, fly_ids_ds, scaler = preprocess_pooled(
        X_by_fly,
        aggregate_to_fps,
        fps,
        scales_s,
        verbose=verbose,
    )
    
    # 2. Random CV split
    n_frames = len(X_norm)
    indices = np.arange(n_frames)
    np.random.shuffle(indices)
    
    n_test = int(n_frames * test_fraction)
    n_val = int((n_frames - n_test) * val_fraction)
    
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    
    X_train = X_norm[train_idx]
    X_val = X_norm[val_idx]
    X_test = X_norm[test_idx]
    
    if verbose:
        print(f"Random CV split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    
    # 3. Create datasets
    dataset_train = BehaviorSequenceDataset(X_train, seq_len=seq_len, stride=1)
    dataset_val = BehaviorSequenceDataset(X_val, seq_len=seq_len, stride=1)
    dataset_test = BehaviorSequenceDataset(X_test, seq_len=seq_len, stride=1)
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    if verbose:
        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    
    # 4. Model
    model = LSTMPredictor(
        n_features=X_norm.shape[1],
        n_hidden=256,
        n_layers=2,
        dropout=0.2,
    ).to(device)
    
    if verbose:
        print(f"Model on device: {device}")
        mem_usage = process.memory_info().rss / 1024 / 1024 / 1024
        print(f"Memory usage: {mem_usage:.2f} GB")
    
    # 5. Training
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
    last_t = time.time()
    
    # Main training loop with progress bar
    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch", ncols=100)
    
    for epoch in epoch_pbar:
        # Train
        model.train()
        train_loss = 0.0
        
        # Progress bar for batches (optional, can be slow)
        train_pbar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1} train",
            unit="batch",
            leave=False,
            ncols=80,
            disable=not verbose  # Only show if verbose
        )
        
        for x_seq, y_true in train_pbar:
            x_seq, y_true = x_seq.to(device), y_true.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(x_seq)
            
            # Update batch progress bar with current loss
            if verbose:
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(dataset_train)
        history['train_loss'].append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(
                val_loader,
                desc=f"  Epoch {epoch+1} val",
                unit="batch",
                leave=False,
                ncols=80,
                disable=not verbose
            )
            
            for x_seq, y_true in val_pbar:
                x_seq, y_true = x_seq.to(device), y_true.to(device)
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_true)
                val_loss += loss.item() * len(x_seq)
                
                if verbose:
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(dataset_val)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_state = model.state_dict().copy()
        else:
            no_improve_count += 1
        
        # Update main progress bar
        now = time.time()
        elapsed = round(now - last_t)
        
        mem_usage = process.memory_info().rss / 1024 / 1024 / 1024
        
        postfix_dict = {
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best_val': f'{best_val_loss:.4f}',
            'no_improve': no_improve_count,
            'mem': f'{mem_usage:.1f}GB',
            'time': f'{elapsed}s'
        }
        epoch_pbar.set_postfix(postfix_dict)
        last_t = now
        
        # Early stopping check
        if no_improve_count >= patience:
            epoch_pbar.close()
            if verbose:
                print(f"\n✓ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    # Test on held-out test set
    if verbose:
        print("\nEvaluating on test set...")
    model.eval()
    test_loss = 0.0
    
    test_pbar = tqdm(
        test_loader,
        desc="Test evaluation",
        unit="batch",
        ncols=80
    )
    
    with torch.no_grad():
        for x_seq, y_true in test_pbar:
            x_seq, y_true = x_seq.to(device), y_true.to(device)
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_true)
            test_loss += loss.item() * len(x_seq)
            test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    test_loss /= len(dataset_test)
    
    lookback_sec = seq_len / aggregate_to_fps
    
    if verbose:
        print("\n" + "="*80)
        print(f"Training Complete!")
        print(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Final Val Loss:   {best_val_loss:.4f}")
        print(f"  Final Test Loss:  {test_loss:.4f}")
        print(f"  Lookback: {seq_len} steps @ {aggregate_to_fps} fps = {lookback_sec:.1f} sec ({lookback_sec/60:.1f} min)")
        print("="*80 + "\n")
    
    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'device': device,
        'seq_len': seq_len,
        'aggregate_to_fps': aggregate_to_fps,
        'lookback_sec': lookback_sec,
        'fps': fps,
        'scales_s': scales_s,
        'test_loss': test_loss,
        'X_by_fly': X_by_fly,
    }



def train_multiscale_lstm_predictor_pooled(
    X_by_fly: dict[int, np.ndarray],
    fps: float = 47.0,
    seq_len: int = 1000,
    aggregate_to_fps: float = 5.0,
    n_epochs: int = 100,
    batch_size: int = 32,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    verbose: bool = True,
    scales_s: list = [1, 5, 30, 300, 3600],
    test_fraction: float = 0.2,
    val_fraction: float = 0.1,
    random_seed: int = 42,
    log_dir: str = './training_logs',
) -> dict:
    """
    Train on pooled data from all flies, with random CV and real-time loss logging.
    
    Identical to previous version, with CSV logging to track progress.
    """
    import psutil
    import os
    from pathlib import Path
    from tqdm import tqdm
    from datetime import datetime
    
    # Create log directory and files
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    loss_csv = log_path / f"losses_{run_id}.csv"
    loss_csv.write_text("epoch,train_loss,val_loss,best_val_loss,no_improve\n")
    
    metrics_txt = log_path / f"metrics_{run_id}.txt"
    metrics_txt.write_text("")
    
    def log_metrics(msg):
        """Log to both console and file."""
        if verbose:
            print(msg)
        with open(metrics_txt, 'a') as f:
            f.write(msg + "\n")
    
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    process = psutil.Process(os.getpid())
    
    # 1. Preprocess: pool all flies, downsample, multiscale features
    log_metrics("Preprocessing data...")
    X_norm, fly_ids_ds, scaler = preprocess_pooled(
        X_by_fly,
        aggregate_to_fps,
        fps,
        scales_s,
        verbose=verbose,
    )
    
    # 2. Random CV split (not temporal — samples from all over the experiment)
    n_frames = len(X_norm)
    indices = np.arange(n_frames)
    np.random.shuffle(indices)
    
    n_test = int(n_frames * test_fraction)
    n_val = int((n_frames - n_test) * val_fraction)
    
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    
    X_train = X_norm[train_idx]
    X_val = X_norm[val_idx]
    X_test = X_norm[test_idx]
    
    if verbose:
        log_metrics(f"Random CV split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    
    # 3. Create datasets
    dataset_train = BehaviorSequenceDataset(X_train, seq_len=seq_len, stride=1)
    dataset_val = BehaviorSequenceDataset(X_val, seq_len=seq_len, stride=1)
    dataset_test = BehaviorSequenceDataset(X_test, seq_len=seq_len, stride=1)
    
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    if verbose:
        log_metrics(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    
    # 4. Model
    model = LSTMPredictor(
        n_features=X_norm.shape[1],
        n_hidden=256,
        n_layers=2,
        dropout=0.2,
    ).to(device)
    
    mem_usage = process.memory_info().rss / 1024 / 1024 / 1024
    if verbose:
        log_metrics(f"Model on device: {device}")
        log_metrics(f"Memory usage: {mem_usage:.2f} GB")
    
    # 5. Training
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
    last_t = time.time()
    
    # Main training loop with progress bar
    epoch_pbar = tqdm(range(n_epochs), desc="Training", unit="epoch", ncols=100)
    
    for epoch in epoch_pbar:
        # Train
        model.train()
        train_loss = 0.0
        
        train_pbar = tqdm(
            train_loader,
            desc=f"  Epoch {epoch+1} train",
            unit="batch",
            leave=False,
            ncols=80,
            disable=not verbose
        )
        
        for x_seq, y_true in train_pbar:
            x_seq, y_true = x_seq.to(device), y_true.to(device)
            
            optimizer.zero_grad()
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_true)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(x_seq)
            
            if verbose:
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(dataset_train)
        history['train_loss'].append(train_loss)
        
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm(
                val_loader,
                desc=f"  Epoch {epoch+1} val",
                unit="batch",
                leave=False,
                ncols=80,
                disable=not verbose
            )
            
            for x_seq, y_true in val_pbar:
                x_seq, y_true = x_seq.to(device), y_true.to(device)
                y_pred = model(x_seq)
                loss = criterion(y_pred, y_true)
                val_loss += loss.item() * len(x_seq)
                
                if verbose:
                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss /= len(dataset_val)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_state = model.state_dict().copy()
        else:
            no_improve_count += 1
        
        # Log to CSV (every epoch)
        with open(loss_csv, 'a') as f:
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},{best_val_loss:.6f},{no_improve_count}\n")
        
        # Update main progress bar
        now = time.time()
        elapsed = round(now - last_t)
        mem_usage = process.memory_info().rss / 1024 / 1024 / 1024
        
        postfix_dict = {
            'train': f'{train_loss:.4f}',
            'val': f'{val_loss:.4f}',
            'best_val': f'{best_val_loss:.4f}',
            'no_improve': no_improve_count,
            'mem': f'{mem_usage:.1f}GB',
            'time': f'{elapsed}s'
        }
        epoch_pbar.set_postfix(postfix_dict)
        last_t = now
        
        # Early stopping check
        if no_improve_count >= patience:
            epoch_pbar.close()
            if verbose:
                log_metrics(f"\n✓ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    # Test on held-out test set
    if verbose:
        log_metrics("\nEvaluating on test set...")
    
    model.eval()
    test_loss = 0.0
    
    test_pbar = tqdm(
        test_loader,
        desc="Test evaluation",
        unit="batch",
        ncols=80
    )
    
    with torch.no_grad():
        for x_seq, y_true in test_pbar:
            x_seq, y_true = x_seq.to(device), y_true.to(device)
            y_pred = model(x_seq)
            loss = criterion(y_pred, y_true)
            test_loss += loss.item() * len(x_seq)
            test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    test_loss /= len(dataset_test)
    
    lookback_sec = seq_len / aggregate_to_fps
    
    if verbose:
        log_metrics("\n" + "="*80)
        log_metrics("Training Complete!")
        log_metrics(f"  Final Train Loss: {history['train_loss'][-1]:.4f}")
        log_metrics(f"  Final Val Loss:   {best_val_loss:.4f}")
        log_metrics(f"  Final Test Loss:  {test_loss:.4f}")
        log_metrics(f"  Lookback: {seq_len} steps @ {aggregate_to_fps} fps = {lookback_sec:.1f} sec ({lookback_sec/60:.1f} min)")
        log_metrics(f"\n  Loss CSV: {loss_csv}")
        log_metrics(f"  Metrics TXT: {metrics_txt}")
        log_metrics("="*80 + "\n")
    
    return {
        'model': model,
        'scaler': scaler,
        'history': history,
        'device': device,
        'seq_len': seq_len,
        'aggregate_to_fps': aggregate_to_fps,
        'lookback_sec': lookback_sec,
        'fps': fps,
        'scales_s': scales_s,
        'test_loss': test_loss,
        'X_by_fly': X_by_fly,
        'loss_csv': str(loss_csv),
        'metrics_txt': str(metrics_txt),
    }