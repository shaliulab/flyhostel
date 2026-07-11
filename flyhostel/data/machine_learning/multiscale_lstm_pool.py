import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import (
    ConcatDataset,
    DataLoader
)
from .dataset import BehaviorSequenceDataset
from .utils_pool import preprocess_pooled
from .lstm import LSTMPredictor


def block_split(n, seq_len, aggregate_to_fps, block_seconds=300,
                test_fraction=0.2, val_fraction=0.1, random_seed=42):
    rng = np.random.default_rng(random_seed)
    block_len = int(block_seconds * aggregate_to_fps)
    blocks = [(s, min(s + block_len, n)) for s in range(0, n, block_len)]
    blocks = [(s, e) for (s, e) in blocks if (e - s) > seq_len + 1]  # must fit a sequence
    rng.shuffle(blocks)                     # shuffle BLOCKS, not frames
    n_test = int(len(blocks) * test_fraction)
    n_val  = int((len(blocks) - n_test) * val_fraction)
    return blocks[n_test + n_val:], blocks[n_test:n_test + n_val], blocks[:n_test]

def datasets_from_blocks(X_norm, blocks, seq_len, stride=1):
    return ConcatDataset([
        BehaviorSequenceDataset(X_norm[s:e], seq_len=seq_len, stride=stride)
        for (s, e) in blocks
    ])

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
    block_seconds: int = 300,
    n_hidden: int = 256,        # ← promoted
    n_layers: int = 2,          # ← promoted
    dropout: float = 0.4,       # ← promoted
    base_keys: list = None,
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

    arch = {'n_hidden': n_hidden, 'n_layers': n_layers, 'dropout': dropout}

    # near the top, alongside loss_csv / metrics_txt:
    ckpt_path = log_path / f"model_{run_id}.pt"
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
    X_norm, fly_ids_ds, scaler, multiscale_keys = preprocess_pooled(
        X_by_fly,
        aggregate_to_fps,
        fps,
        scales_s,
        base_keys=base_keys,
        verbose=verbose,
    )

    log_metrics(f"X_norm rows={len(X_norm)}  effective_fps={len(X_norm)/ (3*9.4*3600):.3f}")
    
    # 2. Random CV split (not temporal — samples from all over the experiment)  
    train_blk, val_blk, test_blk = block_split(
        len(X_norm), seq_len, aggregate_to_fps, block_seconds=block_seconds,
        test_fraction=test_fraction, val_fraction=val_fraction, random_seed=random_seed,
    )
    log_metrics(f"blocks: train={len(train_blk)} val={len(val_blk)} test={len(test_blk)}")

    dataset_train = datasets_from_blocks(X_norm, train_blk, seq_len)
    dataset_val   = datasets_from_blocks(X_norm, val_blk,   seq_len)
    dataset_test  = datasets_from_blocks(X_norm, test_blk,  seq_len)


    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    
    if verbose:
        log_metrics(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
    
    # 4. Model
    model = LSTMPredictor(
        n_features=X_norm.shape[1],
        n_hidden=n_hidden,
        n_layers=n_layers,
        dropout=dropout,
    ).to(device)
    
    mem_usage = process.memory_info().rss / 1024 / 1024 / 1024
    if verbose:
        log_metrics(f"Model on device: {device}")
        log_metrics(f"Memory usage: {mem_usage:.2f} GB")
    
    # 5. Training
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
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
            # persist best-so-far, so a crash never costs more than the gap since last improvement
            torch.save({
                'model_state_dict': best_model_state,
                'arch': arch,                                  # ← architecture, as passed
                'scaler': scaler,                              # convenient but sklearn-version-fragile
                'scaler_mean': np.asarray(scaler.mean_),       # ← portable, version-proof
                'scaler_scale': np.asarray(scaler.scale_),     # ← portable, version-proof
                'epoch': epoch + 1,
                'val_loss': best_val_loss,
                'seq_len': seq_len,
                'fps': fps,
                'aggregate_to_fps': aggregate_to_fps,
                'scales_s': scales_s,
                'feature_keys': multiscale_keys,
                'n_features_expected': X_norm.shape[1],
            }, ckpt_path)
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
        'n_features_expected': X_norm.shape[1],
        'feature_keys': multiscale_keys,
        'base_keys': base_keys,
        'checkpoint_path': str(ckpt_path),
        'arch': arch,
    }