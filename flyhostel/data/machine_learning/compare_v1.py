import numpy as np
from .basic import train_transformer
from .dilation import train_dilated_conv_predictor
from .multiscale_lstm import train_multiscale_lstm_predictor
from .eval import evaluate_transformer
from .utils import preprocess

# ============================================================================
# Full Workflow
# ============================================================================

def compare_predictability(
    X_solo: np.ndarray,
    X_paired: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 32,
    verbose: bool = True,
    model_type = "multiscale_ltm",
    fps: float = 47.0,
    **kwargs
) -> dict:
    """
    Train transformers on solo and paired data, compare predictability.
    
    Returns
    -------
    {
        'solo': {'mse': ..., 'rmse': ..., 'mae': ...},
        'paired': {'mse': ..., 'rmse': ..., 'mae': ...},
        'stabilization_index': (mse_solo - mse_paired) / mse_solo,
    }
    """
    if verbose:
        print("Training on SOLO data...")

    if model_type=="standard":
        seq_len=10
        model_solo = train_transformer(
            X_solo,
            seq_len=seq_len, 
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=verbose
        )
    elif model_type=="dilated_conv":
        seq_len=20
        # Train with 10-second lookback at 0.5s resolution
        model_solo = train_dilated_conv_predictor(
            X_solo,
            bin_duration_s=0.5,
            seq_len=seq_len,
            n_epochs=100,
            fps = fps
        )
    elif model_type=="multiscale_ltm":
        seq_len=1800
           # Look back 30 minutes at 1 fps
        model_solo = train_multiscale_lstm_predictor(
            X_solo,
            fps=fps,
            aggregate_to_fps=1.0,      # Downsample to 1 fps
            seq_len=seq_len,              # 1800 frames @ 1 fps = 30 minutes
            n_epochs=150,
            **kwargs
        )
    
    else:
        raise ValueError

    if verbose:
        print("\nTraining on PAIRED data...")
        
    if model_type=="standard":
        seq_len=10
        model_paired = train_transformer(
            X_paired,
            seq_len=seq_len, 
            n_epochs=n_epochs,
            batch_size=batch_size,
            verbose=verbose
        )
    elif model_type=="dilated_conv":
        seq_len=20
        # Train with 10-second lookback at 0.5s resolution
        model_paired = train_dilated_conv_predictor(
            X_paired,
            bin_duration_s=0.5,
            seq_len=seq_len,
            n_epochs=100,
            fps = fps
        )
    elif model_type=="multiscale_ltm":
        seq_len=1800
           # Look back 30 minutes at 1 fps
        model_paired = train_multiscale_lstm_predictor(
            X_paired,
            fps=fps,
            aggregate_to_fps=1.0,      # Downsample to 1 fps
            seq_len=seq_len,              # 1800 frames @ 1 fps = 30 minutes
            n_epochs=150,
            **kwargs
        )
    else:
        raise ValueError
 

    if verbose:
        print("\nEvaluating...")
    
    # Evaluate on held-out test set (temporal CV)
    # For now, use validation set from training
    X_val=X_solo

    X_val_ms=preprocess(X_val, aggregate_to_fps=1, fps=fps, verbose=verbose, **kwargs)

    eval_solo = evaluate_transformer(model_solo, X_val_ms, seq_len=seq_len)
    eval_paired = evaluate_transformer(model_paired, X_val_ms, seq_len=seq_len)
    
    mse_solo = eval_solo['mse']
    mse_paired = eval_paired['mse']
    
    stabilization_index = (mse_solo - mse_paired) / mse_solo
    
    results = {
        'solo': eval_solo,
        'paired': eval_paired,
        'stabilization_index': stabilization_index,
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SOLO       MSE: {mse_solo:.6f}  |  RMSE: {eval_solo['rmse']:.6f}")
        print(f"PAIRED     MSE: {mse_paired:.6f}  |  RMSE: {eval_paired['rmse']:.6f}")
        print(f"Stabilization Index: {stabilization_index:.3f}")
        print(f"{'='*60}")
        if stabilization_index > 0:
            print(f"→ Partner STABILIZES behavior ({stabilization_index*100:.1f}% better)")
        else:
            print(f"→ Partner DESTABILIZES behavior ({abs(stabilization_index)*100:.1f}% worse)")
    
    return results


