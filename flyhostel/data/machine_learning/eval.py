import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .dataset import BehaviorSequenceDataset


def evaluate_transformer(
    model_dict: dict,
    X_test: np.ndarray,
    seq_len: int = 10,
) -> dict:
    """
    Evaluate transformer on test data.
    
    Parameters
    ----------
    model_dict : output from train_transformer
    X_test     : (n_frames, n_features)
    seq_len    : same as training
    
    Returns
    -------
    {
        'mse': mean squared error,
        'rmse': root MSE,
        'mae': mean absolute error,
        'predictions': (n_test_samples, n_features),
    }
    """
    model = model_dict['model']
    scaler = model_dict['scaler']
    device = model_dict['device']
    
    X_norm = scaler.transform(X_test)
    dataset = BehaviorSequenceDataset(X_norm, seq_len=seq_len, stride=1)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for x_seq, y_true in loader:
            x_seq = x_seq.to(device)
            y_pred = model(x_seq)
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_true.numpy())
    
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # Denormalize
    predictions_denorm = scaler.inverse_transform(predictions)
    targets_denorm = scaler.inverse_transform(targets)
    
    mse = np.mean((predictions_denorm - targets_denorm) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions_denorm - targets_denorm))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions_denorm,
        'targets': targets_denorm,
    }



def evaluate_per_fly(
    model_dict: dict,
    X_by_fly: dict[int, np.ndarray],
) -> dict[int, dict]:
    """
    Evaluate the pooled model on each fly separately (using raw, non-pooled data).
    """
    results_by_fly = {}
    
    for fly_id, X_test in X_by_fly.items():
        eval_result = evaluate_transformer(
            model_dict,
            X_test,
            seq_len=model_dict['seq_len'],
            fps=model_dict['fps'],
            aggregate_to_fps=model_dict['aggregate_to_fps'],
            scales_s=model_dict['scales_s'],
        )
        results_by_fly[fly_id] = eval_result
    
    return results_by_fly