import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .dataset import BehaviorSequenceDataset
from .utils_pool import preprocess_single

def persistence_baseline(model_dict, X_test, seq_len=None,
                         fps=None, aggregate_to_fps=None, scales_s=None, base_keys=None):
    """MSE of 'predict next = current', in the SAME normalized space as the model.
    This is the floor the LSTM must beat to have learned dynamics."""
    scaler           = model_dict['scaler']
    seq_len          = seq_len          if seq_len          is not None else model_dict['seq_len']
    fps              = fps              if fps              is not None else model_dict['fps']
    aggregate_to_fps = aggregate_to_fps if aggregate_to_fps is not None else model_dict['aggregate_to_fps']
    scales_s         = scales_s         if scales_s         is not None else model_dict['scales_s']
    # provenance guard: whatever names we're about to build features with
    # MUST match what the model was trained on
    trained_keys = model_dict.get('base_keys')
    if trained_keys is not None and base_keys is not None:
        assert list(base_keys) == list(trained_keys), (
            "feature-key drift: this model was trained on\n"
            f"  {trained_keys}\n"
            f"but evaluation is using\n  {base_keys}\n"
            "Retrain, or evaluate with the original feature definition."
        )
    elif base_keys is None:
        base_keys=trained_keys


    X_ms   = preprocess_single(X_test, aggregate_to_fps, fps, scales_s, base_keys=base_keys)   # same chain as training
    X_norm = scaler.transform(X_ms)

    dataset = BehaviorSequenceDataset(X_norm, seq_len=seq_len, stride=1)

    errs = []
    for x_seq, y_true in dataset:
        x_seq  = x_seq.numpy()  if hasattr(x_seq, "numpy")  else np.asarray(x_seq)
        y_true = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true)
        y_pred = x_seq[-1]                       # last frame of the window = "current"
        errs.append(np.mean((y_pred - y_true) ** 2))
    return {'mse': float(np.mean(errs))}



def evaluate_transformer(model_dict, X_test, seq_len=10,
                         fps=None, aggregate_to_fps=None, scales_s=None, base_keys=None):
    model  = model_dict['model']
    scaler = model_dict['scaler']
    device = model_dict['device']
    fps              = fps              if fps              is not None else model_dict['fps']
    aggregate_to_fps = aggregate_to_fps if aggregate_to_fps is not None else model_dict['aggregate_to_fps']
    scales_s         = scales_s         if scales_s         is not None else model_dict['scales_s']
    # provenance guard: whatever names we're about to build features with
    # MUST match what the model was trained on
    trained_keys = model_dict.get('base_keys')
    if trained_keys is not None and base_keys is not None:
        assert list(base_keys) == list(trained_keys), (
            "feature-key drift: this model was trained on\n"
            f"  {trained_keys}\n"
            f"but evaluation is using\n  {base_keys}\n"
            "Retrain, or evaluate with the original feature definition."
        )
    elif base_keys is None:
        base_keys=trained_keys

    # reproduce training preprocessing EXACTLY, then transform with the fitted scaler
    X_ms   = preprocess_single(X_test, aggregate_to_fps, fps, scales_s, base_keys=base_keys)

    assert X_ms.shape[1] == model_dict['n_features_expected'], (
        f"feature count drift: eval built {X_ms.shape[1]}, "
        f"scaler expects {model_dict['n_features_expected']}")
    

    X_norm = scaler.transform(X_ms)

    dataset = BehaviorSequenceDataset(X_norm, seq_len=seq_len, stride=1)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x_seq, y_true in loader:
            y_pred = model(x_seq.to(device))
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_true.numpy())

    predictions = np.vstack(predictions)
    targets     = np.vstack(targets)

    # MSE in NORMALIZED space (see note below — this is the consequential choice)
    mse  = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae  = np.mean(np.abs(predictions - targets))

    return {'mse': mse, 'rmse': rmse, 'mae': mae,
            'predictions': predictions, 'targets': targets}


def persistence_baseline_per_fly(
    model_dict: dict,
    X_by_fly: dict[int, np.ndarray],
    **kwargs
):
    for fly_id, X_test in X_by_fly.items():

        persist = persistence_baseline(
            model_dict, X_test,
            seq_len=model_dict['seq_len'],
            fps=model_dict['fps'],
            aggregate_to_fps=model_dict['aggregate_to_fps'],
            scales_s=model_dict['scales_s'],
            **kwargs
        )
        print(f"{fly_id}: mean=1.000  persistence={persist['mse']}")
             

def evaluate_per_fly(
    model_dict: dict,
    X_by_fly: dict[int, np.ndarray],
    **kwargs
) -> dict[int, dict]:
    """
    Evaluate the pooled model on each fly separately (raw, non-pooled data).
    All preprocessing params — including base_keys — come from model_dict,
    so eval reproduces training's exact feature columns by construction.
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
            **kwargs,
        )
        results_by_fly[fly_id] = eval_result
        print(f"{fly_id}: LSTM mse={eval_result['mse']:.4f}  (mean-predictor baseline = 1.000)")

    return results_by_fly


def _preprocess_for_eval(model_dict, X_test, fps, aggregate_to_fps, scales_s, base_keys):
    """Shared preprocessing: identical chain to training, returns normalized array + column names."""
    fps              = fps              if fps              is not None else model_dict['fps']
    aggregate_to_fps = aggregate_to_fps if aggregate_to_fps is not None else model_dict['aggregate_to_fps']
    scales_s         = scales_s         if scales_s         is not None else model_dict['scales_s']
    base_keys        = base_keys        if base_keys        is not None else model_dict.get('base_keys')

    X_ms = preprocess_single(X_test, aggregate_to_fps, fps, scales_s, base_keys=base_keys)
    assert X_ms.shape[1] == model_dict['n_features_expected'], (
        f"feature count drift: eval built {X_ms.shape[1]}, "
        f"scaler expects {model_dict['n_features_expected']}")
    X_norm = model_dict['scaler'].transform(X_ms)
    X_norm = np.asarray(X_norm)
    colnames = model_dict['feature_keys']
    return X_norm, colnames


def model_per_column_mse(model_dict, X_test, seq_len=None,
                         fps=None, aggregate_to_fps=None, scales_s=None, base_keys=None):
    """Per-column MSE of the LSTM's next-frame prediction, in normalized space."""
    model   = model_dict['model']
    device  = model_dict['device']
    seq_len = seq_len if seq_len is not None else model_dict['seq_len']

    X_norm, colnames = _preprocess_for_eval(model_dict, X_test, fps, aggregate_to_fps, scales_s, base_keys)

    dataset = BehaviorSequenceDataset(X_norm, seq_len=seq_len, stride=1)
    loader  = DataLoader(dataset, batch_size=256, shuffle=False)

    model.eval()
    sq_err_sum = None
    n = 0
    with torch.no_grad():
        for x_seq, y_true in loader:
            y_pred = model(x_seq.to(device)).cpu().numpy()
            y_true = y_true.numpy()
            se = (y_pred - y_true) ** 2                  # (batch, n_features)
            sq_err_sum = se.sum(axis=0) if sq_err_sum is None else sq_err_sum + se.sum(axis=0)
            n += se.shape[0]
    per_col = sq_err_sum / n
    return dict(zip(colnames, per_col))


def persistence_per_column_mse(model_dict, X_test, seq_len=None,
                               fps=None, aggregate_to_fps=None, scales_s=None, base_keys=None):
    """Per-column MSE of 'predict next = current', in the SAME normalized space."""
    seq_len = seq_len if seq_len is not None else model_dict['seq_len']
    X_norm, colnames = _preprocess_for_eval(model_dict, X_test, fps, aggregate_to_fps, scales_s, base_keys)

    # for each window i: predict X_norm[i+seq_len-1] (current), true is X_norm[i+seq_len] (next).
    # vectorized over all windows: current rows are [seq_len-1 : -1], next rows are [seq_len : ].
    current = X_norm[seq_len - 1 : -1]
    nxt     = X_norm[seq_len     :   ]
    se = (current - nxt) ** 2                            # (n_windows, n_features)
    per_col = se.mean(axis=0)
    return dict(zip(colnames, per_col))


def compare_baselines_per_column(model_dict, X_test, **kw):
    """Print mean(=1.0) vs persistence vs model, per column, and flag where the model wins."""
    model_mse = model_per_column_mse(model_dict, X_test, **kw)
    persist_mse = persistence_per_column_mse(model_dict, X_test, **kw)

    print(f"\n{'column':<26} {'mean':>6} {'persist':>9} {'model':>9}  {'verdict':<22}")
    print("-" * 78)
    model_wins = 0
    for col in model_mse:
        m, p = model_mse[col], persist_mse[col]
        if m < p:
            verdict = "model beats persistence"
            model_wins += 1
        elif m < 1.0:
            verdict = "beats mean, not persist"
        else:
            verdict = "worse than mean"
        print(f"{col:<26} {1.000:>6.3f} {p:>9.4f} {m:>9.4f}  {verdict:<22}")

    print("-" * 78)
    print(f"model beats persistence on {model_wins}/{len(model_mse)} columns")
    # averaged scalars, for reference against your existing numbers
    print(f"avg persistence MSE: {np.mean(list(persist_mse.values())):.4f}")
    print(f"avg model MSE:       {np.mean(list(model_mse.values())):.4f}")
    return model_mse, persist_mse