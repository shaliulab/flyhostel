import numpy as np
from .multiscale_lstm_pool import train_multiscale_lstm_predictor_pooled
from .eval import evaluate_per_fly

def compare_predictability(
    X_solo_by_fly: dict,
    X_paired_by_fly: dict,
    **kwargs
) -> dict:
    
    # Train one model on ALL solo flies (random CV)
    print("Training on pooled SOLO data...")
    model_solo = train_multiscale_lstm_predictor_pooled(
        X_solo_by_fly,
        seq_len=1000,
        aggregate_to_fps=5.0,
        **kwargs
    )

    # Train one model on ALL paired flies (random CV)
    print("\nTraining on pooled PAIRED data...")
    model_paired = train_multiscale_lstm_predictor_pooled(
        X_paired_by_fly,
        seq_len=1000,
        aggregate_to_fps=5.0,
        **kwargs
    )

    # Evaluate each fly separately
    print("\nEvaluating per-fly predictability...")
    eval_solo_by_fly = evaluate_per_fly(model_solo, X_solo_by_fly)
    eval_paired_by_fly = evaluate_per_fly(model_paired, X_paired_by_fly)

    # Compute stabilization index per fly
    stabilization_by_fly = {}
    for fly_id in X_solo_by_fly.keys():
        mse_solo = eval_solo_by_fly[fly_id]['mse']
        mse_paired = eval_paired_by_fly[fly_id]['mse']
        stab = (mse_solo - mse_paired) / mse_solo
        stabilization_by_fly[fly_id] = stab
        print(f"Fly {fly_id}: MSE_solo={mse_solo:.4f} → MSE_paired={mse_paired:.4f} | Stab={stab:.3f}")

    # Summary
    mean_stab = np.mean(list(stabilization_by_fly.values()))
    print(f"\nMean stabilization index: {mean_stab:.3f}")