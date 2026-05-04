"""
Main training script for Neural Operator engine.
"""

import json
import pandas as pd
import numpy as np

import config
import data_manager
from neural_operator_model import NeuralOperatorTrainer
import push_results

def _time_series_split(X, y, val_frac=0.2):
    """
    FIX: Chronological train/val split — NO shuffling.
    Shuffling time-series data leaks future information into training.
    """
    split = int(len(X) * (1 - val_frac))
    return X[:split], X[split:], y[:split], y[split:]

def run_neural_operator():
    print(f"=== P2-ETF-NEURAL-OPERATOR Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[
        (df_master['Date'] >= config.TRAIN_START) &
        (df_master['Date'] <= config.TRAIN_END)
    ]

    all_results = {}
    top_picks   = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")

        # FIX: get both prices AND returns; prices are needed for Margrabe S1/S2
        prices, returns = data_manager.prepare_prices_and_returns(df_master, tickers)
        available_tickers = returns.columns.tolist()

        if len(returns) < config.MIN_OBSERVATIONS:
            print(f"  Skipping {universe_name}: only {len(returns)} observations.")
            continue

        print("  Generating training data...")
        # FIX: pass prices so Margrabe gets actual dollar values, not log returns
        # generate_training_data now also returns cov normalisation stats
        X, y, cov_mean, cov_std = data_manager.generate_training_data(
            returns, prices, window=63
        )
        n_samples, n_assets, _ = X.shape
        y_flat = y.reshape(n_samples, -1)

        # FIX: chronological split — no shuffle
        X_train, X_val, y_train, y_val = _time_series_split(
            X, y_flat, val_frac=config.VALIDATION_SPLIT
        )
        print(f"  Train: {len(X_train)}, Val: {len(X_val)} samples  (no shuffle)")

        trainer = NeuralOperatorTrainer(
            model_type=config.MODEL_TYPE,
            n_assets=n_assets,
            modes=config.FNO_MODES,
            width=config.FNO_HIDDEN_CHANNELS,
            n_layers=config.FNO_N_LAYERS,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            ranking_weight=config.RANKING_LOSS_WEIGHT,
            seed=config.RANDOM_SEED
        )

        print(f"  Training {trainer.model_type} on {len(X_train)} samples...")
        success = trainer.fit(
            X_train, y_train, X_val, y_val,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            patience=config.EARLY_STOP_PATIENCE
        )

        # --- Inference on the latest observation ---
        latest_cov_norm = X[-1:]   # already normalised by generate_training_data

        if not success:
            print("  FNO timed out — falling back to analytical Margrabe scores.")
            # Reconstruct un-normalised cov for analytical fallback
            latest_cov_raw = latest_cov_norm * cov_std[0] + cov_mean[0]
            latest_prices  = prices.iloc[-1].values.astype(float)
            pred_prices    = data_manager.compute_margrabe_prices_from_cov(
                latest_cov_raw, latest_prices, available_tickers
            )
        else:
            pred_prices = trainer.predict(latest_cov_norm).flatten().reshape(n_assets, n_assets)

        # --- Scoring ---
        # FIX: use the MEAN Margrabe price across ALL counterparts (not just vs SPY).
        # mean_row[i] = average cost to exchange any other ETF FOR ticker i.
        # Higher mean → ticker i tends to dominate all others → better long candidate.
        # We also include the benchmark-relative column for transparency.
        mean_row_scores = pred_prices.mean(axis=1)   # shape (n_assets,)

        if "SPY" in available_tickers:
            benchmark_idx = available_tickers.index("SPY")
        elif "TLT" in available_tickers:
            benchmark_idx = available_tickers.index("TLT")
        else:
            benchmark_idx = 0

        scores = {}
        for i, ticker in enumerate(available_tickers):
            # Blend: 70 % average-over-all-pairs + 30 % benchmark-relative
            avg_score   = float(mean_row_scores[i])
            bench_score = float(pred_prices[benchmark_idx, i])
            scores[ticker] = round(0.7 * avg_score + 0.3 * bench_score, 6)

        score_values = list(scores.values())
        print(
            f"  Score variance: {np.var(score_values):.6f}, "
            f"range: [{np.min(score_values):.4f}, {np.max(score_values):.4f}]"
        )

        sorted_tickers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_picks[universe_name] = [
            {"ticker": t, "score": float(s)} for t, s in sorted_tickers[:3]
        ]
        all_results[universe_name] = scores

    output_payload = {
        "run_date": config.TODAY,
        "config": {
            k: v for k, v in config.__dict__.items()
            if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"
        },
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_neural_operator()
