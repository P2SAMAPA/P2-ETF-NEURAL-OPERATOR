"""
Main training script for Neural Operator engine.
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import config
import data_manager
from neural_operator_model import NeuralOperatorTrainer
import push_results

def run_neural_operator():
    print(f"=== P2-ETF-NEURAL-OPERATOR Run: {config.TODAY} ===")
    df_master = data_manager.load_master_data()
    df_master = df_master[(df_master['Date'] >= config.TRAIN_START) & (df_master['Date'] <= config.TRAIN_END)]

    all_results = {}
    top_picks = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n--- Processing Universe: {universe_name} ---")
        returns = data_manager.prepare_returns_matrix(df_master, tickers)
        if len(returns) < config.MIN_OBSERVATIONS:
            continue

        # Generate full training data (covariance surfaces + Margrabe prices)
        print("  Generating training data...")
        X, y = data_manager.generate_training_data(returns, window=63)
        n_samples, n_assets, _ = X.shape

        # Flatten targets to (n_samples, n_assets * n_assets) for training
        y = y.reshape(n_samples, -1)

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.VALIDATION_SPLIT, random_state=config.RANDOM_SEED)

        trainer = NeuralOperatorTrainer(
            model_type=config.MODEL_TYPE,
            n_assets=n_assets,
            modes=config.FNO_MODES,
            width=config.FNO_HIDDEN_CHANNELS,
            n_layers=config.FNO_N_LAYERS,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            seed=config.RANDOM_SEED
        )

        print(f"  Training {trainer.model_type} on {n_samples} samples...")
        success = trainer.fit(X_train, y_train, X_val, y_val, epochs=config.EPOCHS,
                              batch_size=config.BATCH_SIZE, patience=config.EARLY_STOP_PATIENCE)

        # Predict on the most recent covariance
        latest_cov = X[-1:]  # (1, n_assets, n_assets)
        pred_prices = trainer.predict(latest_cov).flatten().reshape(n_assets, n_assets)

        # Derive ETF scores: average exchange option price when used as asset 1 (higher is better)
        scores = {}
        for i, ticker in enumerate(tickers):
            # Average price when exchanging this ETF for any other
            avg_price = np.mean(pred_prices[i, :])
            scores[ticker] = avg_price

        sorted_tickers = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_picks[universe_name] = [
            {"ticker": t, "score": float(s)} for t, s in sorted_tickers[:3]
        ]
        all_results[universe_name] = scores

    output_payload = {
        "run_date": config.TODAY,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_") and k.isupper() and k != "HF_TOKEN"},
        "daily_trading": {
            "universes": all_results,
            "top_picks": top_picks
        }
    }

    push_results.push_daily_result(output_payload)
    print("\n=== Run Complete ===")

if __name__ == "__main__":
    run_neural_operator()
