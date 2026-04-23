"""
Configuration for P2-ETF-NEURAL-OPERATOR engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-neural-operator-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Training Data ---
TRAIN_START = "2008-01-01"          # Use full history
TRAIN_END = "2026-04-23"            # YTD
VALIDATION_SPLIT = 0.2              # Portion of data for validation
MIN_OBSERVATIONS = 252              # Minimum data required per ETF

# --- FNO Parameters ---
MODEL_TYPE = "FNO"                  # "FNO", "DeepONet", or "MLP" (fallback)
FNO_MODES = 16                      # Number of Fourier modes
FNO_HIDDEN_CHANNELS = 64            # Hidden channel dimension
FNO_N_LAYERS = 4                    # Number of Fourier layers
FNO_DOMAIN_PADDING = 0.1            # Padding fraction for FNO

# --- Training Parameters ---
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 100                        # Full training epochs
EARLY_STOP_PATIENCE = 10
RANDOM_SEED = 42

# --- Runtime Fallback ---
# If training exceeds MAX_RUNTIME_SECONDS, switch to simpler model
MAX_RUNTIME_SECONDS = 18000         # 5 hours (leave 1 hour buffer)
FALLBACK_MODEL = "MLP"              # "DeepONet" or "MLP"

# --- Option Pricing Parameters ---
RISK_FREE_RATE = 0.02               # Annualized
TIME_TO_MATURITY = 1.0              # 1 year

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
