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
TRAIN_START = "2008-01-01"
TRAIN_END = "2026-04-23"
VALIDATION_SPLIT = 0.2
MIN_OBSERVATIONS = 252

# --- FNO Parameters (Increased Capacity) ---
MODEL_TYPE = "FNO"
FNO_MODES = 24                       # Increased from 16
FNO_HIDDEN_CHANNELS = 96             # Increased from 64
FNO_N_LAYERS = 6                     # Increased from 4
FNO_DOMAIN_PADDING = 0.1

# --- Training Parameters ---
BATCH_SIZE = 64                      # Increased batch size
LEARNING_RATE = 0.0005               # Slightly lower LR for stability
WEIGHT_DECAY = 1e-5
EPOCHS = 150                         # More epochs
EARLY_STOP_PATIENCE = 15
RANKING_LOSS_WEIGHT = 0.3            # Weight for pairwise ranking loss
RANDOM_SEED = 42

# --- Runtime Fallback ---
MAX_RUNTIME_SECONDS = 18000
FALLBACK_MODEL = "MLP"

# --- Option Pricing Parameters ---
RISK_FREE_RATE = 0.02
TIME_TO_MATURITY = 1.0

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
