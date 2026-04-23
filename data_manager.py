"""
Data loading and preprocessing for Neural Operator engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import StandardScaler
import config

def load_master_data() -> pd.DataFrame:
    print(f"Downloading {config.HF_DATA_FILE} from {config.HF_DATA_REPO}...")
    file_path = hf_hub_download(
        repo_id=config.HF_DATA_REPO,
        filename=config.HF_DATA_FILE,
        repo_type="dataset",
        token=config.HF_TOKEN,
        cache_dir="./hf_cache"
    )
    df = pd.read_parquet(file_path)
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Prepare a wide-format DataFrame of log returns with Date index."""
    available_tickers = [t for t in tickers if t in df_wide.columns]
    df_long = pd.melt(
        df_wide, id_vars=['Date'], value_vars=available_tickers,
        var_name='ticker', value_name='price'
    )
    df_long = df_long.sort_values(['ticker', 'Date'])
    df_long['log_return'] = df_long.groupby('ticker')['price'].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df_long = df_long.dropna(subset=['log_return'])
    return df_long.pivot(index='Date', columns='ticker', values='log_return')[available_tickers].dropna()

def compute_covariance_surface(returns: pd.DataFrame, window: int = 63) -> np.ndarray:
    """
    Compute rolling covariance matrices and return as a 3D tensor.
    Shape: (num_windows, num_assets, num_assets)
    """
    covs = []
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i-window:i]
        cov = window_returns.cov().values
        covs.append(cov)
    return np.stack(covs)

def compute_margrabe_price(S1: float, S2: float, sigma1: float, sigma2: float, rho: float,
                          T: float = 1.0, r: float = 0.02) -> float:
    """
    Compute Margrabe exchange option price with numerical safeguards.
    """
    from scipy.stats import norm
    sigma_sq = sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2
    if sigma_sq <= 1e-12:
        return max(S1 - S2, 0.0)
    sigma = np.sqrt(sigma_sq)
    if sigma < 1e-8:
        return max(S1 - S2, 0.0)
    if S2 <= 0 or S1 <= 0:
        return 0.0
    d1 = (np.log(S1 / S2) + 0.5 * sigma_sq * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S1 * norm.cdf(d1) - S2 * norm.cdf(d2)
    return max(price, 0.0)

def generate_training_data(returns: pd.DataFrame, window: int = 63) -> tuple:
    """
    Generate input (covariance surface) and target (pairwise Margrabe prices) for training.
    Returns:
        - inputs: (num_samples, num_assets, num_assets) covariance matrices
        - targets: (num_samples, num_assets, num_assets) pairwise exchange option prices
    """
    covs = compute_covariance_surface(returns, window)
    tickers = returns.columns.tolist()
    n_assets = len(tickers)
    n_samples = covs.shape[0]

    targets = np.zeros((n_samples, n_assets, n_assets))
    for i in range(n_samples):
        cov = covs[i]
        cov_annual = cov * 252
        vols = np.sqrt(np.maximum(np.diag(cov_annual), 1e-12))
        corr = cov_annual / np.outer(vols, vols + 1e-12)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)
        S = returns.iloc[i+window-1].values
        for j in range(n_assets):
            for k in range(n_assets):
                if j == k:
                    targets[i, j, k] = 0.0
                else:
                    targets[i, j, k] = compute_margrabe_price(
                        S[j], S[k], vols[j], vols[k], corr[j, k],
                        T=config.TIME_TO_MATURITY, r=config.RISK_FREE_RATE
                    )
    return covs, targets
