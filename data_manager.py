"""
Data loading and preprocessing for Neural Operator engine.
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
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

def prepare_prices_and_returns(df_wide: pd.DataFrame, tickers: list) -> tuple:
    """
    FIX: Return BOTH price matrix and log-return matrix.
    Margrabe requires actual prices (S1, S2), not log returns.
    Returns:
        - prices_df:  wide DataFrame of raw prices indexed by Date
        - returns_df: wide DataFrame of log returns indexed by Date
    """
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

    prices_df = df_long.pivot(index='Date', columns='ticker', values='price')[available_tickers]
    returns_df = df_long.pivot(index='Date', columns='ticker', values='log_return')[available_tickers]

    # Align: drop any date where either is NaN
    common_idx = prices_df.dropna().index.intersection(returns_df.dropna().index)
    return prices_df.loc[common_idx], returns_df.loc[common_idx]

def prepare_returns_matrix(df_wide: pd.DataFrame, tickers: list) -> pd.DataFrame:
    """Kept for backward compatibility — returns log-return matrix only."""
    _, returns_df = prepare_prices_and_returns(df_wide, tickers)
    return returns_df

def compute_covariance_surface(returns: pd.DataFrame, window: int = 63) -> np.ndarray:
    """
    Compute rolling covariance matrices (annualised) and return as a 3D tensor.
    Shape: (num_windows, num_assets, num_assets)

    FIX: annualise inside here so inputs to FNO are on a human-readable scale
    (annual variances ~0.01–0.15 instead of daily ~1e-4), which improves gradient flow.
    """
    covs = []
    for i in range(window, len(returns) + 1):
        window_returns = returns.iloc[i - window:i]
        cov_annual = window_returns.cov().values * 252   # annualise
        covs.append(cov_annual)
    return np.stack(covs)

def normalize_covariance_inputs(covs: np.ndarray) -> tuple:
    """
    FIX: Normalise the covariance tensor so each element has zero mean and unit
    std across the time dimension.  Returns (normalised_covs, mean, std) so the
    same transform can be applied at inference time.
    """
    mean = covs.mean(axis=0, keepdims=True)          # (1, n, n)
    std  = covs.std(axis=0, keepdims=True) + 1e-8    # (1, n, n)
    return (covs - mean) / std, mean, std

def compute_margrabe_price(S1: float, S2: float, sigma1: float, sigma2: float, rho: float,
                           T: float = 1.0, r: float = 0.02) -> float:
    """
    Margrabe exchange option price: the right to give up S2 and receive S1.
    S1, S2 must be actual asset prices (> 0).
    """
    from scipy.stats import norm
    sigma_sq = sigma1**2 + sigma2**2 - 2 * rho * sigma1 * sigma2
    if sigma_sq <= 1e-12:
        return max(S1 - S2, 0.0)
    sigma = np.sqrt(sigma_sq)
    if S2 <= 0 or S1 <= 0:
        return 0.0
    d1 = (np.log(S1 / S2) + 0.5 * sigma_sq * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    price = S1 * norm.cdf(d1) - S2 * norm.cdf(d2)
    return max(price, 0.0)

def compute_margrabe_prices_from_cov(cov_annual: np.ndarray, current_prices: np.ndarray,
                                     tickers: list) -> np.ndarray:
    """
    Analytical Margrabe prices from a SINGLE (already annualised) covariance matrix
    and actual current prices.
    """
    n_assets = len(tickers)
    vols = np.sqrt(np.maximum(np.diag(cov_annual), 1e-12))
    # Compute correlation from annualised cov
    corr = cov_annual / np.outer(vols, vols + 1e-12)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    prices = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j:
                prices[i, j] = compute_margrabe_price(
                    current_prices[i], current_prices[j],
                    vols[i], vols[j], corr[i, j],
                    T=config.TIME_TO_MATURITY, r=config.RISK_FREE_RATE
                )
    return prices

def generate_training_data(returns: pd.DataFrame, prices: pd.DataFrame,
                            window: int = 63) -> tuple:
    """
    Generate (covariance surface, Margrabe target prices) for supervised training.

    FIX 1: covariance matrices are annualised (computed by compute_covariance_surface).
    FIX 2: Margrabe S1/S2 now use actual ETF prices from `prices`, NOT log returns.
    FIX 3: Returns raw (un-normalised) covs plus norm stats for consistent inference.

    Returns:
        covs_norm   : (N, n_assets, n_assets)  normalised input covariance tensors
        targets     : (N, n_assets, n_assets)  pairwise Margrabe prices
        cov_mean    : (1, n_assets, n_assets)  per-element mean used for normalisation
        cov_std     : (1, n_assets, n_assets)  per-element std  used for normalisation
    """
    covs = compute_covariance_surface(returns, window)   # already annualised
    covs_norm, cov_mean, cov_std = normalize_covariance_inputs(covs)

    tickers    = returns.columns.tolist()
    n_assets   = len(tickers)
    n_samples  = covs.shape[0]

    targets = np.zeros((n_samples, n_assets, n_assets))

    for i in range(n_samples):
        cov_annual = covs[i]                           # annualised cov
        vols = np.sqrt(np.maximum(np.diag(cov_annual), 1e-12))
        corr = cov_annual / np.outer(vols, vols + 1e-12)
        np.fill_diagonal(corr, 1.0)
        corr = np.clip(corr, -1.0, 1.0)

        # FIX: use the actual closing PRICE at the END of the window,
        # not a log return.  The returns index and prices index are aligned.
        price_date = returns.index[i + window - 1]
        S = prices.loc[price_date].values.astype(float)

        for j in range(n_assets):
            for k in range(n_assets):
                if j != k:
                    targets[i, j, k] = compute_margrabe_price(
                        S[j], S[k], vols[j], vols[k], corr[j, k],
                        T=config.TIME_TO_MATURITY, r=config.RISK_FREE_RATE
                    )

    return covs_norm, targets, cov_mean, cov_std
