"""
Data loading and feature preparation for Deep-LSTR.
"""

import os
import hashlib
import numpy as np
import pandas as pd
import yfinance as yf

# Numerical stability constant
EPS = 1e-12


def flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten yfinance multi-index columns to lowercase strings.
    Handles format changes in yfinance > 0.2.
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Handle (Price, Ticker) format from yfinance > 0.2
        df.columns = [
            str(c[0]).lower() if isinstance(c, tuple) else str(c).lower()
            for c in df.columns
        ]
    else:
        df.columns = [str(c).lower() for c in df.columns]
    # Remove duplicates if any (keep first)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def get_price_series(df: pd.DataFrame) -> pd.Series:
    """
    Extract closing price series from DataFrame.
    Prioritizes adjusted close over regular close.
    """
    if "adj close" in df.columns:
        return df["adj close"]
    if "adj_close" in df.columns:
        return df["adj_close"]
    if "close" in df.columns:
        return df["close"]
    raise KeyError(f"Cannot find close price. Columns: {df.columns.tolist()}")


def garman_klass_var(df_ohlc: pd.DataFrame) -> pd.Series:
    """
    Garman-Klass (1980) variance estimator from OHLC prices.

    The GK estimator is approximately 7.4 times more efficient than
    the close-to-close squared return estimator under GBM.

    Formula:
        GK = 0.5 * [ln(H/L)]^2 - (2*ln(2) - 1) * [ln(C/O)]^2

    Args:
        df_ohlc: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        Series of Garman-Klass variance estimates
    """
    o = df_ohlc["open"].astype(float)
    h = df_ohlc["high"].astype(float)
    l = df_ohlc["low"].astype(float)
    c = df_ohlc["close"].astype(float)

    log_hl = np.log((h + EPS) / (l + EPS))
    log_co = np.log((c + EPS) / (o + EPS))
    gk = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
    return np.maximum(gk, EPS)


def parkinson_var(df_ohlc: pd.DataFrame) -> pd.Series:
    """
    Parkinson (1980) variance estimator from high-low prices.

    More efficient than close-to-close when drift is zero.

    Formula:
        PK = [ln(H/L)]^2 / (4 * ln(2))

    Args:
        df_ohlc: DataFrame with 'high', 'low' columns

    Returns:
        Series of Parkinson variance estimates
    """
    h = df_ohlc["high"].astype(float)
    l = df_ohlc["low"].astype(float)

    log_hl = np.log((h + EPS) / (l + EPS))
    pk = (log_hl ** 2) / (4.0 * np.log(2.0))
    return np.maximum(pk, EPS)


def rogers_satchell_var(df_ohlc: pd.DataFrame) -> pd.Series:
    """
    Rogers-Satchell (1991) variance estimator from OHLC prices.

    Accounts for drift (non-zero mean returns) unlike Parkinson.

    Formula:
        RS = ln(H/C)*ln(H/O) + ln(L/C)*ln(L/O)

    Args:
        df_ohlc: DataFrame with 'open', 'high', 'low', 'close' columns

    Returns:
        Series of Rogers-Satchell variance estimates
    """
    o = df_ohlc["open"].astype(float)
    h = df_ohlc["high"].astype(float)
    l = df_ohlc["low"].astype(float)
    c = df_ohlc["close"].astype(float)

    log_hc = np.log((h + EPS) / (c + EPS))
    log_ho = np.log((h + EPS) / (o + EPS))
    log_lc = np.log((l + EPS) / (c + EPS))
    log_lo = np.log((l + EPS) / (o + EPS))

    rs = log_hc * log_ho + log_lc * log_lo
    return np.maximum(rs, EPS)


# Volatility estimator registry for robustness checks
VOLATILITY_ESTIMATORS = {
    "garman_klass": garman_klass_var,
    "parkinson": parkinson_var,
    "rogers_satchell": rogers_satchell_var,
}


def get_volatility_estimator(name: str):
    """
    Get volatility estimator function by name.

    Args:
        name: Estimator name ('garman_klass', 'parkinson', 'rogers_satchell')

    Returns:
        Volatility estimator function
    """
    if name not in VOLATILITY_ESTIMATORS:
        raise ValueError(f"Unknown estimator: {name}. Available: {list(VOLATILITY_ESTIMATORS.keys())}")
    return VOLATILITY_ESTIMATORS[name]


def _cache_path(ticker: str, start: str, end: str, interval: str, cache_dir: str) -> str:
    """Build a deterministic cache file path for a given download request."""
    safe_ticker = ticker.replace("^", "").replace("=", "").replace("/", "_")
    key = f"{safe_ticker}_{start}_{end}_{interval}"
    return os.path.join(cache_dir, f"{key}.parquet")


def download_asset_data(
    ticker: str,
    start: str = "2015-01-01",
    end: str = "2025-12-31",
    interval: str = "1d",
    cache_dir: str = None,
) -> pd.DataFrame:
    """
    Download OHLC data from Yahoo Finance with optional parquet caching.

    On first call, downloads from yfinance and saves to parquet.
    On subsequent calls with the same parameters, loads from cache.

    Args:
        ticker: Asset ticker symbol
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        interval: Data frequency (default "1d")
        cache_dir: Directory for parquet cache (default: outputs/.data_cache)

    Returns:
        DataFrame with OHLC data, columns lowercased
    """
    if cache_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, "outputs", ".data_cache")

    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(ticker, start, end, interval, cache_dir)

    if os.path.exists(path):
        df = pd.read_parquet(path)
        return df

    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    df = flatten_yf_columns(df)

    df.to_parquet(path, engine="pyarrow")
    return df


def prepare_features(
    df: pd.DataFrame,
    H: int,
    q_obs_smooth_span: int = 10,
    volatility_estimator: str = "garman_klass",
) -> pd.DataFrame:
    """
    Prepare HAR features and target variable from OHLC data.

    Creates:
        - Log returns (r)
        - Variance estimate (var) and log variance (logv)
        - HAR regressors: x_d (daily), x_w (weekly), x_m (monthly)
        - Target: y = log(mean variance over next H days)
        - Observable transition variable: q_obs (exponential smoothed logv)
        - SSM input features: x1_logv, x2_absr, x3_logvol

    Args:
        df: DataFrame with OHLC and volume data
        H: Forecast horizon in days
        q_obs_smooth_span: Span for exponential smoothing of q_obs
        volatility_estimator: Name of variance estimator
            ('garman_klass', 'parkinson', 'rogers_satchell')

    Returns:
        DataFrame with all features computed, NaN rows dropped
    """
    df = df.copy()

    # Ensure required columns exist
    required = ["open", "high", "low", "close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Handle volume (may be missing for some assets)
    if "volume" not in df.columns:
        df["volume"] = np.nan

    # Price series for returns
    price = get_price_series(df).astype(float)
    df["logp"] = np.log(price.replace(0, np.nan))
    df["r"] = df["logp"].diff()

    # Clip close prices for variance stability
    df["close"] = df["close"].clip(lower=0.01)

    # Variance estimate using selected estimator
    var_fn = get_volatility_estimator(volatility_estimator)
    df["gk_var"] = var_fn(df[["open", "high", "low", "close"]])
    df["logv"] = np.log(df["gk_var"] + EPS)

    # HAR regressors
    df["x_d"] = df["logv"]
    df["x_w"] = df["logv"].rolling(5).mean()
    df["x_m"] = df["logv"].rolling(22).mean()

    # Target: forward-looking H-day average log variance
    df[f"gk_fwd_mean_{H}"] = df["gk_var"].shift(-1).rolling(H).mean()
    df["y"] = np.log(df[f"gk_fwd_mean_{H}"] + EPS)

    # H-day forward return (for risk evaluation)
    df[f"r_fwd_sum_{H}"] = df["r"].shift(-1).rolling(H).sum()

    # Observable transition variable (exponential smoothing of logv)
    df["q_obs"] = df["logv"].ewm(span=q_obs_smooth_span, adjust=False).mean()

    # SSM input features
    df["x1_logv"] = df["logv"]
    df["x2_absr"] = np.abs(df["r"])
    vol = df["volume"].replace(0, np.nan)
    df["x3_logvol"] = np.log(vol).ffill().fillna(0.0)

    # Drop NaN rows
    df = df.dropna()

    return df


def bipower_variation_daily(df: pd.DataFrame) -> pd.Series:
    """
    Approximate bipower variation from daily returns.

    True BPV requires high-frequency data. For daily OHLC, we approximate
    using the product of adjacent absolute returns, scaled appropriately.

    BPV_t approx = mu_1^{-2} * |r_t| * |r_{t-1}|
    where mu_1 = sqrt(2/pi) approx 0.7979

    This captures the continuous component of quadratic variation,
    robust to jumps unlike squared returns.

    Args:
        df: DataFrame with 'r' (log return) column

    Returns:
        Series of bipower variation estimates

    References:
        Barndorff-Nielsen, O.E. & Shephard, N. (2004). Power and Bipower
        Variation with Stochastic Volatility and Jumps. Journal of
        Financial Econometrics, 2(1), 1-37.
    """
    mu_1 = np.sqrt(2.0 / np.pi)
    r = df["r"].values
    abs_r = np.abs(r)

    # BPV_t = mu_1^{-2} * |r_t| * |r_{t-1}|
    bpv = np.zeros(len(r))
    bpv[0] = abs_r[0] ** 2  # Fallback for first obs
    bpv[1:] = (1.0 / mu_1 ** 2) * abs_r[1:] * abs_r[:-1]

    return pd.Series(bpv, index=df.index)


def jump_component(rv: np.ndarray, bv: np.ndarray) -> np.ndarray:
    """
    Compute jump component as max(RV - BV, 0).

    Args:
        rv: Realized variance series
        bv: Bipower variation series

    Returns:
        Jump component (truncated at zero)

    References:
        Andersen, T.G., Bollerslev, T. & Diebold, F.X. (2007).
        Roughing It Up. Review of Economics and Statistics, 89(4).
    """
    return np.maximum(rv - bv, 0)


def standardize_features(X: np.ndarray, mu: np.ndarray = None, sd: np.ndarray = None):
    """
    Z-score standardize features.

    Args:
        X: Feature array (T x p)
        mu: Mean vector (computed from X if None)
        sd: Std vector (computed from X if None)

    Returns:
        Tuple of (standardized X, mu, sd)
    """
    if mu is None:
        mu = np.mean(X, axis=0)
    if sd is None:
        sd = np.std(X, axis=0)
        sd[sd == 0] = 1.0  # Avoid division by zero

    X_std = (X - mu) / sd
    return X_std, mu, sd
