"""
External stress proxy data fetching and alignment for Paper 3.

Fetches held-out validation variables (VIX, VVIX, SKEW from Yahoo;
TED, term spread, HY OAS, BBB OAS from FRED) with parquet caching.
These variables were NOT used in Paper 1's regime inference, ensuring
strict separation between training and validation data.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# Default date range covering Paper 1 OOS period with buffer
DEFAULT_START = "2017-01-01"
DEFAULT_END = "2025-12-31"

# Yahoo Finance tickers and their column names
YAHOO_SERIES = {
    "^VIX": "VIX",
    "^VVIX": "VVIX",
    "^SKEW": "SKEW",
}

# FRED series IDs and their column names
FRED_SERIES = {
    "TEDRATE": "TED",
    "T10Y3M": "TERM_SPREAD",
    "BAMLH0A0HYM2": "HY_OAS",
    "BAMLC0A4CBBB": "BBB_OAS",
}

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"


def _cache_path(cache_dir: Path, source: str, series: str,
                start: str, end: str) -> Path:
    """Build deterministic cache file path."""
    safe_series = series.replace("^", "").replace("=", "")
    return cache_dir / f"{source}_{safe_series}_{start}_{end}.parquet"


def fetch_yahoo(ticker: str, start: str = DEFAULT_START,
                end: str = DEFAULT_END,
                cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch daily Close data from Yahoo Finance with parquet caching.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker (e.g., "^VIX").
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    cache_dir : str, optional
        Directory for parquet cache. Defaults to outputs/.data_cache/.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex (named 'date') and 'Close' column.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "outputs" / ".data_cache"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = _cache_path(cache_dir, "yahoo", ticker, start, end)

    if cache_file.exists():
        logger.info("Loading cached Yahoo data for %s", ticker)
        df = pd.read_parquet(cache_file)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df

    logger.info("Downloading Yahoo data for %s (%s to %s)", ticker, start, end)
    raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw.empty:
        logger.warning("No data returned for Yahoo ticker %s", ticker)
        return pd.DataFrame(columns=["Close"])

    # yfinance may return MultiIndex columns for single ticker
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()

    df.to_parquet(cache_file)
    logger.info("Cached Yahoo data for %s (%d rows)", ticker, len(df))
    return df


def fetch_fred(series_id: str, start: str = DEFAULT_START,
               end: str = DEFAULT_END,
               cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch daily data from FRED CSV API with parquet caching.

    Uses the public CSV endpoint (no API key required).
    Handles FRED's "." convention for missing values.

    Parameters
    ----------
    series_id : str
        FRED series identifier (e.g., "TEDRATE").
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    cache_dir : str, optional
        Directory for parquet cache. Defaults to outputs/.data_cache/.

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex (named 'date') and series_id column.
    """
    if cache_dir is None:
        cache_dir = Path(__file__).parent.parent / "outputs" / ".data_cache"
    else:
        cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_file = _cache_path(cache_dir, "fred", series_id, start, end)

    if cache_file.exists():
        logger.info("Loading cached FRED data for %s", series_id)
        df = pd.read_parquet(cache_file)
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df

    logger.info("Downloading FRED data for %s (%s to %s)",
                series_id, start, end)

    params = {
        "id": series_id,
        "cosd": start,
        "coed": end,
    }

    response = requests.get(FRED_CSV_URL, params=params, timeout=30)
    response.raise_for_status()

    # Parse CSV, treating "." as NaN (FRED convention for missing data)
    # FRED CSV uses "observation_date" as the date column
    from io import StringIO
    df = pd.read_csv(
        StringIO(response.text),
        na_values=["."],
    )

    # Identify the date column (FRED uses "observation_date" or "DATE")
    date_col = None
    for candidate in ["observation_date", "DATE", "date"]:
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError(
            f"No recognized date column in FRED response. "
            f"Columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col)
    df.index.name = "date"

    # Ensure the value column is named after the series
    value_cols = [c for c in df.columns if c != date_col]
    if len(value_cols) == 1:
        df.columns = [series_id]
    elif series_id not in df.columns:
        df.columns = [series_id]
    df = df.sort_index()

    # Convert to numeric (handles any remaining non-numeric values)
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")

    df.to_parquet(cache_file)
    logger.info("Cached FRED data for %s (%d rows, %d non-null)",
                series_id, len(df), df[series_id].notna().sum())
    return df


def load_stress_proxies(config_path: Optional[str] = None,
                        start: str = DEFAULT_START,
                        end: str = DEFAULT_END,
                        cache_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load all external stress proxy variables, align to business days,
    and forward-fill missing values (max 5 days).

    Parameters
    ----------
    config_path : str, optional
        Path to YAML config. Currently unused (series are hardcoded
        for reproducibility), but reserved for future configurability.
    start : str
        Start date.
    end : str
        End date.
    cache_dir : str, optional
        Cache directory for raw downloads.

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with columns:
        VIX, VVIX, SKEW, TED, TERM_SPREAD, HY_OAS, BBB_OAS
        Index is business-day DatetimeIndex.
    """
    frames = {}

    # Fetch Yahoo series
    for ticker, col_name in YAHOO_SERIES.items():
        try:
            df = fetch_yahoo(ticker, start=start, end=end,
                             cache_dir=cache_dir)
            if not df.empty:
                frames[col_name] = df["Close"]
        except Exception:
            logger.exception("Failed to fetch Yahoo ticker %s", ticker)

    # Fetch FRED series
    for series_id, col_name in FRED_SERIES.items():
        try:
            df = fetch_fred(series_id, start=start, end=end,
                            cache_dir=cache_dir)
            if not df.empty:
                frames[col_name] = df[series_id]
        except Exception:
            logger.exception("Failed to fetch FRED series %s", series_id)

    if not frames:
        logger.error("No external data fetched successfully")
        return pd.DataFrame()

    # Combine all series
    combined = pd.DataFrame(frames)

    # Align to business day frequency
    bdays = pd.bdate_range(start=combined.index.min(),
                           end=combined.index.max(),
                           freq="B")
    combined = combined.reindex(bdays)
    combined.index.name = "date"

    # Forward-fill missing values, max 5 business days
    combined = combined.ffill(limit=5)

    n_total = len(combined)
    for col in combined.columns:
        n_valid = combined[col].notna().sum()
        n_missing = n_total - n_valid
        if n_missing > 0:
            logger.info("%s: %d/%d valid (%d missing after ffill)",
                        col, n_valid, n_total, n_missing)

    # Note on TED spread discontinuation
    if "TED" in combined.columns:
        ted_last_valid = combined["TED"].last_valid_index()
        if ted_last_valid is not None:
            logger.info("TED spread last valid observation: %s "
                        "(discontinued mid-2022)", ted_last_valid)

    return combined


def compute_vrp(vix_series: pd.Series,
                rv_series: pd.Series) -> pd.Series:
    """
    Compute the Variance Risk Premium (VRP).

    VRP = VIX^2 / 252 - RV

    where VIX is in percentage points (annualized) and RV is realized
    variance (daily, in log scale from Paper 1). We convert VIX to
    daily implied variance for comparability.

    Parameters
    ----------
    vix_series : pd.Series
        VIX index values (annualized percentage, e.g., 20 = 20%).
    rv_series : pd.Series
        Realized volatility from Paper 1 (log scale, i.e., log(RV)).

    Returns
    -------
    pd.Series
        VRP series aligned on common dates.
    """
    # Align on common dates
    common_idx = vix_series.dropna().index.intersection(
        rv_series.dropna().index)

    vix_aligned = vix_series.loc[common_idx]
    rv_aligned = rv_series.loc[common_idx]

    # VIX is in annualized % points. Convert to daily variance:
    # daily_implied_var = (VIX/100)^2 / 252
    implied_daily_var = (vix_aligned / 100.0) ** 2 / 252.0

    # RV from Paper 1 is log(realized_variance), so realized_var = exp(y)
    realized_var = np.exp(rv_aligned)

    vrp = implied_daily_var - realized_var
    vrp.name = "VRP"
    return vrp


def compute_rolling_correlation(returns_dict: dict,
                                window: int = 22) -> pd.Series:
    """
    Compute mean pairwise rolling correlation across assets.

    Parameters
    ----------
    returns_dict : dict
        Dict mapping ticker -> pd.Series of returns (from Paper 1 rH column).
    window : int
        Rolling window in business days (default 22 = ~1 month).

    Returns
    -------
    pd.Series
        Mean pairwise rolling correlation, daily frequency.
    """
    if len(returns_dict) < 2:
        logger.warning("Need at least 2 assets for rolling correlation, "
                        "got %d", len(returns_dict))
        return pd.Series(dtype=float, name="MEAN_CORR")

    # Build returns DataFrame
    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.dropna(how="all")

    n_assets = returns_df.shape[1]
    n_pairs = n_assets * (n_assets - 1) / 2

    # Compute rolling correlation matrix and extract mean of off-diagonal
    rolling_corr_mean = pd.Series(
        index=returns_df.index,
        dtype=float,
        name="MEAN_CORR",
    )

    for i in range(window - 1, len(returns_df)):
        window_data = returns_df.iloc[i - window + 1: i + 1]
        # Require at least half the window for each pair
        corr_mat = window_data.corr(min_periods=window // 2)

        # Mean of upper triangle (excluding diagonal)
        mask = np.triu(np.ones(corr_mat.shape, dtype=bool), k=1)
        off_diag = corr_mat.values[mask]
        valid = off_diag[~np.isnan(off_diag)]

        if len(valid) > 0:
            rolling_corr_mean.iloc[i] = valid.mean()

    return rolling_corr_mean
