"""
Regime loader for Paper 3.

Loads pre-computed regime indicators (G_obs, G_ssm) from Paper 1's
output CSVs. These regime series are the inputs to all Paper 3
validation analyses.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Tickers to exclude from panel analysis
EXCLUDED_TICKERS = {"IRX"}

# Columns we need from Paper 1 results
REGIME_COLUMNS = ["date", "y", "rH", "G_obs", "G_ssm"]

# Default relative path from paper3 root to paper1 outputs
DEFAULT_EXP_DIR = (
    Path(__file__).parent.parent.parent / "paper1-latent-str" / "outputs" / "exp_v1"
)


def _parse_results_filename(filename: str) -> tuple[Optional[str], Optional[int]]:
    """
    Parse a Paper 1 results filename to extract ticker and horizon.

    Filenames follow the pattern: {TICKER}_H{horizon}_results.csv
    Examples: GSPC_H1_results.csv, AUDUSDX_H5_results.csv

    Returns
    -------
    tuple
        (ticker, horizon) or (None, None) if pattern doesn't match.
    """
    match = re.match(r"^(.+)_H(\d+)_results\.csv$", filename)
    if match:
        return match.group(1), int(match.group(2))
    return None, None


def _ticker_to_display(ticker: str) -> str:
    """
    Convert filename-safe ticker back to display format.

    Paper 1 uses AUDUSDX for AUDUSD=X, CLF for CL=F, etc.
    """
    # Currencies: end with X, original had =X
    # Futures: end with F, original had =F
    # Indices/ETFs: no suffix change
    return ticker


def _load_single_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate a single Paper 1 results CSV."""
    df = pd.read_csv(csv_path)

    if "date" not in df.columns:
        raise ValueError(f"Missing 'date' column in {csv_path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()

    # Validate required columns exist
    missing = [c for c in ["y", "rH", "G_obs", "G_ssm"]
               if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns {missing} in {csv_path}. "
            f"Available: {list(df.columns)}"
        )

    return df


def load_regime_series(exp_dir: Optional[str] = None,
                       ticker: str = "GSPC",
                       horizon: int = 1) -> pd.DataFrame:
    """
    Load regime indicators for a single asset and horizon.

    Parameters
    ----------
    exp_dir : str, optional
        Path to Paper 1 experiment directory.
        Defaults to ../paper1-latent-str/outputs/exp_v1/.
    ticker : str
        Asset ticker as used in filenames (e.g., "GSPC", "AUDUSDX").
    horizon : int
        Forecast horizon (1, 5, or 22).

    Returns
    -------
    pd.DataFrame
        DataFrame with DatetimeIndex and columns:
        y (log RV), rH (returns), G_obs, G_ssm.
    """
    if exp_dir is None:
        exp_dir = DEFAULT_EXP_DIR
    else:
        exp_dir = Path(exp_dir)

    horizon_dir = exp_dir / f"H{horizon}"
    csv_path = horizon_dir / f"{ticker}_H{horizon}_results.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {csv_path}. "
            f"Check ticker name and horizon."
        )

    df = _load_single_csv(csv_path)

    # Return only the columns we need
    result = df[["y", "rH", "G_obs", "G_ssm"]].copy()
    logger.info("Loaded %s H%d: %d observations (%s to %s)",
                ticker, horizon, len(result),
                result.index.min().strftime("%Y-%m-%d"),
                result.index.max().strftime("%Y-%m-%d"))
    return result


def load_regime_panel(exp_dir: Optional[str] = None,
                      horizon: int = 1) -> dict[str, pd.DataFrame]:
    """
    Load regime indicators for all available assets at a given horizon.

    Parameters
    ----------
    exp_dir : str, optional
        Path to Paper 1 experiment directory.
    horizon : int
        Forecast horizon (1, 5, or 22).

    Returns
    -------
    dict
        Dict mapping ticker -> DataFrame with columns:
        y (log RV), rH (returns), G_obs, G_ssm.
    """
    if exp_dir is None:
        exp_dir = DEFAULT_EXP_DIR
    else:
        exp_dir = Path(exp_dir)

    horizon_dir = exp_dir / f"H{horizon}"

    if not horizon_dir.exists():
        raise FileNotFoundError(
            f"Horizon directory not found: {horizon_dir}")

    panel = {}
    csv_files = sorted(horizon_dir.glob(f"*_H{horizon}_results.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No results files found in {horizon_dir} "
            f"matching pattern *_H{horizon}_results.csv"
        )

    for csv_path in csv_files:
        ticker, h = _parse_results_filename(csv_path.name)
        if ticker is None:
            continue

        if ticker in EXCLUDED_TICKERS:
            logger.info("Excluding %s from panel", ticker)
            continue

        try:
            df = _load_single_csv(csv_path)
            panel[ticker] = df[["y", "rH", "G_obs", "G_ssm"]].copy()
        except Exception:
            logger.exception("Failed to load %s", csv_path)

    logger.info("Loaded panel: %d assets for H%d", len(panel), horizon)
    return panel


def get_asset_list(exp_dir: Optional[str] = None,
                   horizon: int = 1) -> list[str]:
    """
    Get sorted list of available tickers for a given horizon.

    Parameters
    ----------
    exp_dir : str, optional
        Path to Paper 1 experiment directory.
    horizon : int
        Forecast horizon.

    Returns
    -------
    list
        Sorted list of ticker strings (excluding IRX).
    """
    if exp_dir is None:
        exp_dir = DEFAULT_EXP_DIR
    else:
        exp_dir = Path(exp_dir)

    horizon_dir = exp_dir / f"H{horizon}"

    if not horizon_dir.exists():
        return []

    tickers = []
    for csv_path in horizon_dir.glob(f"*_H{horizon}_results.csv"):
        ticker, _ = _parse_results_filename(csv_path.name)
        if ticker is not None and ticker not in EXCLUDED_TICKERS:
            tickers.append(ticker)

    return sorted(tickers)


def align_regime_and_external(regime_df: pd.DataFrame,
                              external_df: pd.DataFrame) -> pd.DataFrame:
    """
    Align regime indicators with external stress proxies via inner join on date.

    Parameters
    ----------
    regime_df : pd.DataFrame
        DataFrame with DatetimeIndex, columns include G_obs, G_ssm, y, rH.
    external_df : pd.DataFrame
        DataFrame with DatetimeIndex, columns are stress proxy variables.

    Returns
    -------
    pd.DataFrame
        Inner-joined DataFrame with all columns from both inputs.
        Columns from regime_df are preserved as-is; external columns
        are appended.
    """
    # Ensure both have DatetimeIndex
    if not isinstance(regime_df.index, pd.DatetimeIndex):
        raise TypeError("regime_df must have DatetimeIndex")
    if not isinstance(external_df.index, pd.DatetimeIndex):
        raise TypeError("external_df must have DatetimeIndex")

    # Inner join on date
    aligned = regime_df.join(external_df, how="inner")

    n_regime = len(regime_df)
    n_external = len(external_df)
    n_aligned = len(aligned)

    logger.info(
        "Alignment: regime=%d, external=%d -> aligned=%d observations "
        "(%s to %s)",
        n_regime, n_external, n_aligned,
        aligned.index.min().strftime("%Y-%m-%d") if n_aligned > 0 else "N/A",
        aligned.index.max().strftime("%Y-%m-%d") if n_aligned > 0 else "N/A",
    )

    if n_aligned == 0:
        logger.warning("No overlapping dates between regime and external data")

    return aligned
