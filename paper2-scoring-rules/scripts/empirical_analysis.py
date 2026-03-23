#!/usr/bin/env python3
"""
Paper 2: Regime-Conditional Proper Scoring Rules -- Empirical Analysis
======================================================================

Implements Section 4 of the paper outline:
  4.1  Load Paper 1 results (18-21 assets x 3 horizons)
  4.2  Unconditional vs conditional QLIKE rankings
  4.3  Regime-conditional MCS
  4.4  Risk evaluation conditional on regime (FZ0, VaR violations)

Outputs (CSV + LaTeX) in outputs/tables/:
  Table 2: Panel unconditional QLIKE
  Table 3: Panel conditional QLIKE -- low regime
  Table 4: Panel conditional QLIKE -- high regime
  Table 5: Ranking reversal summary
  Table 6: Regime-conditional MCS inclusion rates
  Table 7: Conditional FZ0 loss (VaR/ES by regime)

Usage:
  python scripts/empirical_analysis.py
  python scripts/empirical_analysis.py --conditioning G_ssm   # default
  python scripts/empirical_analysis.py --conditioning G_proxy
  python scripts/empirical_analysis.py --conditioning G_obs
"""

# ---------------------------------------------------------------------------
# Thread pinning -- must come before any numpy/scipy import
# ---------------------------------------------------------------------------
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
import argparse
import warnings
import logging
from pathlib import Path
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Import Paper 1 evaluation functions
# ---------------------------------------------------------------------------
PAPER1_ROOT = str(
    Path(__file__).resolve().parent.parent.parent / "paper1-latent-str"
)
sys.path.insert(0, PAPER1_ROOT)

from src.metrics import qlike, fz0_loss as metrics_fz0_loss
from src.dm_test import dm_test
from src.mcs import bootstrap_mcs
from src.risk import (
    fz_loss_series_dynamic,
    risk_series_var_es_dynamic,
    kupiec_test,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PAPER1_OUTPUTS = Path(PAPER1_ROOT) / "outputs" / "exp_v1"
PAPER2_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PAPER2_ROOT / "outputs" / "tables"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HORIZONS = [1, 5, 22]

MODEL_COLUMNS = ["har", "str_obs", "str_ssm", "garch", "egarch", "msgarch"]
MODEL_LABELS = {
    "har": "HAR",
    "str_obs": "STR-OBS",
    "str_ssm": "STR-SSM",
    "garch": "GARCH-t",
    "egarch": "EGARCH-t",
    "msgarch": "MS-GARCH-t",
}

# Risk model name mapping (used by Paper 1 risk functions)
RISK_MODEL_NAMES = {
    "har": "har",
    "str_obs": "obs",
    "str_ssm": "ssm",
    "garch": "garch",
    "egarch": "egarch",
    "msgarch": "msgarch",
}

# Regime quartile thresholds for splitting high/low
LOW_QUANTILE = 0.25
HIGH_QUANTILE = 0.75

# Asset class mapping for characterisation
ASSET_CLASSES = {
    "Equity Indices": ["GSPC", "NDX", "RUT", "DJI"],
    "Equity Sectors": ["XLF", "XLK", "XLE", "XLU"],
    "Fixed Income": ["IRX", "TNX", "TYX", "IEF", "TLT"],
    "Currencies": ["EURUSDX", "USDJPYX", "GBPUSDX", "AUDUSDX"],
    "Commodities": ["CLF", "GCF", "NGF", "HGF"],
}

# Invert: ticker -> class
TICKER_TO_CLASS = {}
for cls, tickers in ASSET_CLASSES.items():
    for t in tickers:
        TICKER_TO_CLASS[t] = cls

# Minimum observations per regime subsample to compute loss
MIN_OBS_PER_REGIME = 30

# DM test lag schedule by horizon
DM_LAG = {1: 10, 5: 20, 22: 44}

# MCS parameters
MCS_ALPHA = 0.10
MCS_NBOOT = 5000
MCS_BLOCK = 10

# Risk alphas
RISK_ALPHAS = [0.01, 0.05]

# Numerical guard
EPS = 1e-12

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("paper2_empirical")


# ===================================================================
# 1. Data Loading
# ===================================================================

def discover_tickers() -> List[str]:
    """Scan Paper 1 outputs to find all available tickers across horizons."""
    tickers = set()
    for H in HORIZONS:
        h_dir = PAPER1_OUTPUTS / f"H{H}"
        if not h_dir.exists():
            continue
        for f in h_dir.glob("*_results.csv"):
            # Pattern: TICKER_H{H}_results.csv
            name = f.stem  # e.g. GSPC_H1_results
            parts = name.rsplit(f"_H{H}_results", 1)
            if len(parts) == 2:
                tickers.add(parts[0])
    return sorted(tickers)


def load_results(ticker: str, H: int) -> Optional[pd.DataFrame]:
    """Load a single asset-horizon result CSV from Paper 1."""
    path = PAPER1_OUTPUTS / f"H{H}" / f"{ticker}_H{H}_results.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        # Verify required columns exist
        required = ["y"] + MODEL_COLUMNS
        missing = [c for c in required if c not in df.columns]
        if missing:
            log.warning(
                "Ticker %s H=%d missing columns: %s", ticker, H, missing
            )
            return None
        return df
    except Exception as exc:
        log.warning("Failed to load %s H=%d: %s", ticker, H, exc)
        return None


def load_all_results() -> Dict[Tuple[str, int], pd.DataFrame]:
    """Load all available Paper 1 results.

    Returns:
        Dict keyed by (ticker, H) -> DataFrame
    """
    tickers = discover_tickers()
    log.info("Discovered %d tickers: %s", len(tickers), tickers)

    results = {}
    for ticker in tickers:
        for H in HORIZONS:
            df = load_results(ticker, H)
            if df is not None:
                results[(ticker, H)] = df
                log.info(
                    "  Loaded %s H=%d: %d OOS obs", ticker, H, len(df)
                )

    log.info(
        "Total: %d asset-horizon combinations loaded (of %d possible)",
        len(results),
        len(tickers) * len(HORIZONS),
    )
    return results


# ===================================================================
# 2. Regime Proxy Construction
# ===================================================================

def construct_g_proxy(
    df: pd.DataFrame,
    rolling_window: int = 60,
    min_periods: int = 20,
    ewm_span: int = 5,
) -> np.ndarray:
    """Construct ex-post regime proxy from realized volatility.

    G_proxy is based on where realized volatility sits within its recent
    rolling distribution.  Values near 1 indicate the top of the
    distribution (stress); values near 0 indicate calm.

    Args:
        df: Results DataFrame with column 'y' (log variance).
        rolling_window: Window for computing rolling quantiles.
        min_periods: Minimum observations for rolling stats.
        ewm_span: Exponential smoothing span for the proxy.

    Returns:
        Array of G_proxy values in [0, 1], same length as df.
    """
    rv = np.exp(df["y"].values)  # realized variance
    rv_series = pd.Series(rv, index=df.index)

    rv_median = rv_series.rolling(
        rolling_window, min_periods=min_periods
    ).median()
    rv_q75 = rv_series.rolling(
        rolling_window, min_periods=min_periods
    ).quantile(0.75)

    denominator = (rv_q75 - rv_median).values + EPS
    numerator = rv - rv_median.values

    raw = np.clip(numerator / denominator, 0.0, 1.0)
    smoothed = pd.Series(raw).ewm(span=ewm_span).mean().values
    return np.nan_to_num(smoothed, nan=0.5)


def get_conditioning_variable(
    df: pd.DataFrame, conditioning: str
) -> np.ndarray:
    """Return the regime conditioning variable for subsample splitting.

    Args:
        df: Results DataFrame.
        conditioning: One of 'G_ssm', 'G_obs', 'G_proxy'.

    Returns:
        Array of conditioning values in [0, 1].
    """
    if conditioning == "G_proxy":
        return construct_g_proxy(df)
    elif conditioning in df.columns:
        return df[conditioning].values
    else:
        log.warning(
            "Conditioning variable '%s' not found, falling back to G_proxy",
            conditioning,
        )
        return construct_g_proxy(df)


def regime_masks(
    G: np.ndarray,
    low_q: float = LOW_QUANTILE,
    high_q: float = HIGH_QUANTILE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split observations into low-regime and high-regime subsamples.

    Low regime: G <= quantile(G, low_q)
    High regime: G >= quantile(G, high_q)

    Returns:
        (low_mask, high_mask) boolean arrays
    """
    thresh_low = np.nanquantile(G, low_q)
    thresh_high = np.nanquantile(G, high_q)
    low_mask = G <= thresh_low
    high_mask = G >= thresh_high
    return low_mask, high_mask


# ===================================================================
# 3. Loss Computation
# ===================================================================

def compute_qlike_losses(
    df: pd.DataFrame,
    models: List[str] = MODEL_COLUMNS,
) -> pd.DataFrame:
    """Compute element-wise QLIKE loss for each model.

    Returns:
        DataFrame (T x len(models)) of QLIKE losses.
    """
    y = df["y"].values
    loss_dict = {}
    for m in models:
        if m in df.columns:
            loss_dict[m] = qlike(y, df[m].values)
    return pd.DataFrame(loss_dict, index=df.index)


def mean_qlike_table(
    losses: pd.DataFrame,
    mask: Optional[np.ndarray] = None,
) -> pd.Series:
    """Average QLIKE over observations (optionally masked).

    Returns:
        Series indexed by model name.
    """
    if mask is not None:
        sub = losses.loc[mask]
    else:
        sub = losses
    if len(sub) < MIN_OBS_PER_REGIME:
        return pd.Series(np.nan, index=losses.columns)
    return sub.mean()


# ===================================================================
# 4. DM Tests
# ===================================================================

def pairwise_dm_tests(
    losses: pd.DataFrame,
    lag: int,
    mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Run all pairwise DM tests on a loss DataFrame.

    Args:
        losses: (T x M) loss DataFrame.
        lag: Newey-West truncation lag.
        mask: Optional boolean mask for subsample.

    Returns:
        DataFrame with columns [model_a, model_b, dm_stat, dm_pval].
    """
    if mask is not None:
        sub = losses.loc[mask].values
        cols = losses.columns.tolist()
    else:
        sub = losses.values
        cols = losses.columns.tolist()

    rows = []
    for i, j in combinations(range(len(cols)), 2):
        stat, pval = dm_test(sub[:, i], sub[:, j], L=lag)
        rows.append(
            {
                "model_a": cols[i],
                "model_b": cols[j],
                "dm_stat": stat,
                "dm_pval": pval,
            }
        )
    return pd.DataFrame(rows)


# ===================================================================
# 5. MCS
# ===================================================================

def run_mcs(
    losses: pd.DataFrame,
    mask: Optional[np.ndarray] = None,
    alpha: float = MCS_ALPHA,
    n_boot: int = MCS_NBOOT,
    block_size: int = MCS_BLOCK,
) -> Tuple[List[str], Dict[str, float]]:
    """Run Model Confidence Set on loss DataFrame.

    Args:
        losses: (T x M) loss DataFrame.
        mask: Optional subsample mask.
        alpha: MCS significance level.
        n_boot: Bootstrap replications.
        block_size: Block bootstrap block size.

    Returns:
        (surviving_models, pvalue_dict)
    """
    if mask is not None:
        sub = losses.loc[mask].reset_index(drop=True)
    else:
        sub = losses.reset_index(drop=True)

    if len(sub) < MIN_OBS_PER_REGIME:
        # Not enough data -- all models survive by default
        return losses.columns.tolist(), {m: 1.0 for m in losses.columns}

    # Drop any models that are all-NaN in this subsample
    valid_cols = [c for c in sub.columns if sub[c].notna().sum() > MIN_OBS_PER_REGIME]
    if len(valid_cols) < 2:
        return valid_cols, {m: 1.0 for m in valid_cols}

    sub_clean = sub[valid_cols].dropna()
    if len(sub_clean) < MIN_OBS_PER_REGIME:
        return valid_cols, {m: 1.0 for m in valid_cols}

    try:
        survivors, pvals = bootstrap_mcs(
            sub_clean, alpha=alpha, n_boot=n_boot, block_size=block_size
        )
        return survivors, pvals
    except Exception as exc:
        log.warning("MCS failed: %s", exc)
        return valid_cols, {m: 1.0 for m in valid_cols}


# ===================================================================
# 6. Risk Evaluation Conditional on Regime
# ===================================================================

def conditional_fz0(
    df: pd.DataFrame,
    G: np.ndarray,
    H: int,
    alpha_risk: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Compute FZ0 loss by regime for all models.

    Returns:
        Dict of {model: {'unconditional': fz, 'low': fz, 'high': fz}}
    """
    low_mask, high_mask = regime_masks(G)
    result = {}

    for model_col, risk_name in RISK_MODEL_NAMES.items():
        if model_col not in df.columns:
            continue
        try:
            fz_series = fz_loss_series_dynamic(
                df, risk_name, alpha_risk, H, mean_adjust=True, rolling_nu=True
            )
        except Exception:
            continue

        valid = np.isfinite(fz_series)
        fz_unc = float(np.mean(fz_series[valid])) if valid.sum() > 0 else np.nan

        low_valid = valid & low_mask
        fz_low = (
            float(np.mean(fz_series[low_valid]))
            if low_valid.sum() > MIN_OBS_PER_REGIME
            else np.nan
        )

        high_valid = valid & high_mask
        fz_high = (
            float(np.mean(fz_series[high_valid]))
            if high_valid.sum() > MIN_OBS_PER_REGIME
            else np.nan
        )

        result[model_col] = {
            "unconditional": fz_unc,
            "low": fz_low,
            "high": fz_high,
        }

    return result


def conditional_var_violations(
    df: pd.DataFrame,
    G: np.ndarray,
    H: int,
    alpha_risk: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Compute VaR violation rates by regime for all models.

    Returns:
        Dict of {model: {'unconditional': rate, 'low': rate, 'high': rate,
                         'kupiec_p_low': p, 'kupiec_p_high': p}}
    """
    low_mask, high_mask = regime_masks(G)
    result = {}

    for model_col, risk_name in RISK_MODEL_NAMES.items():
        if model_col not in df.columns:
            continue
        try:
            rH, VaR, ES = risk_series_var_es_dynamic(
                df, risk_name, alpha_risk, H, mean_adjust=True, rolling_nu=True
            )
        except Exception:
            continue

        violations = (rH < VaR).astype(float)
        T_total = len(violations)
        rate_unc = float(violations.mean())

        entry = {"unconditional": rate_unc, "T": T_total}

        # Low regime
        if low_mask.sum() > MIN_OBS_PER_REGIME:
            low_viol = violations[low_mask]
            entry["low"] = float(low_viol.mean())
            _, kup_p, _ = kupiec_test(low_viol, len(low_viol), alpha_risk)
            entry["kupiec_p_low"] = kup_p
        else:
            entry["low"] = np.nan
            entry["kupiec_p_low"] = np.nan

        # High regime
        if high_mask.sum() > MIN_OBS_PER_REGIME:
            high_viol = violations[high_mask]
            entry["high"] = float(high_viol.mean())
            _, kup_p, _ = kupiec_test(high_viol, len(high_viol), alpha_risk)
            entry["kupiec_p_high"] = kup_p
        else:
            entry["high"] = np.nan
            entry["kupiec_p_high"] = np.nan

        result[model_col] = entry

    return result


# ===================================================================
# 7. Ranking Reversal Detection
# ===================================================================

def detect_ranking_reversal(
    unc_qlike: pd.Series,
    low_qlike: pd.Series,
    high_qlike: pd.Series,
) -> Dict:
    """Detect whether the unconditional best model differs from
    conditional best in either regime.

    Returns:
        Dict with reversal details.
    """
    # Drop NaN models
    unc_clean = unc_qlike.dropna()
    low_clean = low_qlike.dropna()
    high_clean = high_qlike.dropna()

    if unc_clean.empty:
        return {"reversal": False, "unc_best": None, "low_best": None, "high_best": None}

    unc_best = unc_clean.idxmin()
    low_best = low_clean.idxmin() if not low_clean.empty else None
    high_best = high_clean.idxmin() if not high_clean.empty else None

    reversal_low = low_best is not None and low_best != unc_best
    reversal_high = high_best is not None and high_best != unc_best

    return {
        "reversal": reversal_low or reversal_high,
        "reversal_low": reversal_low,
        "reversal_high": reversal_high,
        "unc_best": unc_best,
        "low_best": low_best,
        "high_best": high_best,
        "unc_rank": unc_clean.rank().to_dict(),
        "low_rank": low_clean.rank().to_dict() if not low_clean.empty else {},
        "high_rank": high_clean.rank().to_dict() if not high_clean.empty else {},
    }


# ===================================================================
# 8. LaTeX Formatting Helpers
# ===================================================================

def _bold_min(series: pd.Series) -> pd.Series:
    """Format: bold the minimum value in a row."""
    if series.isna().all():
        return series.map(lambda x: "--")
    min_val = series.min()
    def fmt(x):
        if pd.isna(x):
            return "--"
        s = f"{x:.4f}"
        if np.isclose(x, min_val, atol=1e-8):
            return f"\\textbf{{{s}}}"
        return s
    return series.map(fmt)


def _star_pval(p: float) -> str:
    """Return significance stars for p-value."""
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    return ""


def df_to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """Convert DataFrame to a publication-quality LaTeX table string."""
    ncols = len(df.columns)
    col_spec = "l" + "r" * ncols
    lines = []
    lines.append(f"\\begin{{table}}[htbp]")
    lines.append(f"\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    # Header
    header = " & ".join([df.index.name or ""] + [str(c) for c in df.columns])
    lines.append(header + " \\\\")
    lines.append("\\midrule")

    # Rows
    for idx, row in df.iterrows():
        vals = [str(idx)] + [str(v) for v in row.values]
        lines.append(" & ".join(vals) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ===================================================================
# 9. Main Analysis Pipeline
# ===================================================================

def run_single_asset_horizon(
    df: pd.DataFrame,
    ticker: str,
    H: int,
    conditioning: str,
) -> Dict:
    """Run the full analysis for one (asset, horizon) pair.

    Returns:
        Dict with all computed metrics.
    """
    # Conditioning variable
    G = get_conditioning_variable(df, conditioning)
    low_mask, high_mask = regime_masks(G)

    # QLIKE losses
    losses = compute_qlike_losses(df)
    available_models = losses.columns.tolist()

    # Unconditional QLIKE
    unc_qlike = mean_qlike_table(losses)

    # Conditional QLIKE
    low_qlike = mean_qlike_table(losses, mask=low_mask)
    high_qlike = mean_qlike_table(losses, mask=high_mask)

    # DM lag
    lag = DM_LAG.get(H, max(20, 2 * H))

    # Unconditional DM tests
    dm_unc = pairwise_dm_tests(losses, lag=lag)

    # Conditional DM tests
    dm_low = pairwise_dm_tests(losses, lag=lag, mask=low_mask)
    dm_high = pairwise_dm_tests(losses, lag=lag, mask=high_mask)

    # Regime-conditional MCS
    mcs_unc_surv, mcs_unc_pvals = run_mcs(losses)
    mcs_low_surv, mcs_low_pvals = run_mcs(losses, mask=low_mask)
    mcs_high_surv, mcs_high_pvals = run_mcs(losses, mask=high_mask)

    # Ranking reversals
    reversal = detect_ranking_reversal(unc_qlike, low_qlike, high_qlike)

    # Risk evaluation (conditional on regime)
    has_risk_cols = "rH" in df.columns
    fz0_results = {}
    var_results = {}
    if has_risk_cols:
        for alpha_risk in RISK_ALPHAS:
            fz0_results[alpha_risk] = conditional_fz0(df, G, H, alpha_risk)
            var_results[alpha_risk] = conditional_var_violations(df, G, H, alpha_risk)

    return {
        "ticker": ticker,
        "H": H,
        "T": len(df),
        "T_low": int(low_mask.sum()),
        "T_high": int(high_mask.sum()),
        "unc_qlike": unc_qlike,
        "low_qlike": low_qlike,
        "high_qlike": high_qlike,
        "dm_unc": dm_unc,
        "dm_low": dm_low,
        "dm_high": dm_high,
        "mcs_unc_surv": mcs_unc_surv,
        "mcs_unc_pvals": mcs_unc_pvals,
        "mcs_low_surv": mcs_low_surv,
        "mcs_low_pvals": mcs_low_pvals,
        "mcs_high_surv": mcs_high_surv,
        "mcs_high_pvals": mcs_high_pvals,
        "reversal": reversal,
        "fz0": fz0_results,
        "var_violations": var_results,
        "available_models": available_models,
    }


# ===================================================================
# 10. Table Generation
# ===================================================================

def build_panel_qlike_table(
    all_results: List[Dict],
    H: int,
    regime: str = "unconditional",
) -> pd.DataFrame:
    """Build a panel QLIKE table for a given horizon and regime.

    Args:
        all_results: List of result dicts from run_single_asset_horizon.
        H: Horizon to filter.
        regime: One of 'unconditional', 'low', 'high'.

    Returns:
        DataFrame with tickers as rows, models as columns.
    """
    key_map = {
        "unconditional": "unc_qlike",
        "low": "low_qlike",
        "high": "high_qlike",
    }
    qlike_key = key_map[regime]

    rows = {}
    for r in all_results:
        if r["H"] != H:
            continue
        rows[r["ticker"]] = r[qlike_key]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).T
    df.index.name = "Asset"
    # Reorder columns to match MODEL_COLUMNS where available
    ordered_cols = [c for c in MODEL_COLUMNS if c in df.columns]
    df = df[ordered_cols]
    # Rename columns to nice labels
    df.columns = [MODEL_LABELS.get(c, c) for c in ordered_cols]
    return df


def build_formatted_qlike_table(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Add panel mean, bold the best, and return formatted DataFrame."""
    if panel.empty:
        return panel

    # Numeric panel -- add mean row
    mean_row = panel.mean()
    median_row = panel.median()
    panel_out = panel.copy()
    panel_out.loc["Panel Mean"] = mean_row
    panel_out.loc["Panel Median"] = median_row

    # Bold minimum per row
    formatted = panel_out.apply(_bold_min, axis=1)
    return formatted


def build_reversal_summary(
    all_results: List[Dict],
) -> pd.DataFrame:
    """Build Table 5: ranking reversal summary.

    For each asset-horizon, report:
      - unconditional best model
      - low-regime best model
      - high-regime best model
      - whether there is a reversal
      - asset class
    """
    rows = []
    for r in all_results:
        rev = r["reversal"]
        rows.append(
            {
                "Asset": r["ticker"],
                "H": r["H"],
                "Asset Class": TICKER_TO_CLASS.get(r["ticker"], "Unknown"),
                "Unc. Best": MODEL_LABELS.get(rev["unc_best"], rev["unc_best"]),
                "Low Best": MODEL_LABELS.get(rev["low_best"], rev["low_best"])
                if rev["low_best"]
                else "--",
                "High Best": MODEL_LABELS.get(rev["high_best"], rev["high_best"])
                if rev["high_best"]
                else "--",
                "Reversal (Low)": rev.get("reversal_low", False),
                "Reversal (High)": rev.get("reversal_high", False),
                "Any Reversal": rev["reversal"],
            }
        )
    df = pd.DataFrame(rows)
    return df


def build_reversal_summary_aggregated(
    reversal_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate reversal counts by horizon and asset class."""
    rows = []

    for H in HORIZONS:
        sub = reversal_df[reversal_df["H"] == H]
        total = len(sub)
        n_rev = sub["Any Reversal"].sum()
        n_rev_low = sub["Reversal (Low)"].sum()
        n_rev_high = sub["Reversal (High)"].sum()
        rows.append(
            {
                "Horizon": f"H={H}",
                "N Assets": total,
                "Reversals": int(n_rev),
                "Reversal Rate": f"{n_rev / total:.1%}" if total > 0 else "--",
                "Low Reversals": int(n_rev_low),
                "High Reversals": int(n_rev_high),
            }
        )

    # By asset class (across all horizons)
    for cls in ASSET_CLASSES:
        sub = reversal_df[reversal_df["Asset Class"] == cls]
        total = len(sub)
        n_rev = sub["Any Reversal"].sum()
        rows.append(
            {
                "Horizon": cls,
                "N Assets": total,
                "Reversals": int(n_rev),
                "Reversal Rate": f"{n_rev / total:.1%}" if total > 0 else "--",
                "Low Reversals": int(sub["Reversal (Low)"].sum()),
                "High Reversals": int(sub["Reversal (High)"].sum()),
            }
        )

    return pd.DataFrame(rows)


def build_mcs_inclusion_table(
    all_results: List[Dict],
) -> pd.DataFrame:
    """Build Table 6: MCS inclusion rates across the panel.

    For each model, compute the fraction of (asset, horizon) combos
    where it survives the MCS, separately for unconditional, low, and high.
    """
    counts = {m: {"unc": 0, "low": 0, "high": 0} for m in MODEL_COLUMNS}
    totals = {"unc": 0, "low": 0, "high": 0}

    for r in all_results:
        for regime_key, surv_key in [
            ("unc", "mcs_unc_surv"),
            ("low", "mcs_low_surv"),
            ("high", "mcs_high_surv"),
        ]:
            survivors = r[surv_key]
            totals[regime_key] += 1
            for m in MODEL_COLUMNS:
                if m in survivors:
                    counts[m][regime_key] += 1

    rows = []
    for m in MODEL_COLUMNS:
        row = {"Model": MODEL_LABELS.get(m, m)}
        for regime_key, label in [
            ("unc", "MCS Unc."),
            ("low", "MCS Low"),
            ("high", "MCS High"),
        ]:
            n = totals[regime_key]
            if n > 0:
                rate = counts[m][regime_key] / n
                row[label] = f"{rate:.2f}"
            else:
                row[label] = "--"
        rows.append(row)

    return pd.DataFrame(rows).set_index("Model")


def build_mcs_inclusion_by_horizon(
    all_results: List[Dict],
) -> pd.DataFrame:
    """MCS inclusion rates broken down by horizon."""
    rows = []
    for H in HORIZONS:
        h_results = [r for r in all_results if r["H"] == H]
        if not h_results:
            continue

        for m in MODEL_COLUMNS:
            unc_count = sum(1 for r in h_results if m in r["mcs_unc_surv"])
            low_count = sum(1 for r in h_results if m in r["mcs_low_surv"])
            high_count = sum(1 for r in h_results if m in r["mcs_high_surv"])
            n = len(h_results)
            rows.append(
                {
                    "H": H,
                    "Model": MODEL_LABELS.get(m, m),
                    "MCS Unc.": f"{unc_count}/{n} ({unc_count/n:.0%})",
                    "MCS Low": f"{low_count}/{n} ({low_count/n:.0%})",
                    "MCS High": f"{high_count}/{n} ({high_count/n:.0%})",
                }
            )

    return pd.DataFrame(rows)


def build_conditional_fz0_table(
    all_results: List[Dict],
    alpha_risk: float = 0.05,
) -> pd.DataFrame:
    """Build Table 7: Conditional FZ0 loss.

    Panel-averaged FZ0 by regime, for each model.
    """
    # Collect per-(asset, H) FZ0 values
    records = []
    for r in all_results:
        fz0 = r.get("fz0", {}).get(alpha_risk, {})
        if not fz0:
            continue
        for model_col, fz_dict in fz0.items():
            records.append(
                {
                    "ticker": r["ticker"],
                    "H": r["H"],
                    "model": MODEL_LABELS.get(model_col, model_col),
                    "FZ0 Unc.": fz_dict["unconditional"],
                    "FZ0 Low": fz_dict["low"],
                    "FZ0 High": fz_dict["high"],
                }
            )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Panel mean by model and horizon
    agg = df.groupby(["H", "model"])[["FZ0 Unc.", "FZ0 Low", "FZ0 High"]].mean()
    return agg.round(4)


def build_conditional_var_table(
    all_results: List[Dict],
    alpha_risk: float = 0.05,
) -> pd.DataFrame:
    """VaR violation rates by regime, panel-averaged."""
    records = []
    for r in all_results:
        var_res = r.get("var_violations", {}).get(alpha_risk, {})
        if not var_res:
            continue
        for model_col, v_dict in var_res.items():
            records.append(
                {
                    "ticker": r["ticker"],
                    "H": r["H"],
                    "model": MODEL_LABELS.get(model_col, model_col),
                    "Viol. Unc.": v_dict["unconditional"],
                    "Viol. Low": v_dict.get("low", np.nan),
                    "Viol. High": v_dict.get("high", np.nan),
                    "Kupiec p Low": v_dict.get("kupiec_p_low", np.nan),
                    "Kupiec p High": v_dict.get("kupiec_p_high", np.nan),
                }
            )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    agg = df.groupby(["H", "model"])[
        ["Viol. Unc.", "Viol. Low", "Viol. High"]
    ].mean()
    return agg.round(4)


# ===================================================================
# 11. Characterisation of Reversals
# ===================================================================

def characterise_reversals(
    reversal_df: pd.DataFrame,
    all_results: List[Dict],
) -> str:
    """Generate a text summary characterising which assets show reversals."""
    lines = []
    lines.append("=" * 70)
    lines.append("RANKING REVERSAL CHARACTERISATION")
    lines.append("=" * 70)

    total = len(reversal_df)
    n_rev = reversal_df["Any Reversal"].sum()
    lines.append(f"\nOverall: {n_rev}/{total} asset-horizon pairs show ranking reversals ({n_rev/total:.1%})")

    # By horizon
    lines.append("\nBy horizon:")
    for H in HORIZONS:
        sub = reversal_df[reversal_df["H"] == H]
        n = len(sub)
        nr = sub["Any Reversal"].sum()
        lines.append(f"  H={H:2d}: {nr}/{n} ({nr/n:.1%})" if n > 0 else f"  H={H:2d}: no data")

    # By asset class
    lines.append("\nBy asset class:")
    for cls in ASSET_CLASSES:
        sub = reversal_df[reversal_df["Asset Class"] == cls]
        n = len(sub)
        if n == 0:
            continue
        nr = sub["Any Reversal"].sum()
        lines.append(f"  {cls:20s}: {nr}/{n} ({nr/n:.1%})")

    # Which models benefit from conditional evaluation?
    lines.append("\nModels that become best in high-regime but not unconditionally:")
    high_only_wins = {}
    for _, row in reversal_df[reversal_df["Reversal (High)"]].iterrows():
        model = row["High Best"]
        high_only_wins[model] = high_only_wins.get(model, 0) + 1
    for model, count in sorted(high_only_wins.items(), key=lambda x: -x[1]):
        lines.append(f"  {model}: {count} cases")

    lines.append("\nModels that become best in low-regime but not unconditionally:")
    low_only_wins = {}
    for _, row in reversal_df[reversal_df["Reversal (Low)"]].iterrows():
        model = row["Low Best"]
        low_only_wins[model] = low_only_wins.get(model, 0) + 1
    for model, count in sorted(low_only_wins.items(), key=lambda x: -x[1]):
        lines.append(f"  {model}: {count} cases")

    # Regime imbalance
    lines.append("\nRegime subsample sizes (mean across panel):")
    for H in HORIZONS:
        h_results = [r for r in all_results if r["H"] == H]
        if not h_results:
            continue
        t_lows = [r["T_low"] for r in h_results]
        t_highs = [r["T_high"] for r in h_results]
        t_totals = [r["T"] for r in h_results]
        lines.append(
            f"  H={H:2d}: T_low={np.mean(t_lows):.0f} ({np.mean(t_lows)/np.mean(t_totals):.1%}), "
            f"T_high={np.mean(t_highs):.0f} ({np.mean(t_highs)/np.mean(t_totals):.1%}), "
            f"T_total={np.mean(t_totals):.0f}"
        )

    return "\n".join(lines)


# ===================================================================
# 12. Save All Tables
# ===================================================================

def save_table(df: pd.DataFrame, name: str, caption: str, label: str) -> None:
    """Save a table as both CSV and LaTeX."""
    csv_path = OUTPUT_DIR / f"{name}.csv"
    tex_path = OUTPUT_DIR / f"{name}.tex"

    df.to_csv(csv_path)
    log.info("  Saved %s", csv_path)

    latex_str = df_to_latex(df, caption, label)
    with open(tex_path, "w") as f:
        f.write(latex_str)
    log.info("  Saved %s", tex_path)


def save_all_tables(
    all_results: List[Dict],
    conditioning: str,
) -> None:
    """Generate and save all output tables."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cond_label = conditioning.replace("_", "\\_")

    # ----------------------------------------------------------------
    # Table 2: Unconditional QLIKE (one sub-table per horizon)
    # ----------------------------------------------------------------
    log.info("Generating Table 2: Unconditional QLIKE panel...")
    for H in HORIZONS:
        panel = build_panel_qlike_table(all_results, H, "unconditional")
        if panel.empty:
            continue
        formatted = build_formatted_qlike_table(panel)
        save_table(
            formatted,
            f"table2_unc_qlike_H{H}",
            f"Unconditional QLIKE Loss, $H={H}$",
            f"tab:unc_qlike_h{H}",
        )

    # ----------------------------------------------------------------
    # Table 3: Low-regime QLIKE
    # ----------------------------------------------------------------
    log.info("Generating Table 3: Low-regime QLIKE panel...")
    for H in HORIZONS:
        panel = build_panel_qlike_table(all_results, H, "low")
        if panel.empty:
            continue
        formatted = build_formatted_qlike_table(panel)
        save_table(
            formatted,
            f"table3_low_qlike_H{H}",
            f"QLIKE Loss -- Low Regime ($G \\leq Q_{{0.25}}$), $H={H}$. Conditioning: ${cond_label}$",
            f"tab:low_qlike_h{H}",
        )

    # ----------------------------------------------------------------
    # Table 4: High-regime QLIKE
    # ----------------------------------------------------------------
    log.info("Generating Table 4: High-regime QLIKE panel...")
    for H in HORIZONS:
        panel = build_panel_qlike_table(all_results, H, "high")
        if panel.empty:
            continue
        formatted = build_formatted_qlike_table(panel)
        save_table(
            formatted,
            f"table4_high_qlike_H{H}",
            f"QLIKE Loss -- High Regime ($G \\geq Q_{{0.75}}$), $H={H}$. Conditioning: ${cond_label}$",
            f"tab:high_qlike_h{H}",
        )

    # ----------------------------------------------------------------
    # Table 5: Ranking reversals
    # ----------------------------------------------------------------
    log.info("Generating Table 5: Ranking reversals...")
    reversal_df = build_reversal_summary(all_results)
    save_table(
        reversal_df.set_index(["Asset", "H"]),
        "table5_reversals_detail",
        "Ranking Reversals: Unconditional vs Conditional Best Model",
        "tab:reversals_detail",
    )

    reversal_agg = build_reversal_summary_aggregated(reversal_df)
    save_table(
        reversal_agg.set_index("Horizon"),
        "table5_reversals_summary",
        "Ranking Reversal Summary by Horizon and Asset Class",
        "tab:reversals_summary",
    )

    # Characterisation text
    char_text = characterise_reversals(reversal_df, all_results)
    char_path = OUTPUT_DIR / "reversal_characterisation.txt"
    with open(char_path, "w") as f:
        f.write(char_text)
    log.info("  Saved %s", char_path)
    print(char_text)

    # ----------------------------------------------------------------
    # Table 6: MCS inclusion rates
    # ----------------------------------------------------------------
    log.info("Generating Table 6: MCS inclusion rates...")
    mcs_table = build_mcs_inclusion_table(all_results)
    save_table(
        mcs_table,
        "table6_mcs_inclusion",
        f"Regime-Conditional MCS Inclusion Rates ($\\alpha={MCS_ALPHA}$). Conditioning: ${cond_label}$",
        "tab:mcs_inclusion",
    )

    mcs_by_h = build_mcs_inclusion_by_horizon(all_results)
    if not mcs_by_h.empty:
        save_table(
            mcs_by_h.set_index(["H", "Model"]),
            "table6_mcs_by_horizon",
            f"MCS Inclusion by Horizon ($\\alpha={MCS_ALPHA}$). Conditioning: ${cond_label}$",
            "tab:mcs_by_horizon",
        )

    # ----------------------------------------------------------------
    # Table 7: Conditional FZ0 and VaR violations
    # ----------------------------------------------------------------
    log.info("Generating Table 7: Conditional FZ0 loss...")
    for alpha_risk in RISK_ALPHAS:
        fz0_table = build_conditional_fz0_table(all_results, alpha_risk)
        if not fz0_table.empty:
            save_table(
                fz0_table,
                f"table7_fz0_alpha{int(alpha_risk*100):02d}",
                f"Regime-Conditional FZ0 Loss ($\\alpha={alpha_risk}$). Conditioning: ${cond_label}$",
                f"tab:fz0_alpha{int(alpha_risk*100):02d}",
            )

        var_table = build_conditional_var_table(all_results, alpha_risk)
        if not var_table.empty:
            save_table(
                var_table,
                f"table7_var_violations_alpha{int(alpha_risk*100):02d}",
                f"VaR Violation Rates by Regime ($\\alpha={alpha_risk}$). Conditioning: ${cond_label}$",
                f"tab:var_viol_alpha{int(alpha_risk*100):02d}",
            )

    # ----------------------------------------------------------------
    # Raw results pickle for downstream use
    # ----------------------------------------------------------------
    import pickle

    pkl_path = OUTPUT_DIR / "all_results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(all_results, f)
    log.info("  Saved raw results to %s", pkl_path)


# ===================================================================
# 13. Summary Statistics
# ===================================================================

def print_summary(all_results: List[Dict]) -> None:
    """Print a concise summary of the analysis to stdout."""
    print("\n" + "=" * 70)
    print("PAPER 2 EMPIRICAL ANALYSIS -- SUMMARY")
    print("=" * 70)

    n_total = len(all_results)
    tickers = sorted(set(r["ticker"] for r in all_results))
    print(f"\nPanel: {len(tickers)} assets, {len(HORIZONS)} horizons, {n_total} total combinations")
    print(f"Assets: {', '.join(tickers)}")

    for H in HORIZONS:
        h_results = [r for r in all_results if r["H"] == H]
        if not h_results:
            continue

        print(f"\n--- Horizon H={H} ({len(h_results)} assets) ---")

        # Mean unconditional QLIKE
        unc_means = {}
        for m in MODEL_COLUMNS:
            vals = [r["unc_qlike"].get(m, np.nan) for r in h_results]
            unc_means[m] = np.nanmean(vals)

        best_unc = min(unc_means, key=unc_means.get)
        print(f"  Unconditional best (panel mean): {MODEL_LABELS[best_unc]} ({unc_means[best_unc]:.4f})")

        # Mean conditional QLIKE
        for regime, key in [("Low", "low_qlike"), ("High", "high_qlike")]:
            cond_means = {}
            for m in MODEL_COLUMNS:
                vals = [r[key].get(m, np.nan) for r in h_results]
                cond_means[m] = np.nanmean(vals)
            best_cond = min(cond_means, key=cond_means.get)
            print(
                f"  {regime}-regime best (panel mean): {MODEL_LABELS[best_cond]} ({cond_means[best_cond]:.4f})"
            )

        # MCS survival counts
        for regime_label, surv_key in [
            ("Unconditional", "mcs_unc_surv"),
            ("Low-regime", "mcs_low_surv"),
            ("High-regime", "mcs_high_surv"),
        ]:
            surv_counts = {m: 0 for m in MODEL_COLUMNS}
            for r in h_results:
                for m in r[surv_key]:
                    if m in surv_counts:
                        surv_counts[m] += 1
            n = len(h_results)
            top3 = sorted(surv_counts.items(), key=lambda x: -x[1])[:3]
            top3_str = ", ".join(
                f"{MODEL_LABELS[m]} ({c}/{n})" for m, c in top3
            )
            print(f"  MCS {regime_label}: {top3_str}")

        # Reversal count
        n_rev = sum(1 for r in h_results if r["reversal"]["reversal"])
        print(f"  Ranking reversals: {n_rev}/{len(h_results)} ({n_rev/len(h_results):.0%})")


# ===================================================================
# Main
# ===================================================================

def main():
    global MCS_NBOOT, MCS_ALPHA, LOW_QUANTILE, HIGH_QUANTILE

    parser = argparse.ArgumentParser(
        description="Paper 2: Regime-Conditional Scoring Rules -- Empirical Analysis"
    )
    parser.add_argument(
        "--conditioning",
        type=str,
        default="G_ssm",
        choices=["G_ssm", "G_obs", "G_proxy"],
        help="Conditioning variable for regime splitting (default: G_ssm)",
    )
    parser.add_argument(
        "--mcs-nboot",
        type=int,
        default=MCS_NBOOT,
        help=f"MCS bootstrap replications (default: {MCS_NBOOT})",
    )
    parser.add_argument(
        "--mcs-alpha",
        type=float,
        default=MCS_ALPHA,
        help=f"MCS significance level (default: {MCS_ALPHA})",
    )
    parser.add_argument(
        "--low-quantile",
        type=float,
        default=LOW_QUANTILE,
        help=f"Quantile threshold for low regime (default: {LOW_QUANTILE})",
    )
    parser.add_argument(
        "--high-quantile",
        type=float,
        default=HIGH_QUANTILE,
        help=f"Quantile threshold for high regime (default: {HIGH_QUANTILE})",
    )
    args = parser.parse_args()

    # Update globals from args
    MCS_NBOOT = args.mcs_nboot
    MCS_ALPHA = args.mcs_alpha
    LOW_QUANTILE = args.low_quantile
    HIGH_QUANTILE = args.high_quantile

    print("=" * 70)
    print("PAPER 2: REGIME-CONDITIONAL PROPER SCORING RULES")
    print("             Empirical Analysis")
    print("=" * 70)
    print(f"\nConditioning variable: {args.conditioning}")
    print(f"Regime thresholds: low <= Q({LOW_QUANTILE}), high >= Q({HIGH_QUANTILE})")
    print(f"MCS: alpha={MCS_ALPHA}, n_boot={MCS_NBOOT}, block={MCS_BLOCK}")
    print(f"Paper 1 outputs: {PAPER1_OUTPUTS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # ------------------------------------------------------------------
    # 1. Load all Paper 1 results
    # ------------------------------------------------------------------
    log.info("Step 1: Loading Paper 1 results...")
    data = load_all_results()

    if not data:
        log.error("No Paper 1 results found. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2-6. Run analysis for each (asset, horizon) pair
    # ------------------------------------------------------------------
    log.info("Step 2: Running analysis across %d asset-horizon pairs...", len(data))
    all_results = []

    for (ticker, H), df in sorted(data.items()):
        log.info("  Processing %s H=%d (%d obs)...", ticker, H, len(df))
        try:
            result = run_single_asset_horizon(df, ticker, H, args.conditioning)
            all_results.append(result)
        except Exception as exc:
            log.error("  FAILED %s H=%d: %s", ticker, H, exc)
            import traceback
            traceback.print_exc()
            continue

    if not all_results:
        log.error("All analyses failed. Exiting.")
        sys.exit(1)

    log.info("Completed %d / %d analyses successfully.", len(all_results), len(data))

    # ------------------------------------------------------------------
    # 7. Generate and save tables
    # ------------------------------------------------------------------
    log.info("Step 3: Generating output tables...")
    save_all_tables(all_results, args.conditioning)

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print_summary(all_results)

    print(f"\nAll tables saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
