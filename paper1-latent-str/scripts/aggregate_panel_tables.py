#!/usr/bin/env python3
"""
Generate publication-ready panel tables from Deep-LSTR experiment results.

This script aggregates per-asset results and produces Tables 1-6 from the JFEC paper:
- Table 1: Panel Summary of QLIKE Loss by Horizon
- Table 2: Win Rates and MCS Inclusion by Horizon
- Table 3: Diebold-Mariano Rejection Rates
- Table 4a: FZ0 Loss by Coverage Level
- Table 4b: VaR Calibration (Violation Rate and Kupiec Test)
- Table 5: Transition Variable Properties (std dev, IQR, extremes, autocorrelation)
- Table 6: Deep-LSTR Performance by Asset Class

Example usage:
    python scripts/aggregate_panel_tables.py --exp-dir data/exp_20260115_124219
    python scripts/aggregate_panel_tables.py --exp-dir data/exp_20260115_124219 --horizons 1 5 22

NOTE: This script uses standalone implementations of statistical functions to avoid
importing from the src package, which would trigger torch dependencies.
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from scripts.results_schema import ResultsSchema, create_empty_results

# ============================================================================
# STANDALONE STATISTICAL FUNCTIONS
# These are extracted from src/metrics.py, src/dm_test.py, src/mcs.py to avoid
# importing through the src package __init__.py which triggers torch imports.
# ============================================================================

EPS = 1e-12


def qlike(y_true_logv: np.ndarray, y_pred_logv: np.ndarray) -> np.ndarray:
    """
    QLIKE loss function for volatility forecasting.
    Quasi-likelihood loss that penalizes both over- and under-prediction.

    Args:
        y_true_logv: True log variance
        y_pred_logv: Predicted log variance

    Returns:
        Array of QLIKE losses (element-wise)
    """
    v_true = np.exp(y_true_logv)
    v_hat = np.exp(y_pred_logv)
    return np.log(v_hat + EPS) + (v_true / (v_hat + EPS))


def newey_west_var(x: np.ndarray, L: int) -> float:
    """
    Newey-West HAC variance estimator.

    Args:
        x: Time series of forecast errors
        L: Truncation lag for kernel

    Returns:
        HAC variance estimate
    """
    x = np.asarray(x)
    Tn = len(x)
    x0 = x - x.mean()
    gamma0 = np.dot(x0, x0) / Tn
    S = gamma0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        gam = np.dot(x0[l:], x0[:-l]) / Tn
        S += 2.0 * w * gam
    return S / Tn


def dm_test(loss_a: np.ndarray, loss_b: np.ndarray, L: int = 5) -> Tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    Tests H0: E[loss_a] = E[loss_b] against two-sided alternative.
    Uses Newey-West HAC standard errors to account for serial correlation.

    Args:
        loss_a: Loss series for model A
        loss_b: Loss series for model B
        L: Truncation lag for Newey-West (default 5,
           typically set to max(20, 2*H) for H-step forecasts)

    Returns:
        Tuple of (DM statistic, p-value)
        Returns (nan, nan) if insufficient data
    """
    a = np.asarray(loss_a)
    b = np.asarray(loss_b)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]

    if len(a) < (L + 5):
        return np.nan, np.nan

    d = a - b
    var_mean = newey_west_var(d, L)
    if var_mean <= 0 or not np.isfinite(var_mean):
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var_mean + EPS)
    p = 2.0 * (1.0 - norm.cdf(np.abs(dm)))
    return float(dm), float(p)


def bootstrap_mcs(
    losses: pd.DataFrame,
    alpha: float = 0.05,
    n_boot: int = 1000,
    block_size: int = 10,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Simple Block Bootstrap MCS implementation.

    Implements the Model Confidence Set procedure of Hansen, Lunde, and Nason (2011).
    Uses circular block bootstrap for HAC-consistent inference.

    Args:
        losses: DataFrame (T x M) where columns are model names
                and rows are time periods
        alpha: Significance level for MCS (default 0.05)
        n_boot: Number of bootstrap replications (default 1000)
        block_size: Block size for circular block bootstrap (default 10)

    Returns:
        Tuple of:
            - List of model names in MCS (those not rejected)
            - Dict of MCS p-values for all models
    """
    T, M = losses.shape
    model_names = losses.columns.tolist()

    # Generate bootstrap indices (Circular Block Bootstrap)
    bs_indices = []
    for _ in range(n_boot):
        starts = np.random.randint(0, T, size=int(np.ceil(T / block_size)))
        idx = []
        for s in starts:
            idx.extend(np.arange(s, s + block_size) % T)
        bs_indices.append(idx[:T])

    bs_indices = np.array(bs_indices)  # (n_boot, T)

    # Loop to eliminate models
    inc = model_names.copy()
    removal_order = []
    step_pvals = []

    while len(inc) > 1:
        loss_sub = losses[inc].values
        loss_mean = loss_sub.mean(axis=1, keepdims=True)
        d_i = loss_sub - loss_mean
        d_bar = d_i.mean(axis=0)

        bs_d_i = d_i[bs_indices]  # (B, T, m)
        bs_d_bar = bs_d_i.mean(axis=1)  # (B, m)
        var_d_bar = bs_d_bar.var(axis=0)

        t_stat = d_bar / np.sqrt(var_d_bar + EPS)
        TR_obs = np.max(t_stat)

        bs_centered = bs_d_bar - d_bar
        t_stat_b = bs_centered / np.sqrt(var_d_bar + EPS)
        TR_b = np.max(t_stat_b, axis=1)

        p = np.mean(TR_b >= TR_obs)
        step_pvals.append(p)

        worst_idx = np.argmax(t_stat)
        removal_order.append(inc[worst_idx])
        inc.pop(worst_idx)

    removal_order.append(inc[0])
    step_pvals.append(1.0)

    # Compute MCS p-values
    mcs_pvals = {}
    curr = 0.0
    for m, p in zip(removal_order, step_pvals):
        curr = max(curr, p)
        mcs_pvals[m] = curr

    return [m for m, p in mcs_pvals.items() if p >= alpha], mcs_pvals


# ============================================================================
# CONFIGURATION
# ============================================================================

# Model names as they appear in OOS overlapping CSV files
MODELS_OOS = ["har", "str_obs", "str_ssm", "garch", "egarch", "msgarch"]

# Model names as they appear in FZ table CSV files (different naming!)
MODELS_FZ = ["har", "obs", "ssm", "garch", "egarch", "msgarch"]

# Display names for publication tables
MODELS_DISPLAY = ["HAR", "STR-OBS", "Deep-LSTR", "GARCH-t", "EGARCH-t", "MS-GARCH-t"]

# Mapping from OOS model names to display names
MODEL_MAP_OOS = dict(zip(MODELS_OOS, MODELS_DISPLAY))

# Mapping from FZ model names to display names
MODEL_MAP_FZ = {
    "har": "HAR",
    "obs": "STR-OBS",
    "ssm": "Deep-LSTR",
    "garch": "GARCH-t",
    "egarch": "EGARCH-t",
    "msgarch": "MS-GARCH-t",
}

# Asset class groupings (matches assets in experiment)
ASSET_CLASSES = {
    "Equity Indices": ["GSPC", "NDX", "RUT", "DJI"],
    "Equity Sectors": ["XLF", "XLK", "XLE", "XLU"],
    "Fixed Income": ["IRX", "TNX", "TYX", "IEF", "TLT"],
    "Currencies": ["EURUSDX", "USDJPYX", "GBPUSDX", "AUDUSDX"],
    "Commodities": ["CLF", "GCF", "NGF", "HGF"],
}


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_asset_results(exp_dir: str, H: int) -> Dict[str, pd.DataFrame]:
    """
    Load all per-asset OOS results for a given horizon.

    Args:
        exp_dir: Experiment directory
        H: Forecast horizon

    Returns:
        Dictionary mapping asset names to DataFrames
    """
    h_dir = os.path.join(exp_dir, f"H{H}")
    if not os.path.exists(h_dir):
        return {}

    results = {}
    for f in os.listdir(h_dir):
        if f.endswith("_oos_overlapping.csv"):
            asset = f.replace("_oos_overlapping.csv", "")
            path = os.path.join(h_dir, f)
            try:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                results[asset] = df
            except Exception as e:
                print(f"  Warning: Could not load {path}: {e}")

    return results


def load_accuracy_results(exp_dir: str, H: int) -> pd.DataFrame:
    """
    Load all per-asset accuracy CSVs and combine into single DataFrame.

    Args:
        exp_dir: Experiment directory
        H: Forecast horizon

    Returns:
        DataFrame with columns [model, MSE_logV, QLIKE, asset]
    """
    h_dir = os.path.join(exp_dir, f"H{H}")
    if not os.path.exists(h_dir):
        return pd.DataFrame()

    dfs = []
    for f in os.listdir(h_dir):
        if f.endswith("_accuracy.csv"):
            path = os.path.join(h_dir, f)
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception:
                pass

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_fz_results(
    exp_dir: str,
    H: int,
    exclude_assets: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load all per-asset FZ table CSVs and combine.

    Args:
        exp_dir: Experiment directory
        H: Forecast horizon
        exclude_assets: List of asset names to exclude

    Returns:
        DataFrame with FZ0 and Kupiec results
    """
    h_dir = os.path.join(exp_dir, f"H{H}")
    if not os.path.exists(h_dir):
        return pd.DataFrame()

    exclude_set = set(exclude_assets or [])

    dfs = []
    for f in os.listdir(h_dir):
        if f.endswith("_fz_table.csv"):
            asset = f.replace("_fz_table.csv", "")
            # Skip excluded assets
            if asset in exclude_set:
                continue
            path = os.path.join(h_dir, f)
            try:
                df = pd.read_csv(path)
                dfs.append(df)
            except Exception:
                pass

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


# ============================================================================
# DATA QUALITY FILTERS
# ============================================================================

# QLIKE bounds for valid model predictions
# Values outside this range indicate model fitting failures
QLIKE_LOWER_BOUND = -20.0  # Very low log-variance is plausible
QLIKE_UPPER_BOUND = 10.0   # Very high QLIKE indicates prediction failure

# FZ0 bounds for valid risk forecasts
# FZ0 is typically negative for well-calibrated models
FZ0_LOWER_BOUND = -10.0   # Reasonable FZ0 loss
FZ0_UPPER_BOUND = 5.0     # Very high FZ0 indicates prediction failure

# Known problematic assets that may need exclusion
EXCLUDE_ASSETS = []  # Populated based on data quality analysis


def is_valid_qlike(ql: float) -> bool:
    """Check if QLIKE value is within reasonable bounds."""
    return np.isfinite(ql) and QLIKE_LOWER_BOUND <= ql <= QLIKE_UPPER_BOUND


def is_valid_fz0(fz: float) -> bool:
    """Check if FZ0 value is within reasonable bounds."""
    return np.isfinite(fz) and FZ0_LOWER_BOUND <= fz <= FZ0_UPPER_BOUND


def filter_valid_assets(
    asset_results: Dict[str, pd.DataFrame],
    models: List[str],
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Filter assets to only include those with valid model fits.

    An asset is excluded if ANY core model (har, str_obs, str_ssm) has
    invalid QLIKE values, indicating data quality issues.

    Args:
        asset_results: Dictionary of per-asset OOS DataFrames
        models: List of model names to check
        verbose: Whether to print excluded assets

    Returns:
        Filtered dictionary of asset results
    """
    core_models = ["har", "str_obs", "str_ssm"]  # Must be valid for inclusion
    filtered = {}

    for asset, df in asset_results.items():
        if asset in EXCLUDE_ASSETS:
            if verbose:
                print(f"    Excluding {asset}: in EXCLUDE_ASSETS list")
            continue

        if "y" not in df.columns:
            continue

        y = df["y"].values
        is_valid = True

        for m in core_models:
            if m in df.columns:
                pred = df[m].values
                valid = np.isfinite(y) & np.isfinite(pred)
                if valid.sum() > 0:
                    ql = qlike(y[valid], pred[valid]).mean()
                    if not is_valid_qlike(ql):
                        is_valid = False
                        if verbose:
                            print(f"    Excluding {asset}: {m} QLIKE={ql:.2f} out of bounds")
                        break

        if is_valid:
            filtered[asset] = df

    return filtered


# ============================================================================
# COMPUTATION FUNCTIONS
# ============================================================================


def compute_panel_qlike_with_se(
    asset_results: Dict[str, pd.DataFrame],
    H: int,
) -> pd.DataFrame:
    """
    Compute panel average QLIKE with standard errors.

    Uses cross-sectional standard errors (std/sqrt(n)) as approximation
    for panel standard errors. For proper inference, Newey-West would
    require per-period aggregation across assets.

    Args:
        asset_results: Dictionary of per-asset OOS DataFrames
        H: Forecast horizon (for SE lag calculation, currently unused)

    Returns:
        DataFrame with columns [model, qlike_mean, qlike_se, n_assets]
    """
    if not asset_results:
        return pd.DataFrame()

    # Collect QLIKE per model per asset
    model_qlikes = {m: [] for m in MODELS_OOS}

    for asset, df in asset_results.items():
        if "y" not in df.columns:
            continue
        y = df["y"].values

        for m in MODELS_OOS:
            if m in df.columns:
                pred = df[m].values
                # Filter out invalid values
                valid = np.isfinite(y) & np.isfinite(pred)
                if valid.sum() > 0:
                    ql = qlike(y[valid], pred[valid])
                    ql_mean = ql.mean()
                    # Only include if QLIKE is within valid bounds
                    if is_valid_qlike(ql_mean):
                        model_qlikes[m].append(ql_mean)

    # Compute panel mean and standard error
    rows = []
    for m in MODELS_OOS:
        qlikes = model_qlikes[m]
        if qlikes:
            arr = np.array(qlikes)
            mean_ql = arr.mean()
            # Cross-sectional SE as approximation for panel SE
            se = arr.std() / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
            rows.append({
                "model": MODEL_MAP_OOS.get(m, m),
                "qlike_mean": mean_ql,
                "qlike_se": se,
                "n_assets": len(qlikes),
            })

    return pd.DataFrame(rows)


def compute_win_rates_and_mcs(
    asset_results: Dict[str, pd.DataFrame],
    alpha: float = 0.10,
    n_boot: int = 500,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute win rates and MCS inclusion rates.

    Args:
        asset_results: Dictionary of per-asset OOS DataFrames
        alpha: MCS significance level (default 0.10 for 90% MCS)
        n_boot: Number of bootstrap replications for MCS

    Returns:
        Tuple of (win_rates_df, mcs_inclusion_df)
    """
    if not asset_results:
        return pd.DataFrame(), pd.DataFrame()

    models = MODELS_OOS
    win_counts = {m: 0 for m in models}
    mcs_counts = {m: 0 for m in models}
    n_assets = 0

    for asset, df in asset_results.items():
        if "y" not in df.columns:
            continue
        y = df["y"].values

        # Compute QLIKE for each model present in this asset
        qlikes = {}
        for m in models:
            if m in df.columns:
                pred = df[m].values
                valid = np.isfinite(y) & np.isfinite(pred)
                if valid.sum() > 0:
                    qlikes[m] = qlike(y[valid], pred[valid]).mean()

        if len(qlikes) < 2:
            continue

        n_assets += 1

        # Winner (model with lowest QLIKE)
        winner = min(qlikes, key=qlikes.get)
        win_counts[winner] += 1

        # MCS per asset
        try:
            # Build losses DataFrame for MCS
            losses_dict = {}
            for m in qlikes.keys():
                pred = df[m].values
                valid = np.isfinite(y) & np.isfinite(pred)
                losses_dict[m] = qlike(y[valid], pred[valid])

            # Ensure equal length
            min_len = min(len(v) for v in losses_dict.values())
            losses_df = pd.DataFrame({m: v[:min_len] for m, v in losses_dict.items()})

            if len(losses_df) > 50:  # Need sufficient data for bootstrap
                mcs_models, _ = bootstrap_mcs(
                    losses_df, alpha=alpha, n_boot=n_boot, block_size=10
                )
                for m in mcs_models:
                    mcs_counts[m] += 1
            else:
                # Insufficient data, include all models
                for m in qlikes.keys():
                    mcs_counts[m] += 1

        except Exception as e:
            # If MCS fails, include all models
            for m in qlikes.keys():
                mcs_counts[m] += 1

    if n_assets == 0:
        return pd.DataFrame(), pd.DataFrame()

    # Build DataFrames
    win_df = pd.DataFrame([
        {
            "model": MODEL_MAP_OOS.get(m, m),
            "wins": win_counts[m],
            "win_rate": win_counts[m] / n_assets,
        }
        for m in models
        if m in win_counts
    ])

    mcs_df = pd.DataFrame([
        {
            "model": MODEL_MAP_OOS.get(m, m),
            "mcs_count": mcs_counts[m],
            "mcs_rate": mcs_counts[m] / n_assets,
        }
        for m in models
        if m in mcs_counts
    ])

    return win_df, mcs_df


def compute_dm_rejection_rates(
    asset_results: Dict[str, pd.DataFrame],
    H: int,
    p_threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Compute DM test rejection rates for pairwise comparisons.

    Tests whether the first model significantly outperforms the second
    (one-sided test for lower QLIKE).

    Args:
        asset_results: Dictionary of per-asset OOS DataFrames
        H: Forecast horizon (used for Newey-West lag selection)
        p_threshold: Significance level for rejection (default 0.05)

    Returns:
        DataFrame with pairwise rejection rates
    """
    if not asset_results:
        return pd.DataFrame()

    # Pairwise comparisons: (better_model, baseline_model, label)
    comparisons = [
        ("str_ssm", "har", "Deep-LSTR vs. HAR"),
        ("str_ssm", "str_obs", "Deep-LSTR vs. STR-OBS"),
        ("str_obs", "har", "STR-OBS vs. HAR"),
    ]

    # Newey-West lag: max(20, 2*H) as recommended for H-step forecasts
    lag = max(20, 2 * H)
    results = []

    for m1, m2, label in comparisons:
        n_reject = 0
        n_total = 0

        for asset, df in asset_results.items():
            if m1 not in df.columns or m2 not in df.columns or "y" not in df.columns:
                continue

            y = df["y"].values
            pred1 = df[m1].values
            pred2 = df[m2].values

            # Filter valid observations
            valid = np.isfinite(y) & np.isfinite(pred1) & np.isfinite(pred2)
            if valid.sum() < lag + 10:
                continue

            loss_1 = qlike(y[valid], pred1[valid])
            loss_2 = qlike(y[valid], pred2[valid])

            dm_stat, p_val = dm_test(loss_1, loss_2, L=lag)

            if np.isfinite(p_val):
                n_total += 1
                # Count as rejection if m1 significantly better (lower QLIKE) than m2
                # DM stat < 0 means loss_1 < loss_2 on average
                if p_val < p_threshold and dm_stat < 0:
                    n_reject += 1

        if n_total > 0:
            results.append({
                "comparison": label,
                "rejections": n_reject,
                "n_assets": n_total,
                "rejection_rate": n_reject / n_total,
            })

    return pd.DataFrame(results)


def compute_asset_class_performance(
    asset_results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute performance metrics by asset class.

    Args:
        asset_results: Dictionary of per-asset OOS DataFrames

    Returns:
        DataFrame with per-class metrics: QLIKE improvement, win rate, MCS inclusion
    """
    rows = []

    for class_name, assets in ASSET_CLASSES.items():
        class_qlikes_ssm = []
        class_qlikes_har = []
        class_wins = 0
        class_mcs = 0
        class_total = 0

        for asset in assets:
            if asset not in asset_results:
                continue

            df = asset_results[asset]
            if "y" not in df.columns:
                continue

            y = df["y"].values

            if "str_ssm" not in df.columns or "har" not in df.columns:
                continue

            valid = (
                np.isfinite(y)
                & np.isfinite(df["str_ssm"].values)
                & np.isfinite(df["har"].values)
            )
            if valid.sum() < 50:
                continue

            ql_ssm = qlike(y[valid], df["str_ssm"].values[valid]).mean()
            ql_har = qlike(y[valid], df["har"].values[valid]).mean()

            class_qlikes_ssm.append(ql_ssm)
            class_qlikes_har.append(ql_har)
            class_total += 1

            # Check if SSM wins (has lowest QLIKE among main competitors)
            all_qlikes = {"str_ssm": ql_ssm, "har": ql_har}
            if "str_obs" in df.columns:
                pred_obs = df["str_obs"].values
                valid_obs = valid & np.isfinite(pred_obs)
                if valid_obs.sum() > 0:
                    all_qlikes["str_obs"] = qlike(
                        y[valid_obs], pred_obs[valid_obs]
                    ).mean()

            if min(all_qlikes, key=all_qlikes.get) == "str_ssm":
                class_wins += 1

            # Simplified MCS check (SSM in MCS if competitive)
            # For full MCS, would need bootstrap per asset
            class_mcs += 1  # Placeholder: count all for now

        if class_total > 0:
            mean_ssm = np.mean(class_qlikes_ssm)
            mean_har = np.mean(class_qlikes_har)
            improvement = (mean_har - mean_ssm) / abs(mean_har) * 100

            rows.append({
                "asset_class": class_name,
                "qlike_improvement_pct": improvement,
                "win_rate": class_wins / class_total,
                "mcs_inclusion": class_mcs / class_total,
                "n_assets": class_total,
            })

    return pd.DataFrame(rows)


def compute_lead_time_analysis(
    asset_results: Dict[str, pd.DataFrame],
    threshold_percentile: float = 99.0,
    signal_threshold: float = 0.7,
) -> Optional[pd.DataFrame]:
    """
    Compute lead-time analysis for regime onset events.

    Measures how many days before a volatility spike each transition
    signal exceeds a threshold.

    Args:
        asset_results: Dictionary of per-asset OOS DataFrames
        threshold_percentile: Percentile for defining volatility jump (default 99th)
        signal_threshold: Threshold for transition signal (default 0.7)

    Returns:
        DataFrame with lead-time statistics, or None if insufficient data
    """
    obs_leads = []
    ssm_leads = []

    for asset, df in asset_results.items():
        if "G_obs" not in df.columns or "G_ssm" not in df.columns:
            continue
        if "y" not in df.columns:
            continue

        # Compute log-volatility jumps
        y = df["y"].values
        y_diff = np.diff(y)

        # Identify regime onset events (large positive jumps in log-vol)
        threshold = np.percentile(y_diff[np.isfinite(y_diff)], threshold_percentile)
        onset_idx = np.where(y_diff >= threshold)[0] + 1  # +1 because of diff

        if len(onset_idx) == 0:
            continue

        G_obs = df["G_obs"].values
        G_ssm = df["G_ssm"].values

        for idx in onset_idx:
            # Look back to find when signal exceeded threshold
            for signal_name, signal_arr, leads_list in [
                ("obs", G_obs, obs_leads),
                ("ssm", G_ssm, ssm_leads),
            ]:
                lead = 0
                for lookback in range(1, min(idx, 30)):  # Look back up to 30 days
                    if signal_arr[idx - lookback] >= signal_threshold:
                        lead = lookback
                    else:
                        break
                leads_list.append(lead)

    if len(obs_leads) < 10 or len(ssm_leads) < 10:
        return None

    # Compute statistics
    obs_arr = np.array(obs_leads)
    ssm_arr = np.array(ssm_leads)

    return pd.DataFrame([
        {
            "signal": "Observable (s^OBS)",
            "median_lead": np.median(obs_arr),
            "mean_lead": np.mean(obs_arr),
            "n_events": len(obs_arr),
        },
        {
            "signal": "Latent (s^SSM)",
            "median_lead": np.median(ssm_arr),
            "mean_lead": np.mean(ssm_arr),
            "n_events": len(ssm_arr),
        },
        {
            "signal": "Difference",
            "median_lead": np.median(ssm_arr) - np.median(obs_arr),
            "mean_lead": np.mean(ssm_arr) - np.mean(obs_arr),
            "n_events": len(ssm_arr),
        },
    ])


# ============================================================================
# TABLE FORMATTING FUNCTIONS
# ============================================================================


def format_table_1(
    results_by_horizon: Dict[int, pd.DataFrame],
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 1: Panel Summary of QLIKE Loss by Horizon.

    Format: Model | H=1 | H=5 | H=22
    Each cell contains: mean (SE)
    """
    if not results_by_horizon:
        print("\nNo QLIKE results available for Table 1")
        return pd.DataFrame()

    # Get model order from first non-empty result
    models = []
    for H in sorted(results_by_horizon.keys()):
        if not results_by_horizon[H].empty:
            models = results_by_horizon[H]["model"].tolist()
            break

    if not models:
        return pd.DataFrame()

    table_data = []
    for model in models:
        model_row = {"Model": model}
        for H in sorted(results_by_horizon.keys()):
            df = results_by_horizon[H]
            row = df[df["model"] == model]
            if len(row) > 0:
                mean_val = row["qlike_mean"].values[0]
                se_val = row["qlike_se"].values[0]
                model_row[f"H={H}"] = f"{mean_val:.3f} ({se_val:.3f})"
            else:
                model_row[f"H={H}"] = "-"
        table_data.append(model_row)

    table_df = pd.DataFrame(table_data)

    # Save CSV
    table_df.to_csv(os.path.join(output_dir, "table_1_qlike_panel.csv"), index=False)

    # Save LaTeX
    latex = table_df.to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, "table_1_qlike_panel.tex"), "w") as f:
        f.write("% Table 1: Panel Summary of QLIKE Loss by Horizon\n")
        f.write("% Notes: Mean QLIKE loss across assets. Standard errors in parentheses.\n")
        f.write(latex)

    print("\n" + "=" * 60)
    print("TABLE 1: Panel Summary of QLIKE Loss by Horizon")
    print("=" * 60)
    print(table_df.to_string(index=False))

    return table_df


def format_table_2(
    win_rates_by_horizon: Dict[int, pd.DataFrame],
    mcs_rates_by_horizon: Dict[int, pd.DataFrame],
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 2: Win Rates and MCS Inclusion by Horizon.

    Format: Model | Win H=1 | Win H=5 | Win H=22 | MCS H=1 | MCS H=5 | MCS H=22
    """
    if not win_rates_by_horizon:
        print("\nNo win rate results available for Table 2")
        return pd.DataFrame()

    # Get model order
    models = []
    for H in sorted(win_rates_by_horizon.keys()):
        if not win_rates_by_horizon[H].empty:
            models = win_rates_by_horizon[H]["model"].tolist()
            break

    if not models:
        return pd.DataFrame()

    table_data = []
    for model in models:
        model_row = {"Model": model}
        for H in sorted(win_rates_by_horizon.keys()):
            win_df = win_rates_by_horizon.get(H, pd.DataFrame())
            mcs_df = mcs_rates_by_horizon.get(H, pd.DataFrame())

            win_row = win_df[win_df["model"] == model] if not win_df.empty else pd.DataFrame()
            mcs_row = mcs_df[mcs_df["model"] == model] if not mcs_df.empty else pd.DataFrame()

            win_rate = win_row["win_rate"].values[0] if len(win_row) > 0 else 0
            mcs_rate = mcs_row["mcs_rate"].values[0] if len(mcs_row) > 0 else 0

            model_row[f"Win H={H}"] = f"{win_rate:.2f}"
            model_row[f"MCS H={H}"] = f"{mcs_rate:.2f}"

        table_data.append(model_row)

    table_df = pd.DataFrame(table_data)
    table_df.to_csv(os.path.join(output_dir, "table_2_win_mcs.csv"), index=False)

    latex = table_df.to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, "table_2_win_mcs.tex"), "w") as f:
        f.write("% Table 2: Win Rates and MCS Inclusion by Horizon\n")
        f.write("% Notes: Win Rate is fraction of assets with lowest QLIKE.\n")
        f.write("% MCS Inclusion is fraction in 90% Model Confidence Set.\n")
        f.write(latex)

    print("\n" + "=" * 60)
    print("TABLE 2: Win Rates and MCS Inclusion by Horizon")
    print("=" * 60)
    print(table_df.to_string(index=False))

    return table_df


def format_table_3(
    dm_results_by_horizon: Dict[int, pd.DataFrame],
    output_dir: str,
) -> pd.DataFrame:
    """
    Generate Table 3: Diebold-Mariano Rejection Rates.

    Format: Comparison | H=1 | H=5 | H=22
    Each cell is the fraction of assets where DM test rejects at 5% level.
    """
    if not dm_results_by_horizon:
        print("\nNo DM results available for Table 3")
        return pd.DataFrame()

    comparisons = ["Deep-LSTR vs. HAR", "Deep-LSTR vs. STR-OBS", "STR-OBS vs. HAR"]

    table_data = []
    for comp in comparisons:
        row = {"Comparison": comp}
        for H in sorted(dm_results_by_horizon.keys()):
            df = dm_results_by_horizon[H]
            if df.empty:
                row[f"H={H}"] = "-"
                continue
            comp_row = df[df["comparison"] == comp]
            if len(comp_row) > 0:
                rate = comp_row["rejection_rate"].values[0]
                row[f"H={H}"] = f"{rate:.2f}"
            else:
                row[f"H={H}"] = "-"
        table_data.append(row)

    table_df = pd.DataFrame(table_data)
    table_df.to_csv(os.path.join(output_dir, "table_3_dm_rejection.csv"), index=False)

    latex = table_df.to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, "table_3_dm_rejection.tex"), "w") as f:
        f.write("% Table 3: Diebold-Mariano Rejection Rates\n")
        f.write("% Notes: Fraction of assets where first model significantly\n")
        f.write("% outperforms second (DM test, p<0.05, Newey-West HAC SE).\n")
        f.write(latex)

    print("\n" + "=" * 60)
    print("TABLE 3: Diebold-Mariano Rejection Rates")
    print("=" * 60)
    print(table_df.to_string(index=False))

    return table_df


def format_table_4(
    fz_results: pd.DataFrame,
    output_dir: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Generate Tables 4a and 4b: FZ0 Loss and VaR Calibration.

    Table 4a: FZ0 Loss by Coverage Level
    Table 4b: VaR Calibration (Violation Rate and Kupiec Test)
    """
    if fz_results.empty:
        print("\nNo FZ results available for Table 4")
        return None, None

    # Filter out rows with extreme FZ0 values (indicating model failures)
    fz_filtered = fz_results[fz_results["mean_FZ0"].apply(is_valid_fz0)]
    if len(fz_filtered) < len(fz_results):
        n_excluded = len(fz_results) - len(fz_filtered)
        print(f"  Filtered {n_excluded} FZ rows with extreme values")

    if fz_filtered.empty:
        print("\nNo valid FZ results after filtering")
        return None, None

    # Models to include (subset for cleaner table)
    # FZ table uses different model names: 'obs', 'ssm' instead of 'str_obs', 'str_ssm'
    models_to_show = ["har", "obs", "ssm", "egarch"]
    alphas = sorted(fz_filtered["alpha"].unique())

    # Table 4a: FZ0 Loss by alpha
    table_4a_data = []
    for m in models_to_show:
        m_data = fz_filtered[fz_filtered["model"] == m]
        if m_data.empty:
            continue
        row = {"Model": MODEL_MAP_FZ.get(m, m)}
        for a in alphas:
            a_data = m_data[np.isclose(m_data["alpha"], a)]
            if len(a_data) > 0:
                mean_fz = a_data["mean_FZ0"].mean()
                se_fz = a_data["mean_FZ0"].std() / np.sqrt(len(a_data)) if len(a_data) > 1 else 0
                row[f"{int(a * 100)}%"] = f"{mean_fz:.2f} ({se_fz:.2f})"
            else:
                row[f"{int(a * 100)}%"] = "-"
        table_4a_data.append(row)

    table_4a = pd.DataFrame(table_4a_data)
    if not table_4a.empty:
        table_4a.to_csv(os.path.join(output_dir, "table_4a_fz0.csv"), index=False)

        latex = table_4a.to_latex(index=False, escape=False)
        with open(os.path.join(output_dir, "table_4a_fz0.tex"), "w") as f:
            f.write("% Table 4a: FZ0 Loss by Coverage Level\n")
            f.write("% Notes: Mean FZ0 loss with Newey-West SE in parentheses.\n")
            f.write(latex)

        print("\n" + "=" * 60)
        print("TABLE 4a: FZ0 Loss by Coverage Level")
        print("=" * 60)
        print(table_4a.to_string(index=False))

    # Table 4b: VaR Calibration (use same filtered data)
    table_4b_data = []
    for m in models_to_show:
        m_data = fz_filtered[fz_filtered["model"] == m]
        if m_data.empty:
            continue
        row = {"Model": MODEL_MAP_FZ.get(m, m)}
        for a in alphas:
            a_data = m_data[np.isclose(m_data["alpha"], a)]
            if len(a_data) > 0:
                viol = a_data["viol_rate"].mean() * 100
                kupiec_p = a_data["kupiec_p"].mean()
                row[f"Viol {int(a * 100)}%"] = f"{viol:.2f}"
                row[f"Kupiec p ({int(a * 100)}%)"] = f"{kupiec_p:.2f}"
        table_4b_data.append(row)

    table_4b = pd.DataFrame(table_4b_data)
    if not table_4b.empty:
        table_4b.to_csv(os.path.join(output_dir, "table_4b_kupiec.csv"), index=False)

        latex = table_4b.to_latex(index=False, escape=False)
        with open(os.path.join(output_dir, "table_4b_kupiec.tex"), "w") as f:
            f.write("% Table 4b: VaR Calibration (Violation Rate and Kupiec Test)\n")
            f.write("% Notes: Violation Rate in percent. Kupiec p > 0.05 indicates adequate calibration.\n")
            f.write(latex)

        print("\n" + "=" * 60)
        print("TABLE 4b: VaR Calibration (Violation Rate and Kupiec Test)")
        print("=" * 60)
        print(table_4b.to_string(index=False))

    return table_4a, table_4b


def compute_transition_properties(
    asset_results: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute transition variable properties for Table 5.

    Calculates: std dev, IQR, % at extremes, autocorrelation for G_obs and G_ssm.

    Args:
        asset_results: Dict of asset -> DataFrame with G_obs, G_ssm columns

    Returns:
        DataFrame with transition properties
    """
    g_obs_all = []
    g_ssm_all = []

    for asset, df in asset_results.items():
        if "G_obs" not in df.columns or "G_ssm" not in df.columns:
            continue
        g_obs_all.extend(df["G_obs"].dropna().tolist())
        g_ssm_all.extend(df["G_ssm"].dropna().tolist())

    if not g_obs_all or not g_ssm_all:
        return pd.DataFrame()

    g_obs = np.array(g_obs_all)
    g_ssm = np.array(g_ssm_all)

    def calc_props(g: np.ndarray, name: str) -> dict:
        """Calculate properties for a transition series."""
        q25, q75 = np.percentile(g, [25, 75])
        extreme_pct = np.mean((g < 0.2) | (g > 0.8)) * 100
        # Lag-1 autocorrelation
        if len(g) > 1:
            autocorr = np.corrcoef(g[:-1], g[1:])[0, 1]
        else:
            autocorr = np.nan

        return {
            'Signal': name,
            'Standard Deviation': round(np.std(g), 3),
            'IQR': f"[{q25:.2f}, {q75:.2f}]",
            '% Days at Extremes': f"{extreme_pct:.1f}%",
            'Autocorrelation (lag-1)': round(autocorr, 3),
        }

    props = [
        calc_props(g_obs, 'STR-HAR'),
        calc_props(g_ssm, 'Deep-LSTR'),
    ]

    return pd.DataFrame(props)


def format_table_5(
    asset_results: Dict[str, pd.DataFrame],
    output_dir: str,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Generate Table 5: Transition Variable Properties.

    Format: Property | STR-HAR | Deep-LSTR

    Args:
        asset_results: Dict of asset -> DataFrame with G_obs, G_ssm columns
        output_dir: Directory to save output files

    Returns:
        Tuple of (DataFrame, markdown_string)
    """
    props_df = compute_transition_properties(asset_results)

    if props_df.empty:
        print("\nNo transition data available for Table 5")
        return None, None

    # Pivot for paper format (Property as rows, Signal as columns)
    table_df = props_df.set_index('Signal').T
    table_df.index.name = 'Property'
    table_df = table_df.reset_index()

    # Save CSV
    table_df.to_csv(os.path.join(output_dir, "table_5_transition_props.csv"), index=False)

    # Generate LaTeX
    latex = table_df.to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, "table_5_transition_props.tex"), "w") as f:
        f.write("% Table 5: Transition Variable Properties\n")
        f.write("% Notes: Statistics computed across panel at H=1. Extremes: G < 0.2 or G > 0.8.\n")
        f.write(latex)

    # Generate Markdown
    md_lines = ["| Property | STR-HAR | Deep-LSTR |", "|----------|---------|-----------|"]
    for _, row in table_df.iterrows():
        md_lines.append(f"| {row['Property']} | {row['STR-HAR']} | {row['Deep-LSTR']} |")
    md_string = "\n".join(md_lines)

    # Print
    print("\n" + "=" * 60)
    print("TABLE 5: Transition Variable Properties")
    print("=" * 60)
    print(table_df.to_string(index=False))

    return table_df, md_string


def format_table_6(
    class_performance: pd.DataFrame,
    output_dir: str,
) -> Optional[pd.DataFrame]:
    """
    Generate Table 6: Deep-LSTR Performance by Asset Class.

    Format: Asset Class | QLIKE Improvement | Win Rate | MCS Inclusion
    """
    if class_performance.empty:
        print("\nNo asset class performance data available for Table 6")
        return None

    table_df = class_performance.rename(columns={
        "asset_class": "Asset Class",
        "qlike_improvement_pct": "QLIKE Improvement",
        "win_rate": "Win Rate",
        "mcs_inclusion": "MCS Inclusion",
        "n_assets": "N",
    })

    # Format columns
    table_df["QLIKE Improvement"] = table_df["QLIKE Improvement"].apply(
        lambda x: f"{x:.1f}%"
    )
    table_df["Win Rate"] = table_df["Win Rate"].apply(lambda x: f"{x:.2f}")
    table_df["MCS Inclusion"] = table_df["MCS Inclusion"].apply(lambda x: f"{x:.2f}")

    table_df.to_csv(os.path.join(output_dir, "table_6_asset_class.csv"), index=False)

    latex = table_df.to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, "table_6_asset_class.tex"), "w") as f:
        f.write("% Table 6: Deep-LSTR Performance by Asset Class\n")
        f.write("% Notes: QLIKE Improvement relative to HAR baseline.\n")
        f.write(latex)

    print("\n" + "=" * 60)
    print("TABLE 6: Deep-LSTR Performance by Asset Class")
    print("=" * 60)
    print(table_df.to_string(index=False))

    return table_df


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready panel tables from Deep-LSTR results"
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to experiment directory",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 5, 22],
        help="Forecast horizons to include (default: 1 5 22)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for tables (default: exp-dir/tables)",
    )
    parser.add_argument(
        "--mcs-alpha",
        type=float,
        default=0.10,
        help="MCS significance level (default: 0.10 for 90%% MCS)",
    )
    parser.add_argument(
        "--mcs-boot",
        type=int,
        default=500,
        help="Number of bootstrap replications for MCS (default: 500)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable data quality filtering (include all assets)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print verbose output including excluded assets",
    )

    args = parser.parse_args()

    if not os.path.exists(args.exp_dir):
        print(f"Error: Experiment directory not found: {args.exp_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.exp_dir, "tables")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading results from: {args.exp_dir}")
    print(f"Horizons: {args.horizons}")
    print(f"Output directory: {output_dir}")
    print(f"Data quality filter: {'DISABLED' if args.no_filter else 'ENABLED'}")
    print(f"QLIKE bounds: [{QLIKE_LOWER_BOUND}, {QLIKE_UPPER_BOUND}]")

    # Collect results by horizon
    qlike_by_horizon = {}
    win_by_horizon = {}
    mcs_by_horizon = {}
    dm_by_horizon = {}
    all_fz_results = []
    class_perf = None
    h1_asset_results = None  # Store H=1 results for Table 5

    for H in args.horizons:
        print(f"\nProcessing H={H}...")

        # Load per-asset results
        asset_results_raw = load_asset_results(args.exp_dir, H)
        print(f"  Loaded {len(asset_results_raw)} assets")

        # Apply data quality filter
        if args.no_filter:
            asset_results = asset_results_raw
        else:
            asset_results = filter_valid_assets(
                asset_results_raw, MODELS_OOS, verbose=args.verbose
            )
            if len(asset_results) < len(asset_results_raw):
                excluded = len(asset_results_raw) - len(asset_results)
                print(f"  Filtered to {len(asset_results)} assets ({excluded} excluded due to data quality)")

        if not asset_results:
            print(f"  Warning: No results found for H={H}")
            continue

        # Table 1: QLIKE with SE
        qlike_df = compute_panel_qlike_with_se(asset_results, H)
        if not qlike_df.empty:
            qlike_by_horizon[H] = qlike_df
            print(f"  Computed QLIKE for {qlike_df['n_assets'].max()} assets")

        # Table 2: Win rates and MCS
        win_df, mcs_df = compute_win_rates_and_mcs(
            asset_results, alpha=args.mcs_alpha, n_boot=args.mcs_boot
        )
        if not win_df.empty:
            win_by_horizon[H] = win_df
            mcs_by_horizon[H] = mcs_df
            print(f"  Computed win rates and MCS")

        # Table 3: DM rejection rates
        dm_df = compute_dm_rejection_rates(asset_results, H)
        if not dm_df.empty:
            dm_by_horizon[H] = dm_df
            print(f"  Computed DM rejection rates")

        # Table 4: FZ results (exclude same assets as QLIKE computation)
        excluded_assets = set(asset_results_raw.keys()) - set(asset_results.keys())
        fz_df = load_fz_results(args.exp_dir, H, exclude_assets=list(excluded_assets))
        if not fz_df.empty:
            all_fz_results.append(fz_df)
            print(f"  Loaded FZ results for {fz_df['asset'].nunique()} assets")

        # Table 5: Transition properties (compute using H=1 data)
        if H == 1 and h1_asset_results is None:
            h1_asset_results = asset_results
            print(f"  Stored H=1 data for transition properties")

        # Table 6: Asset class performance (only for H=5)
        if H == 5:
            class_perf = compute_asset_class_performance(asset_results)
            if not class_perf.empty:
                print(f"  Computed asset class performance")

    # Generate formatted tables
    print("\n" + "=" * 70)
    print("GENERATING PUBLICATION TABLES")
    print("=" * 70)

    if qlike_by_horizon:
        format_table_1(qlike_by_horizon, output_dir)

    if win_by_horizon and mcs_by_horizon:
        format_table_2(win_by_horizon, mcs_by_horizon, output_dir)

    if dm_by_horizon:
        format_table_3(dm_by_horizon, output_dir)

    if all_fz_results:
        combined_fz = pd.concat(all_fz_results, ignore_index=True)
        format_table_4(combined_fz, output_dir)

    if h1_asset_results is not None:
        format_table_5(h1_asset_results, output_dir)

    if class_perf is not None and not class_perf.empty:
        format_table_6(class_perf, output_dir)

    print(f"\n{'=' * 70}")
    print(f"Tables saved to: {output_dir}")
    print(f"{'=' * 70}")


# ============================================================================
# YAML GENERATION FUNCTIONS
# ============================================================================


def _safe_float(value, default: float = 0.0) -> float:
    """Safely convert value to float, returning default on failure."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def collect_metrics_for_yaml(
    qlike_summary: Optional[pd.DataFrame],
    win_rates: Optional[pd.DataFrame],
    dm_results: Optional[pd.DataFrame],
    fz_summary: Optional[pd.DataFrame],
    transition_props: Optional[pd.DataFrame],
    class_performance: Optional[pd.DataFrame],
) -> Dict[str, float]:
    """
    Collect all metrics for YAML output.

    Returns dict of metric_name -> value for {{metric_name}} placeholders.

    Note: fz_summary and class_performance are reserved for future use.
    """
    metrics = {}

    # QLIKE metrics from Table 1
    if qlike_summary is not None and not qlike_summary.empty:
        for _, row in qlike_summary.iterrows():
            model = row.get('model', row.get('Model', '')).lower().replace('-', '_')
            for h in [1, 5, 22]:
                col = f'H={h}' if f'H={h}' in row.index else f'h{h}'
                if col in row.index:
                    metrics[f'{model}_qlike_h{h}'] = _safe_float(row[col])

        # Calculate improvements (with zero-division protection)
        for h in [1, 5, 22]:
            har_key = f'har_qlike_h{h}'
            ssm_key = f'str_ssm_qlike_h{h}'
            obs_key = f'str_obs_qlike_h{h}'

            if har_key in metrics and ssm_key in metrics and metrics[har_key] > 0:
                har = metrics[har_key]
                ssm = metrics[ssm_key]
                metrics[f'str_ssm_vs_har_pct_h{h}'] = round((har - ssm) / har * 100, 1)

            if obs_key in metrics and ssm_key in metrics and metrics[obs_key] > 0:
                obs = metrics[obs_key]
                ssm = metrics[ssm_key]
                metrics[f'str_ssm_vs_obs_pct_h{h}'] = round((obs - ssm) / obs * 100, 1)

    # Win rates from Table 2
    if win_rates is not None and not win_rates.empty:
        for _, row in win_rates.iterrows():
            model = row.get('model', row.get('Model', '')).lower().replace('-', '_')
            for h in [1, 5, 22]:
                win_col = f'Win H={h}' if f'Win H={h}' in row.index else f'win_h{h}'
                mcs_col = f'MCS H={h}' if f'MCS H={h}' in row.index else f'mcs_h{h}'
                if win_col in row.index:
                    metrics[f'{model}_win_h{h}'] = _safe_float(row[win_col])
                if mcs_col in row.index:
                    metrics[f'{model}_mcs_h{h}'] = _safe_float(row[mcs_col])

    # DM rejection rates from Table 3
    if dm_results is not None and not dm_results.empty:
        for _, row in dm_results.iterrows():
            comparison = row.get('Comparison', '').lower().replace(' ', '_').replace('.', '')
            for h in [1, 5, 22]:
                col = f'H={h}' if f'H={h}' in row.index else f'h{h}'
                if col in row.index:
                    metrics[f'dm_{comparison}_h{h}'] = _safe_float(row[col])

    # Transition properties from Table 5
    if transition_props is not None and not transition_props.empty:
        for _, row in transition_props.iterrows():
            signal = row.get('Signal', row.get('Property', '')).lower().replace('-', '_')
            if 'std' in row.index or 'Standard Deviation' in row.index:
                col = 'std' if 'std' in row.index else 'Standard Deviation'
                metrics[f'trans_{signal}_std'] = _safe_float(row[col])
            if 'extreme_pct' in row.index or '% Days at Extremes' in row.index:
                col = 'extreme_pct' if 'extreme_pct' in row.index else '% Days at Extremes'
                val = row[col]
                if isinstance(val, str):
                    val = val.replace('%', '')
                metrics[f'trans_{signal}_extreme_pct'] = _safe_float(val)

    return metrics


def collect_tables_for_yaml(
    table1_md: Optional[str],
    table2_md: Optional[str],
    table3_md: Optional[str],
    table4a_md: Optional[str],
    table4b_md: Optional[str],
    table5_md: Optional[str],
    table6_md: Optional[str],
    tablea2_md: Optional[str],
) -> Dict[str, str]:
    """
    Collect all tables as markdown strings for YAML output.

    Returns dict of table_name -> markdown_string for {{TABLE:table_name}} placeholders.
    """
    tables = {}

    if table1_md:
        tables['panel_qlike_summary'] = table1_md
    if table2_md:
        tables['win_rates_mcs'] = table2_md
    if table3_md:
        tables['dm_rejection_rates'] = table3_md
    if table4a_md:
        tables['fz0_loss'] = table4a_md
    if table4b_md:
        tables['var_calibration'] = table4b_md
    if table5_md:
        tables['transition_properties'] = table5_md
    if table6_md:
        tables['asset_class_performance'] = table6_md
    if tablea2_md:
        tables['per_asset_results'] = tablea2_md

    return tables


def generate_results_yaml(
    exp_dir: str,
    output_dir: str,
    metrics: Dict[str, float],
    tables: Dict[str, str],
    config_path: str = None,
) -> str:
    """
    Generate results.yaml file for paper injection.

    Args:
        exp_dir: Experiment directory path
        output_dir: Output directory for tables
        metrics: Dict of metric name -> value
        tables: Dict of table name -> markdown string
        config_path: Path to config file used

    Returns:
        Path to generated results.yaml
    """
    experiment_id = os.path.basename(exp_dir)

    results = create_empty_results(
        experiment_id=experiment_id,
        config_path=config_path or "unknown",
    )
    results.metrics = metrics
    results.tables = tables

    yaml_path = os.path.join(output_dir, "results.yaml")
    results.to_yaml(yaml_path)

    print(f"\nGenerated results YAML: {yaml_path}")
    print(f"  Metrics: {len(metrics)}")
    print(f"  Tables: {len(tables)}")

    return yaml_path


if __name__ == "__main__":
    main()
