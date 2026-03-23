"""
Predictive regression analysis for regime-stress relationships.

Tests whether latent volatility regimes predict external stress indicators
beyond what is already captured by realized volatility levels. Uses
Newey-West HAC standard errors to handle serial correlation from
overlapping horizons.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Dict, List, Optional, Tuple, Union


# Minimum observations required to run a regression
MIN_OBS = 60


def predictive_regression(
    y: np.ndarray,
    X: np.ndarray,
    newey_west_lags: int = 22,
    variable_names: Optional[List[str]] = None,
) -> Dict:
    """
    OLS regression with Newey-West HAC standard errors.

    Adds a constant to the regressor matrix automatically.

    Args:
        y: Dependent variable array (T,)
        X: Regressor matrix (T, k) without constant
        newey_west_lags: Truncation lag for Newey-West kernel
        variable_names: Optional names for the X columns (excluding constant)

    Returns:
        Dict with keys: coefficients, se, t_stats, p_values, R2, adj_R2,
        nobs, variable_names
    """
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Drop rows with any NaN
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y_clean = y[mask]
    X_clean = X[mask]

    if len(y_clean) < MIN_OBS:
        return _empty_regression_result(X.shape[1], variable_names)

    X_const = sm.add_constant(X_clean)

    model = sm.OLS(y_clean, X_const)
    results = model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": newey_west_lags},
    )

    k = X_clean.shape[1]
    if variable_names is None:
        variable_names = [f"x{i}" for i in range(k)]

    all_names = ["const"] + list(variable_names)

    return {
        "coefficients": results.params,
        "se": results.bse,
        "t_stats": results.tvalues,
        "p_values": results.pvalues,
        "R2": results.rsquared,
        "adj_R2": results.rsquared_adj,
        "nobs": int(results.nobs),
        "variable_names": all_names,
    }


def incremental_r2(
    y: np.ndarray,
    X_base: np.ndarray,
    X_full: np.ndarray,
    newey_west_lags: int = 22,
) -> Dict:
    """
    Compute incremental R-squared from adding variables to a base model.

    Tests joint significance of the added variables via an F-test
    (Wald test with HAC covariance).

    Args:
        y: Dependent variable array (T,)
        X_base: Base regressor matrix (T, k_base) without constant
        X_full: Full regressor matrix (T, k_full) without constant,
            where k_full > k_base and X_full[:, :k_base] == X_base
        newey_west_lags: Truncation lag for Newey-West kernel

    Returns:
        Dict with keys: r2_base, r2_full, incremental_r2, f_stat, f_pvalue
    """
    y = np.asarray(y, dtype=np.float64)
    X_base = np.asarray(X_base, dtype=np.float64)
    X_full = np.asarray(X_full, dtype=np.float64)

    if X_base.ndim == 1:
        X_base = X_base.reshape(-1, 1)
    if X_full.ndim == 1:
        X_full = X_full.reshape(-1, 1)

    # Drop rows with any NaN across all arrays
    mask = (
        np.isfinite(y)
        & np.all(np.isfinite(X_base), axis=1)
        & np.all(np.isfinite(X_full), axis=1)
    )
    y_c = y[mask]
    X_base_c = X_base[mask]
    X_full_c = X_full[mask]

    if len(y_c) < MIN_OBS:
        return {
            "r2_base": np.nan,
            "r2_full": np.nan,
            "incremental_r2": np.nan,
            "f_stat": np.nan,
            "f_pvalue": np.nan,
        }

    # Base model
    X_base_const = sm.add_constant(X_base_c)
    base_model = sm.OLS(y_c, X_base_const)
    base_results = base_model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": newey_west_lags},
    )

    # Full model
    X_full_const = sm.add_constant(X_full_c)
    full_model = sm.OLS(y_c, X_full_const)
    full_results = full_model.fit(
        cov_type="HAC",
        cov_kwds={"maxlags": newey_west_lags},
    )

    r2_base = base_results.rsquared
    r2_full = full_results.rsquared
    delta_r2 = r2_full - r2_base

    # Wald test for joint significance of added variables
    # The added variables are the last (k_full - k_base) columns of X_full_const
    k_base_with_const = X_base_const.shape[1]
    k_full_with_const = X_full_const.shape[1]
    n_added = k_full_with_const - k_base_with_const

    if n_added > 0:
        # Build restriction matrix: test that added coefficients = 0
        R = np.zeros((n_added, k_full_with_const))
        for i in range(n_added):
            R[i, k_base_with_const + i] = 1.0

        try:
            wald = full_results.wald_test(R, use_f=True, scalar=True)
            f_stat = float(wald.statistic)
            f_pvalue = float(wald.pvalue)
        except Exception:
            # Fall back to standard F-test approximation
            n = len(y_c)
            f_stat = (delta_r2 / n_added) / ((1 - r2_full) / (n - k_full_with_const))
            from scipy.stats import f as f_dist
            f_pvalue = 1.0 - f_dist.cdf(f_stat, n_added, n - k_full_with_const)
    else:
        f_stat = np.nan
        f_pvalue = np.nan

    return {
        "r2_base": r2_base,
        "r2_full": r2_full,
        "incremental_r2": delta_r2,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
    }


def regime_predicts_stress(
    regime_df: pd.DataFrame,
    external_df: pd.DataFrame,
    outcome_var: str,
    regime_type: str = "G_ssm",
    controls: Optional[List[str]] = None,
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Test whether regime indicator predicts future stress outcomes.

    For each horizon h, regresses external_var(t+h) on G(t) + controls(t).
    Controls default to rv_level (current log RV) and lagged_outcome (AR1 term).

    Args:
        regime_df: DataFrame with DatetimeIndex or 'date' column, containing
            regime columns ('G_obs', 'G_ssm') and 'y' (log realized volatility)
        external_df: DataFrame with DatetimeIndex or 'date' column, containing
            the outcome variable
        outcome_var: Column name of the stress variable in external_df
        regime_type: Which regime column to use ('G_ssm' or 'G_obs')
        controls: List of control variable names. If None, defaults to
            ['rv_level', 'lagged_outcome']. These are constructed internally.
        horizons: List of forecast horizons in trading days. Default [1, 5, 22].

    Returns:
        DataFrame with rows per horizon, columns: horizon, coef_regime,
        se_regime, t_regime, p_regime, R2, incremental_R2
    """
    if horizons is None:
        horizons = [1, 5, 22]
    if controls is None:
        controls = ["rv_level", "lagged_outcome"]

    # Align data on dates
    merged = _align_regime_external(regime_df, external_df, outcome_var)
    if merged is None or len(merged) < MIN_OBS:
        return _empty_stress_result(horizons)

    rows = []
    for h in horizons:
        nw_lags = max(20, 2 * h)

        # Forward-shift the outcome variable by h days
        merged[f"y_ahead_{h}"] = merged[outcome_var].shift(-h)

        # Build control variables at time t
        control_arrays = []
        control_names = []

        if "rv_level" in controls:
            control_arrays.append(merged["y"].values)
            control_names.append("rv_level")

        if "lagged_outcome" in controls:
            control_arrays.append(merged[outcome_var].values)
            control_names.append("lagged_outcome")

        # Regime variable at time t
        G = merged[regime_type].values
        y_ahead = merged[f"y_ahead_{h}"].values

        # Base model: controls only
        if len(control_arrays) > 0:
            X_base = np.column_stack(control_arrays)
        else:
            X_base = np.empty((len(G), 0))

        # Full model: controls + regime
        X_full = np.column_stack(control_arrays + [G]) if len(control_arrays) > 0 else G.reshape(-1, 1)

        # Run base regression
        if X_base.shape[1] > 0:
            base_result = predictive_regression(
                y_ahead, X_base, newey_west_lags=nw_lags,
                variable_names=control_names,
            )
        else:
            base_result = {"R2": 0.0}

        # Run full regression
        full_result = predictive_regression(
            y_ahead, X_full, newey_west_lags=nw_lags,
            variable_names=control_names + [regime_type],
        )

        # Incremental R2
        if X_base.shape[1] > 0:
            incr = incremental_r2(
                y_ahead, X_base, X_full, newey_west_lags=nw_lags,
            )
            incr_r2 = incr["incremental_r2"]
        else:
            incr_r2 = full_result["R2"]

        # Extract regime coefficient (last non-constant variable)
        if full_result["nobs"] > 0:
            # Regime is the last variable; in the coefficient vector,
            # const is index 0, so regime is at index -1
            regime_idx = len(full_result["coefficients"]) - 1
            rows.append({
                "horizon": h,
                "coef_regime": full_result["coefficients"][regime_idx],
                "se_regime": full_result["se"][regime_idx],
                "t_regime": full_result["t_stats"][regime_idx],
                "p_regime": full_result["p_values"][regime_idx],
                "R2": full_result["R2"],
                "incremental_R2": incr_r2,
                "nobs": full_result["nobs"],
            })
        else:
            rows.append({
                "horizon": h,
                "coef_regime": np.nan,
                "se_regime": np.nan,
                "t_regime": np.nan,
                "p_regime": np.nan,
                "R2": np.nan,
                "incremental_R2": np.nan,
                "nobs": 0,
            })

        # Clean up temporary column
        merged.drop(columns=[f"y_ahead_{h}"], inplace=True)

    return pd.DataFrame(rows)


def panel_predictive_regression(
    regime_dict: Dict[str, pd.DataFrame],
    external_df: pd.DataFrame,
    outcome_var: str,
    regime_type: str = "G_ssm",
    horizons: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Run predictive regressions per asset and summarize across the panel.

    Args:
        regime_dict: Dict mapping ticker -> DataFrame with regime columns
        external_df: DataFrame with external stress variables
        outcome_var: Column name of the stress variable
        regime_type: Which regime column ('G_ssm' or 'G_obs')
        horizons: List of forecast horizons. Default [1, 5, 22].

    Returns:
        DataFrame with rows per horizon, columns: horizon, mean_coef,
        frac_significant, frac_positive, mean_R2, mean_incremental_R2,
        n_assets
    """
    if horizons is None:
        horizons = [1, 5, 22]

    # Collect per-asset results
    asset_results = {}
    for ticker, regime_df in regime_dict.items():
        result = regime_predicts_stress(
            regime_df,
            external_df,
            outcome_var=outcome_var,
            regime_type=regime_type,
            horizons=horizons,
        )
        if result is not None and len(result) > 0:
            asset_results[ticker] = result

    if not asset_results:
        return _empty_panel_result(horizons)

    panel_rows = []
    for h in horizons:
        coefs = []
        p_values = []
        r2_values = []
        incr_r2_values = []

        for ticker, result in asset_results.items():
            h_row = result[result["horizon"] == h]
            if len(h_row) == 0:
                continue
            h_row = h_row.iloc[0]

            if np.isfinite(h_row["coef_regime"]):
                coefs.append(h_row["coef_regime"])
            if np.isfinite(h_row["p_regime"]):
                p_values.append(h_row["p_regime"])
            if np.isfinite(h_row["R2"]):
                r2_values.append(h_row["R2"])
            if np.isfinite(h_row["incremental_R2"]):
                incr_r2_values.append(h_row["incremental_R2"])

        n_assets = len(coefs)
        panel_rows.append({
            "horizon": h,
            "mean_coef": np.mean(coefs) if coefs else np.nan,
            "frac_significant": (
                np.mean([p < 0.05 for p in p_values]) if p_values else np.nan
            ),
            "frac_positive": (
                np.mean([c > 0 for c in coefs]) if coefs else np.nan
            ),
            "mean_R2": np.mean(r2_values) if r2_values else np.nan,
            "mean_incremental_R2": (
                np.mean(incr_r2_values) if incr_r2_values else np.nan
            ),
            "n_assets": n_assets,
        })

    return pd.DataFrame(panel_rows)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _align_regime_external(
    regime_df: pd.DataFrame,
    external_df: pd.DataFrame,
    outcome_var: str,
) -> Optional[pd.DataFrame]:
    """
    Inner-join regime and external data on date index.

    Args:
        regime_df: Regime DataFrame
        external_df: External stress DataFrame
        outcome_var: Column to verify exists in external_df

    Returns:
        Merged DataFrame with DatetimeIndex, or None if insufficient overlap
    """
    if outcome_var not in external_df.columns:
        return None

    # Normalize to DatetimeIndex
    r = regime_df.copy()
    e = external_df.copy()

    if "date" in r.columns:
        r = r.set_index("date")
    if "date" in e.columns:
        e = e.set_index("date")

    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index)
    if not isinstance(e.index, pd.DatetimeIndex):
        e.index = pd.to_datetime(e.index)

    # Keep only needed columns from regime data
    regime_cols = [c for c in ["G_obs", "G_ssm", "y", "rH"] if c in r.columns]
    r = r[regime_cols]

    # Keep outcome variable (and any extra columns) from external
    e = e[[outcome_var]]

    merged = r.join(e, how="inner")

    if len(merged) < MIN_OBS:
        return None

    return merged


def _empty_regression_result(
    n_vars: int,
    variable_names: Optional[List[str]] = None,
) -> Dict:
    """Return a placeholder result when regression cannot be estimated."""
    if variable_names is None:
        variable_names = [f"x{i}" for i in range(n_vars)]
    all_names = ["const"] + list(variable_names)
    k = len(all_names)
    return {
        "coefficients": np.full(k, np.nan),
        "se": np.full(k, np.nan),
        "t_stats": np.full(k, np.nan),
        "p_values": np.full(k, np.nan),
        "R2": np.nan,
        "adj_R2": np.nan,
        "nobs": 0,
        "variable_names": all_names,
    }


def _empty_stress_result(horizons: List[int]) -> pd.DataFrame:
    """Return empty DataFrame when stress regression cannot be run."""
    return pd.DataFrame([
        {
            "horizon": h,
            "coef_regime": np.nan,
            "se_regime": np.nan,
            "t_regime": np.nan,
            "p_regime": np.nan,
            "R2": np.nan,
            "incremental_R2": np.nan,
            "nobs": 0,
        }
        for h in horizons
    ])


def _empty_panel_result(horizons: List[int]) -> pd.DataFrame:
    """Return empty DataFrame when panel regression has no valid assets."""
    return pd.DataFrame([
        {
            "horizon": h,
            "mean_coef": np.nan,
            "frac_significant": np.nan,
            "frac_positive": np.nan,
            "mean_R2": np.nan,
            "mean_incremental_R2": np.nan,
            "n_assets": 0,
        }
        for h in horizons
    ])
