"""
Lead-lag analysis between regime indicators and external stress proxies.

Provides cross-correlation functions with Newey-West standard errors,
Granger causality tests, and panel-level aggregation utilities.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tools.tools import add_constant


def _bartlett_kernel_weight(j: int, bandwidth: int) -> float:
    """Bartlett kernel weight for lag j given bandwidth."""
    if abs(j) >= bandwidth:
        return 0.0
    return 1.0 - abs(j) / bandwidth


def _newey_west_se_correlation(x: np.ndarray, y: np.ndarray,
                                rho: float, bandwidth: int) -> float:
    """
    Compute Newey-West standard error for a sample correlation coefficient.

    Uses the delta method: SE(rho) = sqrt(V / T) where V is the HAC
    variance of the product z_t = (x_t - xbar)(y_t - ybar).

    Parameters
    ----------
    x : array-like
        First series (demeaned not required, handled internally).
    y : array-like
        Second series.
    rho : float
        Sample correlation between x and y.
    bandwidth : int
        Bartlett kernel bandwidth for HAC estimation.

    Returns
    -------
    float
        Newey-West standard error of the correlation.
    """
    T = len(x)
    if T < 3:
        return np.nan

    x_dm = x - np.mean(x)
    y_dm = y - np.mean(y)
    sx = np.std(x, ddof=1)
    sy = np.std(y, ddof=1)
    if sx < 1e-15 or sy < 1e-15:
        return np.nan

    # Standardized product: under H0 this has mean rho
    z = (x_dm * y_dm) / (sx * sy)
    z_dm = z - np.mean(z)

    # HAC variance with Bartlett kernel
    gamma_0 = np.mean(z_dm ** 2)
    hac_var = gamma_0
    for j in range(1, bandwidth):
        if j >= T:
            break
        gamma_j = np.mean(z_dm[j:] * z_dm[:-j])
        w = _bartlett_kernel_weight(j, bandwidth)
        hac_var += 2 * w * gamma_j

    se = np.sqrt(max(hac_var, 0) / T)
    return se


def cross_correlation(regime_series: pd.Series, external_series: pd.Series,
                      max_lag: int = 20) -> pd.DataFrame:
    """
    Compute cross-correlation at each lag k in [-max_lag, max_lag].

    Convention: positive k means regime LEADS external.
        rho(k) = corr(regime[t], external[t + k])

    Parameters
    ----------
    regime_series : pd.Series
        Regime indicator (e.g., G_ssm) with DatetimeIndex.
    external_series : pd.Series
        External stress proxy (e.g., VIX) with DatetimeIndex.
    max_lag : int
        Maximum lag in both directions. Default 20 trading days.

    Returns
    -------
    pd.DataFrame
        Columns: lag, rho, se, ci_lower, ci_upper.
        Rows ordered from -max_lag to +max_lag.
    """
    # Align on common dates
    common = regime_series.dropna().index.intersection(external_series.dropna().index)
    common = common.sort_values()
    r = regime_series.reindex(common).values.astype(np.float64)
    e = external_series.reindex(common).values.astype(np.float64)
    T = len(common)

    bandwidth = max(20, 2 * max_lag)
    results = []

    for k in range(-max_lag, max_lag + 1):
        # corr(regime[t], external[t + k])
        if k >= 0:
            r_slice = r[:T - k] if k > 0 else r
            e_slice = e[k:] if k > 0 else e
        else:
            r_slice = r[-k:]
            e_slice = e[:T + k]

        n = len(r_slice)
        if n < 10:
            results.append({
                'lag': k, 'rho': np.nan, 'se': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan
            })
            continue

        rho = np.corrcoef(r_slice, e_slice)[0, 1]
        se = _newey_west_se_correlation(r_slice, e_slice, rho, bandwidth)
        ci_lower = rho - 1.96 * se
        ci_upper = rho + 1.96 * se

        results.append({
            'lag': k,
            'rho': rho,
            'se': se,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

    return pd.DataFrame(results)


def granger_causality(regime_series: pd.Series, external_series: pd.Series,
                      max_order: int = 5) -> dict:
    """
    Test Granger causality from regime to external variable.

    Runs ADF tests on both series and differences if needed to ensure
    stationarity before running the Granger test.

    Parameters
    ----------
    regime_series : pd.Series
        Regime indicator with DatetimeIndex.
    external_series : pd.Series
        External stress proxy with DatetimeIndex.
    max_order : int
        Maximum lag order to test. Default 5.

    Returns
    -------
    dict
        Keys are lag orders (1, 2, ..., max_order).
        Values are dicts with 'F_stat', 'p_value', 'regime_differenced',
        'external_differenced'.
    """
    # Align series
    common = regime_series.dropna().index.intersection(external_series.dropna().index)
    common = common.sort_values()
    r = regime_series.reindex(common)
    e = external_series.reindex(common)

    # Stationarity checks (ADF at 5% level)
    regime_differenced = False
    external_differenced = False

    adf_r = adfuller(r.values, maxlag=max_order, autolag='AIC')
    if adf_r[1] > 0.05:
        r = r.diff().dropna()
        regime_differenced = True

    adf_e = adfuller(e.values, maxlag=max_order, autolag='AIC')
    if adf_e[1] > 0.05:
        e = e.diff().dropna()
        external_differenced = True

    # Re-align after differencing
    common = r.dropna().index.intersection(e.dropna().index)
    common = common.sort_values()
    r = r.reindex(common)
    e = e.reindex(common)

    if len(common) < max_order + 10:
        return {
            i: {
                'F_stat': np.nan, 'p_value': np.nan,
                'regime_differenced': regime_differenced,
                'external_differenced': external_differenced
            }
            for i in range(1, max_order + 1)
        }

    # Granger test: does regime Granger-cause external?
    # statsmodels expects [y, x] where we test if x (col 1) causes y (col 0)
    data = pd.concat([e.rename('external'), r.rename('regime')], axis=1).dropna()

    try:
        gc_results = grangercausalitytests(
            data[['external', 'regime']].values,
            maxlag=max_order,
            verbose=False
        )
    except Exception:
        return {
            i: {
                'F_stat': np.nan, 'p_value': np.nan,
                'regime_differenced': regime_differenced,
                'external_differenced': external_differenced
            }
            for i in range(1, max_order + 1)
        }

    output = {}
    for lag_order in range(1, max_order + 1):
        test_result = gc_results[lag_order]
        # Extract F-test results (ssr_ftest)
        f_stat = test_result[0]['ssr_ftest'][0]
        p_value = test_result[0]['ssr_ftest'][1]
        output[lag_order] = {
            'F_stat': f_stat,
            'p_value': p_value,
            'regime_differenced': regime_differenced,
            'external_differenced': external_differenced
        }

    return output


def panel_cross_correlation(regime_dict: dict, external_df: pd.DataFrame,
                            variable: str, regime_type: str = 'G_ssm',
                            max_lag: int = 20) -> dict:
    """
    Compute cross-correlation for each asset and aggregate to panel mean.

    Parameters
    ----------
    regime_dict : dict
        Dict of {ticker: DataFrame} where each DataFrame has columns
        including 'G_obs', 'G_ssm' and a DatetimeIndex or 'date' column.
    external_df : pd.DataFrame
        DataFrame with DatetimeIndex and columns for external variables.
    variable : str
        Column name in external_df to correlate against.
    regime_type : str
        'G_ssm' or 'G_obs'. Default 'G_ssm'.
    max_lag : int
        Maximum lag. Default 20.

    Returns
    -------
    dict
        'mean_ccf': pd.DataFrame with mean cross-correlation across assets.
        'individual': dict of {ticker: pd.DataFrame} with per-asset CCFs.
        'n_assets': int, number of assets included.
    """
    if variable not in external_df.columns:
        raise ValueError(f"Variable '{variable}' not found in external_df columns: "
                         f"{list(external_df.columns)}")

    ext = external_df[variable].dropna()
    individual = {}

    for ticker, df in regime_dict.items():
        # Extract regime series
        if isinstance(df.index, pd.DatetimeIndex):
            regime_s = df[regime_type].dropna()
        elif 'date' in df.columns:
            regime_s = df.set_index('date')[regime_type].dropna()
            regime_s.index = pd.to_datetime(regime_s.index)
        else:
            continue

        # Need sufficient overlap
        overlap = regime_s.index.intersection(ext.index)
        if len(overlap) < 50:
            continue

        ccf = cross_correlation(regime_s, ext, max_lag=max_lag)
        individual[ticker] = ccf

    if not individual:
        return {
            'mean_ccf': pd.DataFrame(),
            'individual': {},
            'n_assets': 0
        }

    # Compute mean CCF across assets
    all_rhos = np.column_stack([df['rho'].values for df in individual.values()])
    mean_rho = np.nanmean(all_rhos, axis=1)
    # SE of the mean: std across assets / sqrt(n)
    n_assets = all_rhos.shape[1]
    se_mean = np.nanstd(all_rhos, axis=1, ddof=1) / np.sqrt(n_assets)

    # Use lag values from first result (all should be identical)
    first_ccf = next(iter(individual.values()))
    mean_ccf = pd.DataFrame({
        'lag': first_ccf['lag'].values,
        'rho': mean_rho,
        'se': se_mean,
        'ci_lower': mean_rho - 1.96 * se_mean,
        'ci_upper': mean_rho + 1.96 * se_mean,
    })

    return {
        'mean_ccf': mean_ccf,
        'individual': individual,
        'n_assets': n_assets
    }


def peak_lag(ccf_result: pd.DataFrame) -> tuple:
    """
    Find the lag with maximum absolute correlation.

    Parameters
    ----------
    ccf_result : pd.DataFrame
        Output from cross_correlation() with columns: lag, rho, se, ci_lower, ci_upper.

    Returns
    -------
    tuple
        (lag, rho, p_value) where p_value is from a two-sided z-test
        using the Newey-West SE at that lag.
    """
    if ccf_result.empty or ccf_result['rho'].isna().all():
        return (np.nan, np.nan, np.nan)

    idx = ccf_result['rho'].abs().idxmax()
    row = ccf_result.loc[idx]
    lag = int(row['lag'])
    rho = row['rho']
    se = row['se']

    if np.isnan(se) or se < 1e-15:
        p_value = np.nan
    else:
        z = abs(rho) / se
        p_value = 2 * (1 - stats.norm.cdf(z))

    return (lag, rho, p_value)
