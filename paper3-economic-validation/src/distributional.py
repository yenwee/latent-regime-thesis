"""
Distributional comparison of external stress proxies across regime states.

Tests whether high-regime and low-regime periods exhibit significantly
different distributions of held-out financial stress indicators, using
nonparametric tests and block bootstrap inference.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

def classify_regimes(G_series: pd.Series, threshold: float = 0.5) -> pd.Series:
    """
    Classify regime states using a fixed threshold.

    Parameters
    ----------
    G_series : pd.Series
        Regime transition function values in [0, 1].
    threshold : float
        Values >= threshold are classified as high regime. Default 0.5.

    Returns
    -------
    pd.Series
        Boolean Series (True = high regime) with same index as G_series.
    """
    return (G_series >= threshold).rename('high_regime')


def classify_regimes_quantile(G_series: pd.Series,
                               q_high: float = 0.75,
                               q_low: float = 0.25) -> pd.Series:
    """
    Classify regime states using within-sample quantile thresholds.

    Observations in the top quantile (>= q_high) are high regime,
    observations in the bottom quantile (<= q_low) are low regime,
    and observations between the two quantiles are set to NaN (excluded).

    This is more appropriate for G_ssm which is concentrated near 0.5.

    Parameters
    ----------
    G_series : pd.Series
        Regime transition function values.
    q_high : float
        Quantile threshold for high regime (default 0.75).
    q_low : float
        Quantile threshold for low regime (default 0.25).

    Returns
    -------
    pd.Series
        Boolean Series where True = high regime, False = low regime,
        NaN = ambiguous (between quantiles). Same index as G_series.
    """
    val_high = G_series.quantile(q_high)
    val_low = G_series.quantile(q_low)

    result = pd.Series(np.nan, index=G_series.index, name='high_regime',
                       dtype=object)
    result.loc[G_series >= val_high] = True
    result.loc[G_series <= val_low] = False

    return result


# ---------------------------------------------------------------------------
# Block bootstrap utilities
# ---------------------------------------------------------------------------

def _circular_block_bootstrap(data: np.ndarray, block_size: int,
                               n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """
    Circular block bootstrap for time-dependent data.

    Parameters
    ----------
    data : np.ndarray
        1-D array of observations.
    block_size : int
        Size of each contiguous block.
    n_boot : int
        Number of bootstrap replications.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape (n_boot, T) with bootstrapped samples.
    """
    T = len(data)
    n_blocks = int(np.ceil(T / block_size))
    boot_samples = np.empty((n_boot, T))

    for b in range(n_boot):
        # Draw random block starting positions (circular)
        starts = rng.integers(0, T, size=n_blocks)
        sample = np.concatenate([
            np.take(data, range(s, s + block_size), mode='wrap')
            for s in starts
        ])
        boot_samples[b] = sample[:T]

    return boot_samples


# ---------------------------------------------------------------------------
# Conditional statistics
# ---------------------------------------------------------------------------

def conditional_means(external_series: pd.Series, regime_mask: pd.Series,
                      block_size: int = 22, n_boot: int = 1000,
                      seed: int = 42) -> dict:
    """
    Compute conditional means and bootstrap SE for the difference.

    Parameters
    ----------
    external_series : pd.Series
        External stress proxy with DatetimeIndex.
    regime_mask : pd.Series
        Boolean Series (True = high regime). NaN entries are dropped.
    block_size : int
        Block size for circular block bootstrap. Default 22 (1 month).
    n_boot : int
        Number of bootstrap replications. Default 1000.
    seed : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    dict
        mean_high, mean_low, diff, se_diff, ci_lower, ci_upper,
        cohens_d, n_high, n_low.
    """
    # Align and drop NaN
    common = external_series.dropna().index.intersection(regime_mask.dropna().index)
    common = common.sort_values()
    ext = external_series.reindex(common).values.astype(np.float64)
    mask = regime_mask.reindex(common)

    # Drop NaN in mask (from quantile classification)
    valid = mask.notna()
    ext = ext[valid.values]
    mask_bool = mask[valid].astype(bool).values

    n_high = int(mask_bool.sum())
    n_low = int((~mask_bool).sum())

    if n_high < 5 or n_low < 5:
        return {
            'mean_high': np.nan, 'mean_low': np.nan,
            'diff': np.nan, 'se_diff': np.nan,
            'ci_lower': np.nan, 'ci_upper': np.nan,
            'cohens_d': np.nan, 'n_high': n_high, 'n_low': n_low
        }

    mean_high = float(np.mean(ext[mask_bool]))
    mean_low = float(np.mean(ext[~mask_bool]))
    diff = mean_high - mean_low

    # Cohen's d: (mean_high - mean_low) / pooled_std
    var_high = np.var(ext[mask_bool], ddof=1)
    var_low = np.var(ext[~mask_bool], ddof=1)
    pooled_std = np.sqrt(((n_high - 1) * var_high + (n_low - 1) * var_low)
                         / (n_high + n_low - 2))
    cohens_d = diff / pooled_std if pooled_std > 1e-15 else np.nan

    # Block bootstrap for SE of the difference
    rng = np.random.default_rng(seed)
    # Bootstrap pairs (ext, mask) together to preserve temporal structure
    T = len(ext)
    paired = np.column_stack([ext, mask_bool.astype(float)])
    boot_diffs = np.empty(n_boot)

    n_blocks = int(np.ceil(T / block_size))
    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        indices = np.concatenate([
            np.arange(s, s + block_size) % T for s in starts
        ])[:T]
        boot_ext = paired[indices, 0]
        boot_mask = paired[indices, 1].astype(bool)
        bh = boot_ext[boot_mask]
        bl = boot_ext[~boot_mask]
        if len(bh) > 0 and len(bl) > 0:
            boot_diffs[b] = np.mean(bh) - np.mean(bl)
        else:
            boot_diffs[b] = np.nan

    se_diff = float(np.nanstd(boot_diffs, ddof=1))
    ci_lower = float(np.nanpercentile(boot_diffs, 2.5))
    ci_upper = float(np.nanpercentile(boot_diffs, 97.5))

    return {
        'mean_high': mean_high,
        'mean_low': mean_low,
        'diff': diff,
        'se_diff': se_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'cohens_d': cohens_d,
        'n_high': n_high,
        'n_low': n_low
    }


# ---------------------------------------------------------------------------
# Nonparametric tests
# ---------------------------------------------------------------------------

def ks_test_by_regime(external_series: pd.Series,
                      regime_mask: pd.Series) -> dict:
    """
    Two-sample Kolmogorov-Smirnov test across regime states.

    Parameters
    ----------
    external_series : pd.Series
        External stress proxy.
    regime_mask : pd.Series
        Boolean Series (True = high regime). NaN entries are dropped.

    Returns
    -------
    dict
        ks_stat, p_value, n_high, n_low.
    """
    common = external_series.dropna().index.intersection(regime_mask.dropna().index)
    common = common.sort_values()
    ext = external_series.reindex(common)
    mask = regime_mask.reindex(common)

    valid = mask.notna()
    ext = ext[valid]
    mask = mask[valid].astype(bool)

    high = ext[mask].values
    low = ext[~mask].values

    if len(high) < 5 or len(low) < 5:
        return {'ks_stat': np.nan, 'p_value': np.nan,
                'n_high': len(high), 'n_low': len(low)}

    stat, pval = stats.ks_2samp(high, low)
    return {'ks_stat': float(stat), 'p_value': float(pval),
            'n_high': len(high), 'n_low': len(low)}


def mann_whitney_by_regime(external_series: pd.Series,
                           regime_mask: pd.Series) -> dict:
    """
    Mann-Whitney U test across regime states.

    Parameters
    ----------
    external_series : pd.Series
        External stress proxy.
    regime_mask : pd.Series
        Boolean Series (True = high regime). NaN entries are dropped.

    Returns
    -------
    dict
        mw_stat, p_value, n_high, n_low.
    """
    common = external_series.dropna().index.intersection(regime_mask.dropna().index)
    common = common.sort_values()
    ext = external_series.reindex(common)
    mask = regime_mask.reindex(common)

    valid = mask.notna()
    ext = ext[valid]
    mask = mask[valid].astype(bool)

    high = ext[mask].values
    low = ext[~mask].values

    if len(high) < 5 or len(low) < 5:
        return {'mw_stat': np.nan, 'p_value': np.nan,
                'n_high': len(high), 'n_low': len(low)}

    stat, pval = stats.mannwhitneyu(high, low, alternative='two-sided')
    return {'mw_stat': float(stat), 'p_value': float(pval),
            'n_high': len(high), 'n_low': len(low)}


def levene_by_regime(external_series: pd.Series,
                     regime_mask: pd.Series) -> dict:
    """
    Levene's test for equality of variances across regime states.

    Parameters
    ----------
    external_series : pd.Series
        External stress proxy.
    regime_mask : pd.Series
        Boolean Series (True = high regime). NaN entries are dropped.

    Returns
    -------
    dict
        levene_stat, p_value, n_high, n_low.
    """
    common = external_series.dropna().index.intersection(regime_mask.dropna().index)
    common = common.sort_values()
    ext = external_series.reindex(common)
    mask = regime_mask.reindex(common)

    valid = mask.notna()
    ext = ext[valid]
    mask = mask[valid].astype(bool)

    high = ext[mask].values
    low = ext[~mask].values

    if len(high) < 5 or len(low) < 5:
        return {'levene_stat': np.nan, 'p_value': np.nan,
                'n_high': len(high), 'n_low': len(low)}

    stat, pval = stats.levene(high, low)
    return {'levene_stat': float(stat), 'p_value': float(pval),
            'n_high': len(high), 'n_low': len(low)}


def variance_ratio(external_series: pd.Series, regime_mask: pd.Series,
                   n_boot: int = 1000, block_size: int = 22,
                   seed: int = 42) -> dict:
    """
    Compute Var(high) / Var(low) with bootstrap confidence interval.

    Parameters
    ----------
    external_series : pd.Series
        External stress proxy.
    regime_mask : pd.Series
        Boolean Series (True = high regime). NaN entries are dropped.
    n_boot : int
        Number of bootstrap replications. Default 1000.
    block_size : int
        Block size for circular block bootstrap. Default 22.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    dict
        var_ratio, ci_lower, ci_upper, var_high, var_low, n_high, n_low.
    """
    common = external_series.dropna().index.intersection(regime_mask.dropna().index)
    common = common.sort_values()
    ext = external_series.reindex(common)
    mask = regime_mask.reindex(common)

    valid = mask.notna()
    ext_vals = ext[valid].values.astype(np.float64)
    mask_vals = mask[valid].astype(bool).values

    n_high = int(mask_vals.sum())
    n_low = int((~mask_vals).sum())

    if n_high < 5 or n_low < 5:
        return {
            'var_ratio': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan,
            'var_high': np.nan, 'var_low': np.nan,
            'n_high': n_high, 'n_low': n_low
        }

    var_high = float(np.var(ext_vals[mask_vals], ddof=1))
    var_low = float(np.var(ext_vals[~mask_vals], ddof=1))
    vr = var_high / var_low if var_low > 1e-15 else np.nan

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    T = len(ext_vals)
    paired = np.column_stack([ext_vals, mask_vals.astype(float)])
    boot_vr = np.empty(n_boot)
    n_blocks = int(np.ceil(T / block_size))

    for b in range(n_boot):
        starts = rng.integers(0, T, size=n_blocks)
        indices = np.concatenate([
            np.arange(s, s + block_size) % T for s in starts
        ])[:T]
        b_ext = paired[indices, 0]
        b_mask = paired[indices, 1].astype(bool)
        bh = b_ext[b_mask]
        bl = b_ext[~b_mask]
        if len(bh) > 2 and len(bl) > 2:
            vh = np.var(bh, ddof=1)
            vl = np.var(bl, ddof=1)
            boot_vr[b] = vh / vl if vl > 1e-15 else np.nan
        else:
            boot_vr[b] = np.nan

    ci_lower = float(np.nanpercentile(boot_vr, 2.5))
    ci_upper = float(np.nanpercentile(boot_vr, 97.5))

    return {
        'var_ratio': vr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'var_high': var_high,
        'var_low': var_low,
        'n_high': n_high,
        'n_low': n_low
    }


# ---------------------------------------------------------------------------
# Full distributional comparison
# ---------------------------------------------------------------------------

def full_distributional_comparison(external_df: pd.DataFrame,
                                   G_series: pd.Series,
                                   variables: list,
                                   threshold: float = 0.5,
                                   use_quantile: bool = False,
                                   q_high: float = 0.75,
                                   q_low: float = 0.25,
                                   block_size: int = 22,
                                   n_boot: int = 1000,
                                   seed: int = 42) -> pd.DataFrame:
    """
    Run all distributional tests for all variables.

    Parameters
    ----------
    external_df : pd.DataFrame
        DataFrame with DatetimeIndex and columns for external variables.
    G_series : pd.Series
        Regime transition function values.
    variables : list
        Column names in external_df to test.
    threshold : float
        Fixed threshold for classify_regimes. Default 0.5.
    use_quantile : bool
        If True, use quantile-based classification instead of fixed threshold.
    q_high : float
        Upper quantile for quantile classification. Default 0.75.
    q_low : float
        Lower quantile for quantile classification. Default 0.25.
    block_size : int
        Block size for bootstrap. Default 22.
    n_boot : int
        Bootstrap replications. Default 1000.
    seed : int
        Random seed. Default 42.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: variable, mean_high, mean_low, diff, se,
        cohens_d, ks_stat, ks_p, mw_stat, mw_p, levene_stat, levene_p,
        var_ratio.
    """
    if use_quantile:
        regime_mask = classify_regimes_quantile(G_series, q_high=q_high, q_low=q_low)
    else:
        regime_mask = classify_regimes(G_series, threshold=threshold)

    rows = []
    rng_base = np.random.default_rng(seed)

    for i, var in enumerate(variables):
        if var not in external_df.columns:
            continue

        ext = external_df[var]
        # Use a different seed per variable for independence, but deterministic
        var_seed = seed + i

        cm = conditional_means(ext, regime_mask, block_size=block_size,
                               n_boot=n_boot, seed=var_seed)
        ks = ks_test_by_regime(ext, regime_mask)
        mw = mann_whitney_by_regime(ext, regime_mask)
        lv = levene_by_regime(ext, regime_mask)
        vr = variance_ratio(ext, regime_mask, n_boot=n_boot,
                            block_size=block_size, seed=var_seed)

        rows.append({
            'variable': var,
            'mean_high': cm['mean_high'],
            'mean_low': cm['mean_low'],
            'diff': cm['diff'],
            'se': cm['se_diff'],
            'cohens_d': cm['cohens_d'],
            'ks_stat': ks['ks_stat'],
            'ks_p': ks['p_value'],
            'mw_stat': mw['mw_stat'],
            'mw_p': mw['p_value'],
            'levene_stat': lv['levene_stat'],
            'levene_p': lv['p_value'],
            'var_ratio': vr['var_ratio'],
            'n_high': cm['n_high'],
            'n_low': cm['n_low'],
        })

    return pd.DataFrame(rows)
