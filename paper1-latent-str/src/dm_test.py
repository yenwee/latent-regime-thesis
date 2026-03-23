"""
Diebold-Mariano test with Newey-West HAC standard errors.
"""

import numpy as np
from scipy.stats import norm

# Numerical stability constant
EPS = 1e-12


def newey_west_var(x, L):
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


def dm_test(loss_a, loss_b, L=5):
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
