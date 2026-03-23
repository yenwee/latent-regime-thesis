# src/smoothers.py
"""
Alternative transition variable smoothers for STR-HAR ablation study.

Implements:
- EWMA (Exponentially Weighted Moving Average)
- Rolling Mean
- Kalman Local Level Model

These serve as baselines to demonstrate that Deep-LSTR's advantage
comes from nonlinear latent dynamics, not just smoothing.

All inner loops are JIT-compiled with Numba for performance.

References:
- RiskMetrics (1996) for EWMA
- Durbin & Koopman (2012) for Kalman local level
"""

import numpy as np
from numba import njit
from typing import Tuple
from scipy.optimize import minimize


# ============================================================
# Numba-compiled kernels
# ============================================================

@njit(cache=True)
def _ewma_kernel(x, lam):
    """EWMA recursion: q_t = lam * q_{t-1} + (1 - lam) * x_t."""
    T = len(x)
    q = np.empty(T)
    q[0] = x[0]
    one_m_lam = 1.0 - lam
    for t in range(1, T):
        q[t] = lam * q[t - 1] + one_m_lam * x[t]
    return q


@njit(cache=True)
def _rolling_mean_kernel(x, window):
    """Rolling mean with expanding window for initial observations."""
    T = len(x)
    q = np.empty(T)
    cum = 0.0
    for t in range(T):
        cum += x[t]
        start = t - window + 1
        if start > 0:
            cum -= x[start - 1]
            q[t] = cum / window
        else:
            q[t] = cum / (t + 1)
    return q


@njit(cache=True)
def _kalman_nll(x, log_var_obs, log_var_state):
    """Kalman filter negative log-likelihood for local level model."""
    var_obs = np.exp(log_var_obs)
    var_state = np.exp(log_var_state)
    T = len(x)
    mu = x[0]
    P = var_obs + var_state
    nll = 0.0
    for t in range(1, T):
        P_pred = P + var_state
        v = x[t] - mu
        F = P_pred + var_obs
        nll += 0.5 * (np.log(F) + v * v / F)
        K = P_pred / F
        mu = mu + K * v
        P = (1.0 - K) * P_pred
    return nll


@njit(cache=True)
def _kalman_filter_smooth(x, var_obs, var_state):
    """Kalman filter + RTS smoother for local level model."""
    T = len(x)

    # Forward pass: Kalman filter
    mu_filt = np.empty(T)
    P_filt = np.empty(T)
    mu_filt[0] = x[0]
    P_filt[0] = var_obs + var_state

    for t in range(1, T):
        P_pred = P_filt[t - 1] + var_state
        K = P_pred / (P_pred + var_obs)
        mu_filt[t] = mu_filt[t - 1] + K * (x[t] - mu_filt[t - 1])
        P_filt[t] = (1.0 - K) * P_pred

    # Backward pass: RTS smoother
    mu_smooth = np.empty(T)
    mu_smooth[T - 1] = mu_filt[T - 1]

    for t in range(T - 2, -1, -1):
        P_pred = P_filt[t] + var_state
        J = P_filt[t] / P_pred
        mu_smooth[t] = mu_filt[t] + J * (mu_smooth[t + 1] - mu_filt[t])

    return mu_smooth


# ============================================================
# Public API
# ============================================================

def ewma_smoother(x: np.ndarray, lam: float = 0.94) -> np.ndarray:
    """
    Exponentially Weighted Moving Average smoother.

    EWMA: q_t = lambda * q_{t-1} + (1 - lambda) * x_t

    Standard in RiskMetrics volatility smoothing. Lambda controls
    decay rate: 0.90 (fast), 0.94 (medium), 0.97 (slow).

    Args:
        x: Input series (e.g., log realized variance)
        lam: Decay parameter in (0, 1). Higher = slower adaptation.

    Returns:
        Smoothed series of same length as x

    References:
        RiskMetrics Technical Document, 4th ed. (1996)
    """
    if not 0 < lam < 1:
        raise ValueError(f"Lambda must be in (0, 1), got {lam}")
    return _ewma_kernel(np.asarray(x, dtype=np.float64), lam)


def rolling_mean_smoother(x: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Simple rolling mean smoother.

    Uses expanding window for initial observations where full window
    is not available.

    Args:
        x: Input series
        window: Rolling window size (20 = monthly, 60 = quarterly)

    Returns:
        Smoothed series of same length as x
    """
    if window < 1:
        raise ValueError(f"Window must be >= 1, got {window}")
    return _rolling_mean_kernel(np.asarray(x, dtype=np.float64), window)


def kalman_local_level_smoother(
    x: np.ndarray,
    var_obs: float = None,
    var_state: float = None,
) -> Tuple[np.ndarray, float, float]:
    """
    Kalman filter/smoother for local level model.

    State-space model:
        Observation: x_t = mu_t + eps_t,  eps_t ~ N(0, var_obs)
        State:       mu_t = mu_{t-1} + eta_t,  eta_t ~ N(0, var_state)

    If variances not provided, estimates them via MLE.
    Returns smoothed state estimates (Rauch-Tung-Striebel smoother).

    This is the "best-in-class" linear Gaussian smoother. If Deep-LSTR
    beats Kalman-smoothed transition, the advantage is clearly nonlinear.

    Args:
        x: Observation series (e.g., log realized variance)
        var_obs: Observation noise variance (estimated if None)
        var_state: State transition variance (estimated if None)

    Returns:
        Tuple of (smoothed_state, var_obs, var_state)

    References:
        Durbin, J. and Koopman, S.J. (2012). Time Series Analysis by
        State Space Methods, 2nd ed. Oxford University Press.

        Harvey, A.C. (1989). Forecasting, Structural Time Series Models
        and the Kalman Filter. Cambridge University Press.
    """
    x = np.asarray(x, dtype=np.float64)

    if var_obs is None or var_state is None:
        var_obs, var_state = _estimate_local_level_params(x)

    var_obs = max(var_obs, 1e-10)
    var_state = max(var_state, 1e-10)

    mu_smooth = _kalman_filter_smooth(x, var_obs, var_state)
    return mu_smooth, var_obs, var_state


def _estimate_local_level_params(x: np.ndarray) -> Tuple[float, float]:
    """
    Estimate local level model parameters via MLE.

    Uses prediction error decomposition for likelihood evaluation.
    """
    var_x = np.var(x)
    if var_x < 1e-10:
        return 1e-10, 1e-10

    def nll(params):
        return _kalman_nll(x, params[0], params[1])

    init = [np.log(var_x * 0.5), np.log(var_x * 0.1)]
    res = minimize(nll, init, method='L-BFGS-B')

    return np.exp(res.x[0]), np.exp(res.x[1])


# Pre-configured smoothers for ablation study
TRANSITION_SMOOTHERS = {
    # EWMA variants: fast / medium / slow smoothing
    "ewma_090": lambda x: ewma_smoother(x, lam=0.90),
    "ewma_094": lambda x: ewma_smoother(x, lam=0.94),
    "ewma_097": lambda x: ewma_smoother(x, lam=0.97),
    # Rolling mean: monthly / quarterly
    "rolling_20": lambda x: rolling_mean_smoother(x, window=20),
    "rolling_60": lambda x: rolling_mean_smoother(x, window=60),
    # Kalman local level (optimal linear smoother)
    "kalman": lambda x: kalman_local_level_smoother(x)[0],
}


def get_transition_smoother(name: str):
    """
    Get transition smoother function by name.

    Args:
        name: Smoother name from TRANSITION_SMOOTHERS

    Returns:
        Function that takes array x and returns smoothed array
    """
    if name not in TRANSITION_SMOOTHERS:
        raise ValueError(
            f"Unknown smoother: {name}. "
            f"Available: {list(TRANSITION_SMOOTHERS.keys())}"
        )
    return TRANSITION_SMOOTHERS[name]
