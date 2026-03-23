#!/usr/bin/env python3
"""
Paper 2, Section 3: Simulation Study -- Regime-Conditional Proper Scoring Rules.

This script implements a comprehensive Monte Carlo simulation to demonstrate when
and why unconditional QLIKE misranks regime-switching volatility models relative
to regime-conditional QLIKE.

Three DGPs:
    DGP 1: Two-state Markov-switching (discrete regime, known)
    DGP 2: Smooth transition (continuous regime, logistic)
    DGP 3: No regime (null case, single-state GARCH)

Four models fitted to each simulated path:
    HAR        -- linear, no regime awareness
    STR-HAR    -- regime-switching with estimated logistic transition
    Oracle-STR -- STR-HAR given the true regime indicator (DGP 1/2 only)
    GARCH(1,1) -- simple variance recursion baseline

For each (DGP, parameter combination) we run 200 Monte Carlo replications and
record:
    - Unconditional QLIKE ranking
    - Conditional QLIKE ranking per regime (calm / stress)
    - Misranking frequency (unconditional vs conditional disagree)
    - Size and power of the regime-conditional Diebold-Mariano test

Outputs saved to: outputs/simulation/

References:
    Patton, A.J. (2011). Volatility forecast comparison using imperfect
        volatility proxies. Journal of Econometrics, 160(1), 246-256.
    Giacomini, R. & White, H. (2006). Tests of conditional predictive
        ability. Econometrica, 74(6), 1545-1578.
    Gneiting, T. (2011). Making and evaluating point forecasts. JASA, 106(494).

Usage:
    python scripts/simulation_study.py
    python scripts/simulation_study.py --n-mc 50 --quick   # fast dev run
"""

# -----------------------------------------------------------------------
# Thread pinning -- must come before any numpy import
# -----------------------------------------------------------------------
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import argparse
import warnings
from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import Paper 1 utilities
PAPER1_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "paper1-latent-str",
)
sys.path.insert(0, PAPER1_ROOT)

from src.metrics import qlike
from src.dm_test import dm_test

warnings.filterwarnings("ignore", category=RuntimeWarning)

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
EPS = 1e-12
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "outputs", "simulation")

# Default simulation grid
DEFAULT_REGIME_IMBALANCE = [0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_VOL_CONTRAST = [1.5, 2.0, 3.0, 5.0]
DEFAULT_SAMPLE_SIZES = [500, 1000, 2000]
DEFAULT_N_MC = 200

# DGP base parameters (log-variance scale)
SIGMA_CALM_LOG = -9.0       # calm state mean log-variance
DF_STRESS = 5.0             # Student-t df for stress innovations
TRAIN_FRAC = 0.6            # fraction used for fitting
MIN_REGIME_OBS = 20         # minimum obs per regime for conditional eval

# DM test parameters
DM_LAG = 5
DM_ALPHA = 0.05


# =====================================================================
# Data Generating Processes
# =====================================================================

def _markov_chain(T, p_calm_calm, p_stress_stress, rng):
    """
    Simulate a two-state Markov chain.

    States: 0 = calm, 1 = stress.

    Parameters
    ----------
    T : int
        Length of chain.
    p_calm_calm : float
        P(s_t = 0 | s_{t-1} = 0).
    p_stress_stress : float
        P(s_t = 1 | s_{t-1} = 1).
    rng : np.random.Generator

    Returns
    -------
    states : ndarray of int, shape (T,)
    """
    states = np.empty(T, dtype=np.int32)
    # Stationary probability of calm
    pi_calm = (1 - p_stress_stress) / (2 - p_calm_calm - p_stress_stress + EPS)
    states[0] = 0 if rng.random() < pi_calm else 1

    for i in range(1, T):
        if states[i - 1] == 0:
            states[i] = 0 if rng.random() < p_calm_calm else 1
        else:
            states[i] = 1 if rng.random() < p_stress_stress else 0
    return states


def _transition_probs_from_imbalance(p_calm_target, persistence=0.95):
    """
    Compute Markov transition probabilities so that
    P(calm) ~ p_calm_target in stationarity, with given persistence.

    From detailed balance:
        pi_calm = (1 - p11) / (2 - p00 - p11)

    We fix p00 = persistence and solve for p11.

    Parameters
    ----------
    p_calm_target : float
        Desired stationary probability of calm state.
    persistence : float
        Diagonal element p00 = P(calm|calm).

    Returns
    -------
    p00, p11 : float
        Transition probabilities.
    """
    p00 = persistence
    # pi_calm = (1 - p11) / (2 - p00 - p11)
    # => pi_calm * (2 - p00 - p11) = 1 - p11
    # => 2*pi - pi*p00 - pi*p11 = 1 - p11
    # => p11*(1 - pi) = 1 - 2*pi + pi*p00
    # => p11 = (1 - 2*pi + pi*p00) / (1 - pi)
    pi = p_calm_target
    p11 = (1.0 - 2.0 * pi + pi * p00) / (1.0 - pi + EPS)
    p11 = np.clip(p11, 0.01, 0.999)
    return p00, p11


def dgp_markov_switching(T, p_calm, vol_contrast, rng):
    """
    DGP 1: Two-state Markov-switching log-variance process.

    Calm:   y_t ~ N(mu_calm, sigma_calm^2)
    Stress: y_t ~ mu_stress + sigma_stress * t_{df_stress}

    Parameters
    ----------
    T : int
        Sample size.
    p_calm : float
        Stationary probability of calm state.
    vol_contrast : float
        sigma_stress / sigma_calm in variance space.
    rng : np.random.Generator

    Returns
    -------
    y : ndarray, shape (T,)
        Simulated log-variance.
    regime : ndarray, shape (T,)
        True regime indicator (0=calm, 1=stress).
    """
    p00, p11 = _transition_probs_from_imbalance(p_calm)
    states = _markov_chain(T, p00, p11, rng)

    # Log-variance means
    mu_calm = SIGMA_CALM_LOG
    # vol_contrast is ratio of standard deviations in variance space
    # In log-var space: mu_stress = mu_calm + 2 * log(vol_contrast)
    mu_stress = mu_calm + 2.0 * np.log(vol_contrast)

    # Innovation scales in log-var space
    scale_calm = 0.25
    scale_stress = 0.40

    y = np.empty(T)
    for i in range(T):
        if states[i] == 0:
            y[i] = mu_calm + scale_calm * rng.standard_normal()
        else:
            y[i] = mu_stress + scale_stress * (
                rng.standard_t(DF_STRESS) / np.sqrt(DF_STRESS / (DF_STRESS - 2))
            )

    regime = states.astype(np.float64)
    return y, regime


def dgp_smooth_transition(T, p_calm, vol_contrast, rng):
    """
    DGP 2: Smooth transition (STR-like) log-variance process.

    A latent driver s_t follows an AR(1) process. The regime indicator is
    G_t = 1 / (1 + exp(-gamma * (s_t - c))), producing a continuous
    transition between calm and stress.

    Parameters
    ----------
    T : int
    p_calm : float
        Target fraction of time in calm (G < 0.5).
    vol_contrast : float
        Volatility ratio between stress and calm.
    rng : np.random.Generator

    Returns
    -------
    y : ndarray, shape (T,)
    regime : ndarray, shape (T,) -- continuous in [0, 1]
    """
    mu_calm = SIGMA_CALM_LOG
    mu_stress = mu_calm + 2.0 * np.log(vol_contrast)

    # AR(1) latent driver
    phi = 0.97
    sigma_s = 0.3
    s = np.empty(T)
    s[0] = rng.standard_normal() * sigma_s / np.sqrt(1 - phi ** 2)
    for i in range(1, T):
        s[i] = phi * s[i - 1] + sigma_s * rng.standard_normal()

    # Set threshold c so that P(G < 0.5) ~ p_calm
    # G < 0.5 iff s < c, so c = quantile(s, p_calm)
    c = np.quantile(s, p_calm)
    gamma_str = 5.0

    G = 1.0 / (1.0 + np.exp(-gamma_str * (s - c)))

    # Innovation scale blends between regimes
    scale_calm = 0.25
    scale_stress = 0.40

    y = np.empty(T)
    for i in range(T):
        mu_i = (1.0 - G[i]) * mu_calm + G[i] * mu_stress
        sig_i = (1.0 - G[i]) * scale_calm + G[i] * scale_stress
        y[i] = mu_i + sig_i * rng.standard_normal()

    return y, G


def dgp_no_regime(T, p_calm, vol_contrast, rng):
    """
    DGP 3: No regime -- single-state GARCH(1,1)-like process.

    This is the null case: no regime structure. The p_calm and vol_contrast
    arguments are ignored (used only for grid consistency).

    Parameters
    ----------
    T : int
    p_calm : float (unused)
    vol_contrast : float (unused)
    rng : np.random.Generator

    Returns
    -------
    y : ndarray, shape (T,)
        Simulated log-variance from a persistent AR(1) in log-var.
    regime : ndarray, shape (T,)
        Constant 0.5 (no true regime).
    """
    # AR(1) in log-variance, calibrated to typical equity RV dynamics
    phi = 0.98
    sigma_eta = 0.20
    mu = SIGMA_CALM_LOG + 1.0  # midpoint between calm and stress

    y = np.empty(T)
    y[0] = mu + rng.standard_normal() * sigma_eta / np.sqrt(1 - phi ** 2)
    for i in range(1, T):
        y[i] = mu * (1 - phi) + phi * y[i - 1] + sigma_eta * rng.standard_normal()

    regime = np.full(T, 0.5)
    return y, regime


DGP_REGISTRY = {
    "MS": dgp_markov_switching,
    "STR": dgp_smooth_transition,
    "H0": dgp_no_regime,
}


# =====================================================================
# HAR Features from simulated log-variance
# =====================================================================

def _build_har_features(y):
    """
    Construct HAR-style features from a log-variance series.

    x_d = y_{t-1}         (daily)
    x_w = mean(y_{t-5:t})  (weekly)
    x_m = mean(y_{t-22:t}) (monthly)

    Returns a DataFrame with columns [y, x_d, x_w, x_m] starting at
    index 22 (to avoid NaN).
    """
    T = len(y)
    s = pd.Series(y)
    x_d = s.shift(1)
    x_w = s.rolling(5, min_periods=5).mean().shift(1)
    x_m = s.rolling(22, min_periods=22).mean().shift(1)

    df = pd.DataFrame({"y": y, "x_d": x_d, "x_w": x_w, "x_m": x_m})
    df = df.iloc[22:].reset_index(drop=True)
    return df


# =====================================================================
# Model fitting and forecasting
# =====================================================================

def _fit_har(train_df):
    """Fit HAR by OLS. Returns coefficient vector."""
    Y = train_df["y"].values
    X = np.column_stack([
        np.ones(len(train_df)),
        train_df["x_d"].values,
        train_df["x_w"].values,
        train_df["x_m"].values,
    ])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta


def _predict_har(beta, df):
    """Generate HAR predictions."""
    X = np.column_stack([
        np.ones(len(df)),
        df["x_d"].values,
        df["x_w"].values,
        df["x_m"].values,
    ])
    return X @ beta


def _logistic(x):
    """Numerically stable logistic sigmoid."""
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _fit_str_har(train_df, q_col="q"):
    """
    Fit STR-HAR with two regimes using L-BFGS-B multi-start.

    Model: yhat = (1-G)*yL + G*yH
           G = logistic(gamma * (q - c))
           yL = b0L + bdL*x_d + bwL*x_w + bmL*x_m
           yH = b0H + bdH*x_d + bwH*x_w + bmH*x_m

    Parameters (9): [b0L, bdL, bwL, bmL, b0H, bdH, bwH, bmH, gamma]
    Transition centered at c=0 after standardization.

    Returns (params, mu, sd) -- standardization statistics.
    """
    cols = ["y", "x_d", "x_w", "x_m", q_col]
    tmp = train_df[cols].copy()
    mu = tmp.mean()
    sd = tmp.std().replace(0, 1.0)
    tmp_std = (tmp - mu) / sd

    y_std = tmp_std["y"].values
    q_std = tmp_std[q_col].values
    xd_std = tmp_std["x_d"].values
    xw_std = tmp_std["x_w"].values
    xm_std = tmp_std["x_m"].values

    def _sse(params):
        b0L, bdL, bwL, bmL, b0H, bdH, bwH, bmH, gamma = params
        G = _logistic(gamma * q_std)
        yL = b0L + bdL * xd_std + bwL * xw_std + bmL * xm_std
        yH = b0H + bdH * xd_std + bwH * xw_std + bmH * xm_std
        yhat = (1.0 - G) * yL + G * yH
        resid = y_std - yhat
        return float(np.sum(resid * resid) + 1e-3 * gamma ** 2)

    init0 = np.array([0.0, 0.3, 0.3, 0.3, 0.0, 0.3, 0.3, 0.3, 2.0])
    bounds = [
        (-10, 10), (-5, 5), (-5, 5), (-5, 5),
        (-10, 10), (-5, 5), (-5, 5), (-5, 5),
        (1e-3, 12.0),
    ]

    # 2 random starts with 500 maxiter is sufficient for simulation.
    # (Production code in Paper 1 uses 3 starts + 1500 iter + basinhopping.)
    best_x, best_f = init0.copy(), _sse(init0)
    for k in range(2):
        x0 = init0.copy()
        if k > 0:
            x0 += np.random.normal(scale=0.3, size=x0.shape)
            x0[8] = np.clip(x0[8], 0.5, 8.0)
        res = minimize(_sse, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        if np.isfinite(res.fun) and res.fun < best_f:
            best_x, best_f = res.x.copy(), res.fun

    return best_x, mu, sd


def _predict_str_har(params, mu, sd, df, q_col="q"):
    """Generate STR-HAR predictions and transition weights."""
    cols = ["y", "x_d", "x_w", "x_m", q_col]
    tmp = df[cols].copy()
    tmp_std = (tmp - mu) / sd

    b0L, bdL, bwL, bmL, b0H, bdH, bwH, bmH, gamma = params[:9]
    q_std = tmp_std[q_col].values
    G = _logistic(gamma * q_std)
    yL = b0L + bdL * tmp_std["x_d"].values + bwL * tmp_std["x_w"].values + bmL * tmp_std["x_m"].values
    yH = b0H + bdH * tmp_std["x_d"].values + bwH * tmp_std["x_w"].values + bmH * tmp_std["x_m"].values
    yhat_std = (1.0 - G) * yL + G * yH
    yhat = yhat_std * sd["y"] + mu["y"]
    return yhat, G


def _fit_garch_simple(y_series):
    """
    Fit a simple GARCH(1,1) in log-variance space using MLE on an AR(1).

    This is a lightweight surrogate: model log(RV_t) as AR(1) with
    Gaussian innovations.

    Returns (mu, phi, sigma_eta).
    """
    y = y_series.copy()
    T = len(y)
    if T < 10:
        return float(np.mean(y)), 0.5, float(np.std(y))

    y_lag = y[:-1]
    y_cur = y[1:]

    # OLS for AR(1): y_t = c + phi * y_{t-1}
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    beta, *_ = np.linalg.lstsq(X, y_cur, rcond=None)
    c_hat, phi_hat = beta
    phi_hat = np.clip(phi_hat, -0.999, 0.999)
    resid = y_cur - (c_hat + phi_hat * y_lag)
    sigma_eta = float(np.std(resid))
    mu_hat = c_hat / (1.0 - phi_hat + EPS)

    return mu_hat, phi_hat, max(sigma_eta, 1e-6)


def _predict_garch_simple(mu, phi, y_lag):
    """One-step GARCH-AR(1) prediction: yhat = mu*(1-phi) + phi*y_{t-1}."""
    return mu * (1.0 - phi) + phi * y_lag


# =====================================================================
# Scoring and evaluation
# =====================================================================

def _qlike_array(y_true, y_pred):
    """Element-wise QLIKE loss (wraps Paper 1 implementation)."""
    return qlike(y_true, y_pred)


def _conditional_qlike(y_true, y_pred, regime, threshold=0.5):
    """
    Compute QLIKE separately for calm (regime < threshold) and stress.

    Returns (qlike_calm, qlike_stress, n_calm, n_stress).
    """
    losses = _qlike_array(y_true, y_pred)
    calm = regime < threshold
    stress = ~calm

    n_calm = int(calm.sum())
    n_stress = int(stress.sum())

    q_calm = float(np.mean(losses[calm])) if n_calm >= MIN_REGIME_OBS else np.nan
    q_stress = float(np.mean(losses[stress])) if n_stress >= MIN_REGIME_OBS else np.nan

    return q_calm, q_stress, n_calm, n_stress


def _conditional_dm_test(loss_a, loss_b, regime, threshold=0.5):
    """
    Regime-conditional Diebold-Mariano test.

    Returns dict with keys: dm_calm, p_calm, dm_stress, p_stress.
    """
    calm = regime < threshold
    stress = ~calm

    result = {}
    if calm.sum() >= DM_LAG + 10:
        dm_c, p_c = dm_test(loss_a[calm], loss_b[calm], L=DM_LAG)
        result["dm_calm"] = dm_c
        result["p_calm"] = p_c
    else:
        result["dm_calm"] = np.nan
        result["p_calm"] = np.nan

    if stress.sum() >= DM_LAG + 10:
        dm_s, p_s = dm_test(loss_a[stress], loss_b[stress], L=DM_LAG)
        result["dm_stress"] = dm_s
        result["p_stress"] = p_s
    else:
        result["dm_stress"] = np.nan
        result["p_stress"] = np.nan

    return result


# =====================================================================
# Single replication
# =====================================================================

def _run_single_replication(args):
    """
    Run one Monte Carlo replication for a given DGP and parameter set.

    Parameters
    ----------
    args : tuple
        (dgp_name, p_calm, vol_contrast, T, mc_seed)

    Returns
    -------
    dict with all metrics for this replication.
    """
    dgp_name, p_calm, vol_contrast, T, mc_seed = args
    rng = np.random.default_rng(mc_seed)

    # Also set the legacy random state for scipy compatibility
    np.random.seed(mc_seed % (2**31))

    # ---- Generate data ----
    dgp_fn = DGP_REGISTRY[dgp_name]
    y_raw, regime_raw = dgp_fn(T, p_calm, vol_contrast, rng)

    # Build HAR features (drops first 22 obs)
    df = _build_har_features(y_raw)
    regime = regime_raw[22: 22 + len(df)]

    if len(df) < 100:
        return None

    # Train / test split
    n_train = int(len(df) * TRAIN_FRAC)
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    regime_test = regime[n_train:]
    y_test = test_df["y"].values
    n_test = len(test_df)

    if n_test < 50:
        return None

    # ---- Construct transition variable for STR-HAR ----
    # Use x_d (lagged daily log-var) as observable transition variable
    # This is the "estimated" regime -- model must discover it
    train_df["q_est"] = train_df["x_d"].values
    test_df["q_est"] = test_df["x_d"].values

    # True regime for oracle STR
    regime_train = regime[:n_train]
    train_df["q_true"] = regime_train
    test_df["q_true"] = regime_test

    # ---- Fit models ----
    # 1) HAR
    har_beta = _fit_har(train_df)
    yhat_har = _predict_har(har_beta, test_df)

    # 2) STR-HAR (estimated transition from x_d)
    try:
        str_params, str_mu, str_sd = _fit_str_har(train_df, q_col="q_est")
        yhat_str, G_str = _predict_str_har(str_params, str_mu, str_sd, test_df, q_col="q_est")
    except Exception:
        yhat_str = yhat_har.copy()

    # 3) Oracle STR-HAR (given true regime)
    if dgp_name != "H0":
        try:
            ora_params, ora_mu, ora_sd = _fit_str_har(train_df, q_col="q_true")
            yhat_ora, G_ora = _predict_str_har(ora_params, ora_mu, ora_sd, test_df, q_col="q_true")
        except Exception:
            yhat_ora = yhat_har.copy()
    else:
        yhat_ora = yhat_har.copy()

    # 4) GARCH-AR(1)
    mu_g, phi_g, sig_g = _fit_garch_simple(train_df["y"].values)
    yhat_garch = mu_g * (1.0 - phi_g) + phi_g * test_df["x_d"].values

    # ---- Evaluate ----
    models = {
        "HAR": yhat_har,
        "STR": yhat_str,
        "Oracle": yhat_ora,
        "GARCH": yhat_garch,
    }

    # Clip predictions to prevent extreme QLIKE from outlier forecasts
    y_train = train_df["y"].values
    clip_lo = y_train.min() - 5.0
    clip_hi = y_train.max() + 5.0
    models = {name: np.clip(yhat, clip_lo, clip_hi) for name, yhat in models.items()}

    # Regime threshold for conditional evaluation
    # For MS: 0.5 (binary), for STR: 0.5 (continuous), for H0: 0.5
    threshold = 0.5

    # Unconditional QLIKE
    uncond = {}
    for name, yhat in models.items():
        losses = _qlike_array(y_test, yhat)
        uncond[name] = float(np.mean(losses))

    # Conditional QLIKE
    cond_calm = {}
    cond_stress = {}
    n_calm_test = 0
    n_stress_test = 0

    for name, yhat in models.items():
        qc, qs, nc, ns = _conditional_qlike(y_test, yhat, regime_test, threshold)
        cond_calm[name] = qc
        cond_stress[name] = qs
        n_calm_test = nc
        n_stress_test = ns

    # Rankings -- all-model comparison
    def _rank_dict(d):
        """Return model name with lowest loss (best)."""
        valid = {k: v for k, v in d.items() if np.isfinite(v)}
        if not valid:
            return None
        return min(valid, key=valid.get)

    uncond_winner = _rank_dict(uncond)
    calm_winner = _rank_dict(cond_calm)
    stress_winner = _rank_dict(cond_stress)

    # Misranking (all-model): unconditional and conditional calm disagree
    misrank_calm = (uncond_winner != calm_winner) if (uncond_winner and calm_winner) else None
    misrank_stress = (uncond_winner != stress_winner) if (uncond_winner and stress_winner) else None

    misrank_any = False
    if misrank_calm is not None and misrank_calm:
        misrank_any = True
    if misrank_stress is not None and misrank_stress:
        misrank_any = True

    # ---- Pairwise misranking: STR vs HAR (the key comparison) ----
    # This is the central result: does unconditional QLIKE agree with
    # conditional QLIKE about which of {STR, HAR} is better?
    #
    # "pairwise_misrank_str_har": unconditional says one wins, but at
    # least one conditional regime says the other wins.
    str_beats_har_uncond = uncond["STR"] < uncond["HAR"]

    # Pairwise conditional: does STR beat HAR in calm? in stress?
    str_beats_har_calm = None
    str_beats_har_stress = None
    if np.isfinite(cond_calm["STR"]) and np.isfinite(cond_calm["HAR"]):
        str_beats_har_calm = cond_calm["STR"] < cond_calm["HAR"]
    if np.isfinite(cond_stress["STR"]) and np.isfinite(cond_stress["HAR"]):
        str_beats_har_stress = cond_stress["STR"] < cond_stress["HAR"]

    # Pairwise misranking: unconditional and at least one conditional disagree
    pw_misrank_calm = None
    pw_misrank_stress = None
    pw_misrank_any = False

    if str_beats_har_calm is not None:
        pw_misrank_calm = (str_beats_har_uncond != str_beats_har_calm)
        if pw_misrank_calm:
            pw_misrank_any = True
    if str_beats_har_stress is not None:
        pw_misrank_stress = (str_beats_har_uncond != str_beats_har_stress)
        if pw_misrank_stress:
            pw_misrank_any = True

    # Specific pattern: HAR wins unconditionally but STR wins in stress
    # (Corollary from Theorem 1: calm dominance masks stress advantage)
    har_wins_uncond_str_wins_stress = (
        (not str_beats_har_uncond)
        and (str_beats_har_stress is not None)
        and str_beats_har_stress
    )

    # ---- DM tests: STR vs HAR ----
    loss_har = _qlike_array(y_test, yhat_har)
    loss_str = _qlike_array(y_test, yhat_str)
    loss_ora = _qlike_array(y_test, yhat_ora)

    # Unconditional DM: STR vs HAR
    dm_uncond, p_uncond = dm_test(loss_str, loss_har, L=DM_LAG)

    # Conditional DM: STR vs HAR
    cond_dm = _conditional_dm_test(loss_str, loss_har, regime_test, threshold)

    # Unconditional DM: Oracle vs HAR
    dm_ora_uncond, p_ora_uncond = dm_test(loss_ora, loss_har, L=DM_LAG)

    # Conditional DM: Oracle vs HAR
    cond_dm_ora = _conditional_dm_test(loss_ora, loss_har, regime_test, threshold)

    # ---- Build result record ----
    result = {
        "dgp": dgp_name,
        "p_calm": p_calm,
        "vol_contrast": vol_contrast,
        "T": T,
        "seed": mc_seed,
        "n_test": n_test,
        "n_calm_test": n_calm_test,
        "n_stress_test": n_stress_test,
        # Unconditional QLIKE
        "qlike_har": uncond["HAR"],
        "qlike_str": uncond["STR"],
        "qlike_oracle": uncond["Oracle"],
        "qlike_garch": uncond["GARCH"],
        # Conditional QLIKE -- calm
        "qlike_har_calm": cond_calm["HAR"],
        "qlike_str_calm": cond_calm["STR"],
        "qlike_oracle_calm": cond_calm["Oracle"],
        "qlike_garch_calm": cond_calm["GARCH"],
        # Conditional QLIKE -- stress
        "qlike_har_stress": cond_stress["HAR"],
        "qlike_str_stress": cond_stress["STR"],
        "qlike_oracle_stress": cond_stress["Oracle"],
        "qlike_garch_stress": cond_stress["GARCH"],
        # Rankings (all-model)
        "winner_uncond": uncond_winner,
        "winner_calm": calm_winner,
        "winner_stress": stress_winner,
        "misrank_calm": misrank_calm,
        "misrank_stress": misrank_stress,
        "misrank_any": misrank_any,
        # Pairwise misranking: STR vs HAR
        "str_beats_har_uncond": str_beats_har_uncond,
        "str_beats_har_calm": str_beats_har_calm,
        "str_beats_har_stress": str_beats_har_stress,
        "pw_misrank_calm": pw_misrank_calm,
        "pw_misrank_stress": pw_misrank_stress,
        "pw_misrank_any": pw_misrank_any,
        "har_wins_uncond_str_wins_stress": har_wins_uncond_str_wins_stress,
        # DM test: STR vs HAR
        "dm_str_har_uncond": dm_uncond,
        "p_str_har_uncond": p_uncond,
        "dm_str_har_calm": cond_dm["dm_calm"],
        "p_str_har_calm": cond_dm["p_calm"],
        "dm_str_har_stress": cond_dm["dm_stress"],
        "p_str_har_stress": cond_dm["p_stress"],
        # DM test: Oracle vs HAR
        "dm_ora_har_uncond": dm_ora_uncond,
        "p_ora_har_uncond": p_ora_uncond,
        "dm_ora_har_calm": cond_dm_ora["dm_calm"],
        "p_ora_har_calm": cond_dm_ora["p_calm"],
        "dm_ora_har_stress": cond_dm_ora["dm_stress"],
        "p_ora_har_stress": cond_dm_ora["p_stress"],
    }

    return result


# =====================================================================
# Aggregation and summary tables
# =====================================================================

def _compute_summary_tables(results_df):
    """
    Aggregate raw MC results into summary tables.

    Returns dict of DataFrames.
    """
    tables = {}

    # --- Table 1: Misranking frequency by DGP and parameters ---
    group_cols = ["dgp", "p_calm", "vol_contrast", "T"]
    agg = results_df.groupby(group_cols).agg(
        n_reps=("seed", "count"),
        # All-model misranking
        misrank_calm_rate=("misrank_calm", "mean"),
        misrank_stress_rate=("misrank_stress", "mean"),
        misrank_any_rate=("misrank_any", "mean"),
        # Pairwise STR vs HAR misranking
        pw_misrank_calm_rate=("pw_misrank_calm", lambda x: x.dropna().mean()
                              if x.dropna().shape[0] > 0 else np.nan),
        pw_misrank_stress_rate=("pw_misrank_stress", lambda x: x.dropna().mean()
                                if x.dropna().shape[0] > 0 else np.nan),
        pw_misrank_any_rate=("pw_misrank_any", "mean"),
        # Specific pattern: HAR wins uncond but STR wins stress
        har_uncond_str_stress_rate=("har_wins_uncond_str_wins_stress", "mean"),
        # Winner frequencies
        uncond_har_rate=("winner_uncond", lambda x: (x == "HAR").mean()),
        uncond_str_rate=("winner_uncond", lambda x: (x == "STR").mean()),
        uncond_oracle_rate=("winner_uncond", lambda x: (x == "Oracle").mean()),
        uncond_garch_rate=("winner_uncond", lambda x: (x == "GARCH").mean()),
        calm_har_rate=("winner_calm", lambda x: (x == "HAR").mean()),
        calm_str_rate=("winner_calm", lambda x: (x == "STR").mean()),
        stress_oracle_rate=("winner_stress", lambda x: (x == "Oracle").mean()),
        # STR beats HAR rates
        str_beats_har_uncond_rate=("str_beats_har_uncond", "mean"),
        str_beats_har_stress_rate=("str_beats_har_stress", lambda x: x.dropna().mean()
                                   if x.dropna().shape[0] > 0 else np.nan),
    ).reset_index()
    tables["misranking"] = agg

    # --- Table 2: DM test size/power ---
    dm_agg = results_df.groupby(group_cols).agg(
        # STR vs HAR: unconditional
        rej_str_uncond=("p_str_har_uncond", lambda x: (x < DM_ALPHA).mean()),
        # STR vs HAR: conditional calm
        rej_str_calm=("p_str_har_calm", lambda x: (x.dropna() < DM_ALPHA).mean()
                      if x.dropna().shape[0] > 0 else np.nan),
        # STR vs HAR: conditional stress
        rej_str_stress=("p_str_har_stress", lambda x: (x.dropna() < DM_ALPHA).mean()
                        if x.dropna().shape[0] > 0 else np.nan),
        # Oracle vs HAR: unconditional
        rej_ora_uncond=("p_ora_har_uncond", lambda x: (x < DM_ALPHA).mean()),
        # Oracle vs HAR: conditional calm
        rej_ora_calm=("p_ora_har_calm", lambda x: (x.dropna() < DM_ALPHA).mean()
                      if x.dropna().shape[0] > 0 else np.nan),
        # Oracle vs HAR: conditional stress
        rej_ora_stress=("p_ora_har_stress", lambda x: (x.dropna() < DM_ALPHA).mean()
                        if x.dropna().shape[0] > 0 else np.nan),
    ).reset_index()
    tables["dm_tests"] = dm_agg

    # --- Table 3: Mean QLIKE by model and regime (collapsed over MC) ---
    qlike_cols_uncond = ["qlike_har", "qlike_str", "qlike_oracle", "qlike_garch"]
    qlike_cols_calm = ["qlike_har_calm", "qlike_str_calm", "qlike_oracle_calm", "qlike_garch_calm"]
    qlike_cols_stress = ["qlike_har_stress", "qlike_str_stress", "qlike_oracle_stress", "qlike_garch_stress"]

    qlike_agg = results_df.groupby(group_cols).agg(
        **{col: (col, "mean") for col in qlike_cols_uncond + qlike_cols_calm + qlike_cols_stress}
    ).reset_index()
    tables["qlike_means"] = qlike_agg

    return tables


def _print_summary(tables, dgp_name):
    """Print formatted console summary for one DGP."""
    mr = tables["misranking"]
    mr_dgp = mr[mr["dgp"] == dgp_name]
    if mr_dgp.empty:
        return

    print(f"\n{'='*78}")
    print(f"  DGP: {dgp_name}")
    print(f"{'='*78}")

    # Pairwise misranking: STR vs HAR (key paper result)
    print(f"\n  Pairwise Misranking: STR vs HAR (uncond vs conditional disagree):")
    print(f"  {'p_calm':>7s} {'contrast':>9s} {'T':>6s} {'n':>5s} "
          f"{'pw_any':>7s} {'pw_calm':>8s} {'pw_str':>8s} "
          f"{'HAR>STR':>8s}")
    print(f"  {'-'*68}")
    print(f"  {'':>7s} {'':>9s} {'':>6s} {'':>5s} "
          f"{'':>7s} {'':>8s} {'':>8s} "
          f"{'unc+str':>8s}")
    print(f"  {'-'*68}")

    for _, row in mr_dgp.iterrows():
        def _f(v):
            return f"{v:8.3f}" if np.isfinite(v) else f"{'N/A':>8s}"
        print(f"  {row['p_calm']:7.1f} {row['vol_contrast']:9.1f} {int(row['T']):6d} "
              f"{int(row['n_reps']):5d} "
              f"{row['pw_misrank_any_rate']:7.3f} "
              f"{_f(row['pw_misrank_calm_rate'])} "
              f"{_f(row['pw_misrank_stress_rate'])} "
              f"{row['har_uncond_str_stress_rate']:8.3f}")

    # All-model misranking
    print(f"\n  All-Model Misranking (unconditional vs conditional winner disagree):")
    print(f"  {'p_calm':>7s} {'contrast':>9s} {'T':>6s} {'n':>5s} "
          f"{'any':>7s} {'calm':>7s} {'stress':>7s}")
    print(f"  {'-'*55}")

    for _, row in mr_dgp.iterrows():
        print(f"  {row['p_calm']:7.1f} {row['vol_contrast']:9.1f} {int(row['T']):6d} "
              f"{int(row['n_reps']):5d} "
              f"{row['misrank_any_rate']:7.3f} "
              f"{row['misrank_calm_rate']:7.3f} "
              f"{row['misrank_stress_rate']:7.3f}")

    # DM test summary
    dm = tables["dm_tests"]
    dm_dgp = dm[dm["dgp"] == dgp_name]
    if not dm_dgp.empty:
        print(f"\n  DM Test Rejection Rates (alpha={DM_ALPHA}):")
        print(f"  {'p_calm':>7s} {'contrast':>9s} {'T':>6s} "
              f"{'STR unc':>9s} {'STR calm':>9s} {'STR str':>9s} "
              f"{'Ora unc':>9s} {'Ora calm':>9s} {'Ora str':>9s}")
        print(f"  {'-'*78}")

        for _, row in dm_dgp.iterrows():
            def _fmt(v):
                return f"{v:9.3f}" if np.isfinite(v) else f"{'N/A':>9s}"
            print(f"  {row['p_calm']:7.1f} {row['vol_contrast']:9.1f} {int(row['T']):6d} "
                  f"{_fmt(row['rej_str_uncond'])} "
                  f"{_fmt(row['rej_str_calm'])} "
                  f"{_fmt(row['rej_str_stress'])} "
                  f"{_fmt(row['rej_ora_uncond'])} "
                  f"{_fmt(row['rej_ora_calm'])} "
                  f"{_fmt(row['rej_ora_stress'])}")

    # Winner frequencies for most imbalanced case
    extreme = mr_dgp[mr_dgp["p_calm"] == mr_dgp["p_calm"].max()]
    if not extreme.empty:
        row = extreme.iloc[0]
        print(f"\n  Unconditional winner freq (most imbalanced, p_calm={row['p_calm']:.1f}):")
        print(f"    HAR={row['uncond_har_rate']:.3f}  STR={row['uncond_str_rate']:.3f}  "
              f"Oracle={row['uncond_oracle_rate']:.3f}  GARCH={row['uncond_garch_rate']:.3f}")


def _export_latex_tables(tables, output_dir):
    """
    Export summary tables as LaTeX .tex files alongside the CSVs.

    Parameters
    ----------
    tables : dict of DataFrames
        Summary tables from _compute_summary_tables().
    output_dir : str
        Directory to write .tex files.
    """
    float_fmt = "%.3f"
    for name, tbl in tables.items():
        tex_path = os.path.join(output_dir, f"table_{name}.tex")
        tbl.to_latex(tex_path, index=False, float_format=float_fmt, na_rep="--")
        print(f"  LaTeX table saved: {tex_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Paper 2 simulation study: regime-conditional scoring rules."
    )
    parser.add_argument("--n-mc", type=int, default=DEFAULT_N_MC,
                        help=f"Monte Carlo replications (default {DEFAULT_N_MC})")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="Parallel workers (default: cpu_count-1)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: fewer params, fewer MC reps")
    parser.add_argument("--dgp", type=str, default=None, choices=["MS", "STR", "H0"],
                        help="Run only one DGP")
    parser.add_argument("--seed", type=int, default=20260321,
                        help="Base random seed")
    args = parser.parse_args()

    # Determine grid
    if args.quick:
        regime_imbalance = [0.5, 0.8]
        vol_contrast = [2.0, 5.0]
        sample_sizes = [500]
        n_mc = min(args.n_mc, 50)
    else:
        regime_imbalance = DEFAULT_REGIME_IMBALANCE
        vol_contrast = DEFAULT_VOL_CONTRAST
        sample_sizes = DEFAULT_SAMPLE_SIZES
        n_mc = args.n_mc

    dgp_names = [args.dgp] if args.dgp else ["MS", "STR", "H0"]

    # For DGP H0, vol_contrast and p_calm do not matter --
    # run only one configuration per sample size
    tasks = []
    seed_counter = args.seed

    for dgp_name in dgp_names:
        if dgp_name == "H0":
            # Single config per T (p_calm and vol_contrast are irrelevant)
            for T in sample_sizes:
                for mc in range(n_mc):
                    tasks.append((dgp_name, 0.5, 1.0, T, seed_counter))
                    seed_counter += 1
        else:
            for p_calm, vc, T in product(regime_imbalance, vol_contrast, sample_sizes):
                for mc in range(n_mc):
                    tasks.append((dgp_name, p_calm, vc, T, seed_counter))
                    seed_counter += 1

    n_tasks = len(tasks)
    n_configs = n_tasks // n_mc if n_mc > 0 else 0

    print("=" * 78)
    print("  PAPER 2 SIMULATION STUDY: REGIME-CONDITIONAL SCORING RULES")
    print("=" * 78)
    print(f"  DGPs:             {dgp_names}")
    print(f"  Regime imbalance: {regime_imbalance}")
    print(f"  Vol contrast:     {vol_contrast}")
    print(f"  Sample sizes:     {sample_sizes}")
    print(f"  MC replications:  {n_mc}")
    print(f"  Total configs:    {n_configs}")
    print(f"  Total tasks:      {n_tasks}")

    # Determine parallelism
    if args.n_jobs is not None:
        n_workers = args.n_jobs
    else:
        n_workers = max(1, os.cpu_count() - 1)
    print(f"  Workers:          {n_workers}")
    print("=" * 78)

    # ---- Run simulations ----
    t0 = time.time()
    results = []

    if n_workers == 1:
        # Sequential for debugging
        for i, task in enumerate(tasks):
            res = _run_single_replication(task)
            if res is not None:
                results.append(res)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_tasks - i - 1) / rate
                print(f"  [{i+1:>6d}/{n_tasks}] {elapsed:6.1f}s elapsed, "
                      f"ETA {eta:6.1f}s ({rate:.1f} tasks/s)")
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_run_single_replication, t): t for t in tasks}
            for future in as_completed(futures):
                completed += 1
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                except Exception as exc:
                    task_params = futures[future]
                    print(
                        f"  [ERROR] Task failed: dgp={task_params[0]}, "
                        f"p_calm={task_params[1]}, vol_contrast={task_params[2]}, "
                        f"T={task_params[3]}, seed={task_params[4]} -- {exc}",
                        file=sys.stderr,
                    )
                if completed % 200 == 0:
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (n_tasks - completed) / rate if rate > 0 else 0
                    print(f"  [{completed:>6d}/{n_tasks}] {elapsed:6.1f}s elapsed, "
                          f"ETA {eta:6.1f}s ({rate:.1f} tasks/s)")

    elapsed_total = time.time() - t0
    print(f"\n  Completed {len(results)}/{n_tasks} replications in {elapsed_total:.1f}s")

    if not results:
        print("  ERROR: No valid results. Check DGP parameters.")
        sys.exit(1)

    # ---- Assemble results ----
    results_df = pd.DataFrame(results)

    # ---- Save raw results ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw_path = os.path.join(OUTPUT_DIR, "raw_results.csv")
    results_df.to_csv(raw_path, index=False)
    print(f"\n  Raw results saved to: {raw_path}")

    # ---- Compute and save summary tables ----
    tables = _compute_summary_tables(results_df)

    for name, tbl in tables.items():
        path = os.path.join(OUTPUT_DIR, f"table_{name}.csv")
        tbl.to_csv(path, index=False)
        print(f"  Table saved: {path}")

    # ---- Export LaTeX tables ----
    _export_latex_tables(tables, OUTPUT_DIR)

    # ---- Console summaries ----
    for dgp_name in dgp_names:
        _print_summary(tables, dgp_name)

    # ---- Key findings summary ----
    print(f"\n{'='*78}")
    print("  KEY FINDINGS SUMMARY")
    print(f"{'='*78}")

    for dgp_name in dgp_names:
        sub = results_df[results_df["dgp"] == dgp_name]
        if sub.empty:
            continue

        n_valid = len(sub)

        # Pairwise misranking: STR vs HAR
        pw_any = sub["pw_misrank_any"].mean()
        har_unc_str_str = sub["har_wins_uncond_str_wins_stress"].mean()

        # Most extreme imbalance case
        if dgp_name != "H0":
            extreme = sub[sub["p_calm"] == sub["p_calm"].max()]
            pw_extreme = extreme["pw_misrank_any"].mean() if not extreme.empty else np.nan
            har_unc_str_str_extreme = (
                extreme["har_wins_uncond_str_wins_stress"].mean()
                if not extreme.empty else np.nan
            )
        else:
            pw_extreme = pw_any
            har_unc_str_str_extreme = har_unc_str_str

        print(f"\n  DGP={dgp_name}: {n_valid} replications")
        print(f"    Pairwise misranking (STR vs HAR):    {pw_any:.3f}")
        print(f"    HAR-uncond/STR-stress pattern:       {har_unc_str_str:.3f}")
        if dgp_name != "H0":
            print(f"    PW misrank at max imbalance:         {pw_extreme:.3f}")
            print(f"    HAR-unc/STR-str at max imbalance:    {har_unc_str_str_extreme:.3f}")

        # All-model misranking
        mr_any = sub["misrank_any"].mean()
        print(f"    All-model misranking rate:            {mr_any:.3f}")

        # Unconditional DM rejection rate (STR vs HAR)
        rej_uncond = (sub["p_str_har_uncond"] < DM_ALPHA).mean()
        rej_stress = sub["p_str_har_stress"].dropna()
        rej_stress_rate = (rej_stress < DM_ALPHA).mean() if len(rej_stress) > 0 else np.nan
        print(f"    DM(STR vs HAR) uncond reject rate:   {rej_uncond:.3f}")
        if np.isfinite(rej_stress_rate):
            print(f"    DM(STR vs HAR) stress reject rate:   {rej_stress_rate:.3f}")

        if dgp_name == "H0":
            # Size check: rejection should be ~alpha under null
            print(f"    (Expected under H0: ~{DM_ALPHA:.3f})")

    print(f"\n{'='*78}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Total runtime:    {elapsed_total:.1f}s")
    print(f"{'='*78}")


if __name__ == "__main__":
    main()
