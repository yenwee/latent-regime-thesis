#!/usr/bin/env python3
"""
Paper 2 Option B: Learned Transition Function
Can we learn the optimal nonlinear transition mechanism from data?

Paper 1 assumes G(q) = logistic(gamma * q). But what if the true
regime switch is asymmetric, has a different threshold, or is non-monotonic?

Tests:
1. Compare parametric transitions: logistic, exponential, double-logistic
2. Implement neural transition: G(q) = MLP(q) constrained to [0,1]
3. Implement spline transition: G(q) = monotonic cubic spline
4. Compare QLIKE across all transitions on multiple assets
5. Visualize learned G(q) — does it reveal economic structure?

Uses existing Paper 1 data pipeline. Tests on 4 diverse assets.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PAPER1_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "paper1-latent-str")
sys.path.insert(0, PAPER1_ROOT)

import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from src.data import download_asset_data, prepare_features
from src.vrnn import train_deep_ssm, project_latent_to_scalar
from src.str_har import fit_har_ols, har_predict, logistic
from src.metrics import qlike

np.random.seed(123)

TEST_ASSETS = ["^GSPC", "XLE", "^TNX", "GC=F"]
H = 5
ROLL_WINDOW = 600

print("=" * 70)
print("PAPER 2 OPTION B: LEARNED TRANSITION FUNCTION")
print("=" * 70)

# ============================================================
# Transition functions
# ============================================================

def logistic_transition(q, gamma):
    """Standard logistic: G(q) = 1/(1+exp(-gamma*q))"""
    return 1.0 / (1.0 + np.exp(-np.clip(gamma * q, -50, 50)))

def exponential_transition(q, gamma):
    """Exponential: G(q) = 1 - exp(-(gamma*q)^2)"""
    return 1.0 - np.exp(-np.clip((gamma * q) ** 2, 0, 50))

def asymmetric_logistic(q, gamma, alpha_asym):
    """Asymmetric logistic: different speeds for entering vs exiting stress.
    alpha_asym > 1: faster entry into stress, slower exit
    alpha_asym < 1: slower entry, faster exit"""
    x = gamma * q
    x = np.clip(x, -50, 50)
    pos = x >= 0
    result = np.zeros_like(x)
    result[pos] = 1.0 / (1.0 + np.exp(-alpha_asym * x[pos]))
    result[~pos] = 1.0 / (1.0 + np.exp(-x[~pos] / alpha_asym))
    return result

def threshold_logistic(q, gamma, c):
    """Shifted logistic with learned threshold: G(q) = logistic(gamma*(q-c))"""
    return 1.0 / (1.0 + np.exp(-np.clip(gamma * (q - c), -50, 50)))

def neural_transition_eval(q, params):
    """Simple 2-layer neural net: q -> [4 hidden, tanh] -> [1, sigmoid]
    params: [w1(4), b1(4), w2(4), b2(1)] = 13 params"""
    w1 = params[:4]
    b1 = params[4:8]
    w2 = params[8:12]
    b2 = params[12]
    h = np.tanh(q[:, None] * w1[None, :] + b1[None, :])  # T x 4
    out = h @ w2 + b2  # T
    return 1.0 / (1.0 + np.exp(-np.clip(out, -50, 50)))


# ============================================================
# STR-HAR with custom transition
# ============================================================

def str_har_predict(beta_low, beta_high, G, X):
    """STR-HAR prediction: y = (1-G)*X@beta_low + G*X@beta_high"""
    return (1.0 - G) * (X @ beta_low) + G * (X @ beta_high)

def fit_str_har_custom(y, X, q, transition_fn, transition_params_init, transition_bounds,
                       gamma_lam=1e-3):
    """
    Fit STR-HAR with a custom transition function.
    Jointly optimizes HAR coefficients and transition parameters.
    """
    n_trans = len(transition_params_init)
    # Total params: 4 (beta_low) + 4 (beta_high) + n_trans (transition)
    n_total = 8 + n_trans

    def obj(params):
        beta_low = params[:4]
        beta_high = params[4:8]
        trans_params = params[8:]

        try:
            G = transition_fn(q, *trans_params)
        except Exception:
            return 1e10

        yhat = str_har_predict(beta_low, beta_high, G, X)
        resid = y - yhat
        sse = np.sum(resid ** 2)

        # Regularize transition params
        reg = gamma_lam * np.sum(trans_params[:1] ** 2) if n_trans > 0 else 0
        return sse + reg

    # Initialize: OLS for beta, default for transition
    beta_init = np.linalg.lstsq(X, y, rcond=None)[0]
    init = np.concatenate([beta_init, beta_init, transition_params_init])

    bounds = [(-10, 10)] * 4 + [(-10, 10)] * 4 + list(transition_bounds)

    best_f = np.inf
    best_x = init
    for k in range(3):
        x0 = init.copy()
        if k > 0:
            x0[8:] += np.random.normal(0, 0.3, n_trans)
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds, options={"maxiter": 1500})
        if res.fun < best_f:
            best_f = res.fun
            best_x = res.x

    return best_x[:4], best_x[4:8], best_x[8:]


# ============================================================
# 1. Prepare data
# ============================================================
print("\n[1/4] Preparing data and training VRNNs...")

asset_data = {}
for ticker in TEST_ASSETS:
    df = download_asset_data(ticker, start="2015-01-01", end="2025-12-31")
    df = prepare_features(df, H, q_obs_smooth_span=10)

    # Train VRNN
    X_cols = ["x1_logv", "x2_absr", "x3_logvol"]
    split = ROLL_WINDOW
    mu_X = df.iloc[:split][X_cols].mean()
    sd_X = df.iloc[:split][X_cols].std().replace(0, 1.0)
    X_std = ((df[X_cols] - mu_X) / sd_X).values

    X_train = torch.tensor(X_std[:split], dtype=torch.float32)
    X_infer = torch.tensor(X_std, dtype=torch.float32)

    _, Z, elbo = train_deep_ssm(
        X_train, X_infer, latent_dim=2, gru_hidden=16, dec_hidden=32,
        lr=0.002, epochs=400, patience=40, verbose=False,
    )
    q = project_latent_to_scalar(Z, df["x1_logv"].values, split)
    df["q_ssm"] = q
    asset_data[ticker] = df
    print(f"  {ticker}: {len(df)} days, ELBO={elbo:.4f}")

# ============================================================
# 2. Compare transition functions per asset
# ============================================================
print("\n[2/4] Comparing transition functions...")

transition_configs = {
    "logistic": {
        "fn": lambda q, gamma: logistic_transition(q, gamma),
        "init": [2.0],
        "bounds": [(0.01, 15.0)],
    },
    "exponential": {
        "fn": lambda q, gamma: exponential_transition(q, gamma),
        "init": [2.0],
        "bounds": [(0.01, 15.0)],
    },
    "asymmetric": {
        "fn": lambda q, gamma, alpha: asymmetric_logistic(q, gamma, alpha),
        "init": [2.0, 1.5],
        "bounds": [(0.01, 15.0), (0.2, 5.0)],
    },
    "threshold": {
        "fn": lambda q, gamma, c: threshold_logistic(q, gamma, c),
        "init": [2.0, 0.0],
        "bounds": [(0.01, 15.0), (-3.0, 3.0)],
    },
    "neural": {
        "fn": lambda q, *p: neural_transition_eval(q, np.array(p)),
        "init": list(np.random.normal(0, 0.5, 13)),
        "bounds": [(-5, 5)] * 13,
    },
}

all_results = {}
for ticker in TEST_ASSETS:
    df = asset_data[ticker]
    split = ROLL_WINDOW

    # Standardize
    cols = ["y", "x_d", "x_w", "x_m", "q_ssm"]
    train = df.iloc[:split][cols]
    mu = train.mean()
    sd = train.std().replace(0, 1.0)
    df_std = (df[cols] - mu) / sd

    y_train = df_std.iloc[:split]["y"].values
    X_train = np.column_stack([np.ones(split), df_std.iloc[:split][["x_d", "x_w", "x_m"]].values])
    q_train = df_std.iloc[:split]["q_ssm"].values

    # OOS forecast
    ticker_results = {}
    for trans_name, cfg in transition_configs.items():
        forecasts = []
        for i in range(split, len(df) - 1):
            if i - ROLL_WINDOW < 0:
                continue

            # Train window
            tw = df_std.iloc[i - ROLL_WINDOW:i]
            y_tw = tw["y"].values
            X_tw = np.column_stack([np.ones(ROLL_WINDOW), tw[["x_d", "x_w", "x_m"]].values])
            q_tw = tw["q_ssm"].values

            # Test point
            tp = df_std.iloc[i:i+1]
            X_tp = np.column_stack([np.ones(1), tp[["x_d", "x_w", "x_m"]].values])
            q_tp = tp["q_ssm"].values

            try:
                beta_l, beta_h, trans_p = fit_str_har_custom(
                    y_tw, X_tw, q_tw, cfg["fn"], cfg["init"], cfg["bounds"]
                )
                G_test = cfg["fn"](q_tp, *trans_p)
                yhat_std = str_har_predict(beta_l, beta_h, G_test, X_tp)
                yhat = yhat_std * sd["y"] + mu["y"]
                forecasts.append({
                    "date": df.index[i],
                    "y": float(df.iloc[i]["y"]),
                    "yhat": float(yhat[0]),
                })
            except Exception:
                continue

        if forecasts:
            fdf = pd.DataFrame(forecasts)
            q_val = float(qlike(fdf["y"], fdf["yhat"]).mean())
            ticker_results[trans_name] = {"qlike": q_val, "n": len(forecasts)}

    all_results[ticker] = ticker_results
    print(f"  {ticker}: done ({len(forecasts)} OOS days)")

# ============================================================
# 3. Results table
# ============================================================
print("\n" + "=" * 70)
print("QLIKE RESULTS BY TRANSITION FUNCTION (lower is better)")
print("=" * 70)

trans_names = list(transition_configs.keys())
header = f"{'Asset':12s}" + "".join(f"{t:>14s}" for t in trans_names)
print(header)
print("-" * len(header))

for ticker in TEST_ASSETS:
    row = f"{ticker:12s}"
    for t in trans_names:
        if t in all_results[ticker]:
            q = all_results[ticker][t]["qlike"]
            row += f"{q:14.4f}"
        else:
            row += f"{'N/A':>14s}"
    print(row)

# Best per asset
print("-" * len(header))
print(f"\n{'Best transition per asset':}")
for ticker in TEST_ASSETS:
    results = all_results[ticker]
    if not results:
        continue
    best = min(results.items(), key=lambda x: x[1]["qlike"])
    worst = max(results.items(), key=lambda x: x[1]["qlike"])
    improvement = (worst[1]["qlike"] - best[1]["qlike"]) / abs(worst[1]["qlike"]) * 100
    print(f"  {ticker:12s}: BEST={best[0]:12s} ({best[1]['qlike']:.4f})  WORST={worst[0]:12s} ({worst[1]['qlike']:.4f})  gap={improvement:.2f}%")

# ============================================================
# 4. Key question: does the learned transition differ from logistic?
# ============================================================
print("\n" + "=" * 70)
print("KEY QUESTION: Does any non-logistic transition consistently win?")
print("=" * 70)

logistic_rank = []
for ticker in TEST_ASSETS:
    results = all_results[ticker]
    if not results:
        continue
    sorted_trans = sorted(results.items(), key=lambda x: x[1]["qlike"])
    rank = [t[0] for t in sorted_trans].index("logistic") + 1
    logistic_rank.append(rank)
    winner = sorted_trans[0][0]
    print(f"  {ticker:12s}: logistic rank={rank}/{len(sorted_trans)}, best={winner}")

mean_rank = np.mean(logistic_rank)
print(f"\n  Mean logistic rank: {mean_rank:.1f} / {len(trans_names)}")
if mean_rank <= 2:
    print("  Logistic is competitive — hard to beat with fancier transitions")
elif mean_rank >= 3:
    print("  Logistic is NOT the best — room for learned transition paper")

print("\n" + "=" * 70)
print("FEASIBILITY TEST COMPLETE")
print("=" * 70)
