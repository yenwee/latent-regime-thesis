#!/usr/bin/env python3
"""
Paper 2 Extended Feasibility: Forecast ALL 6 test assets.
Reuses the shared/PCA/asset-specific regimes from the first test.
Key question: does shared regime help more on noisy/less efficient assets?
"""
import os
import sys
import time
import numpy as np
import pandas as pd

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
from src.str_har import fit_str2_window_robust, str2_forecast_one, fit_har_ols, har_predict
from src.metrics import qlike
from sklearn.decomposition import PCA

np.random.seed(123)

TEST_ASSETS = ["^GSPC", "XLF", "XLE", "^TNX", "EURUSD=X", "GC=F"]
H = 5
ROLL_WINDOW = 600
DATA_START = "2015-01-01"
DATA_END = "2025-12-31"
SSM_EPOCHS = 400
SSM_PATIENCE = 40

print("=" * 70)
print("PAPER 2 EXTENDED FEASIBILITY: ALL 6 ASSETS")
print("=" * 70)

# ============================================================
# 1. Prepare data (same as before)
# ============================================================
print("\n[1/4] Preparing data...")
asset_dfs = {}
for ticker in TEST_ASSETS:
    df = download_asset_data(ticker, start=DATA_START, end=DATA_END)
    df = prepare_features(df, H, q_obs_smooth_span=10)
    asset_dfs[ticker] = df

common_idx = asset_dfs[TEST_ASSETS[0]].index
for ticker in TEST_ASSETS[1:]:
    common_idx = common_idx.intersection(asset_dfs[ticker].index)
common_idx = common_idx.sort_values()

for ticker in TEST_ASSETS:
    asset_dfs[ticker] = asset_dfs[ticker].loc[common_idx]

rv_panel = pd.DataFrame(
    {ticker: asset_dfs[ticker]["logv"] for ticker in TEST_ASSETS},
    index=common_idx
)
split_idx = ROLL_WINDOW
print(f"  Panel: {rv_panel.shape}, OOS start: {common_idx[split_idx].date()}")

# ============================================================
# 2. Train shared VRNN + baselines (same as before)
# ============================================================
print("\n[2/4] Training shared VRNN...")
train_panel = rv_panel.iloc[:split_idx]
mu_panel = train_panel.mean()
sd_panel = train_panel.std().replace(0, 1.0)
panel_std = (rv_panel - mu_panel) / sd_panel

X_train = torch.tensor(panel_std.iloc[:split_idx].values, dtype=torch.float32)
X_infer = torch.tensor(panel_std.values, dtype=torch.float32)

_, Z_shared, elbo = train_deep_ssm(
    X_train, X_infer, latent_dim=4, gru_hidden=32, dec_hidden=64,
    lr=0.001, epochs=SSM_EPOCHS, patience=SSM_PATIENCE, verbose=False,
)
mean_logv = rv_panel.mean(axis=1).values
q_shared = project_latent_to_scalar(Z_shared, mean_logv, split_idx)
print(f"  Shared VRNN: ELBO/T={elbo:.4f}")

# PCA
pca = PCA(n_components=1)
pca.fit(panel_std.iloc[:split_idx].values)
q_pca = pca.transform(panel_std.values).flatten()
if np.corrcoef(q_pca[:split_idx], mean_logv[:split_idx])[0, 1] < 0:
    q_pca = -q_pca

# Average RV
q_avg = panel_std.mean(axis=1).values

# Asset-specific VRNNs
print("\n[3/4] Training asset-specific VRNNs...")
q_asset_specific = {}
for ticker in TEST_ASSETS:
    asset_cols = ["x1_logv", "x2_absr", "x3_logvol"]
    df_a = asset_dfs[ticker]
    mu_a = df_a.iloc[:split_idx][asset_cols].mean()
    sd_a = df_a.iloc[:split_idx][asset_cols].std().replace(0, 1.0)
    X_a_train = torch.tensor(((df_a.iloc[:split_idx][asset_cols] - mu_a) / sd_a).values, dtype=torch.float32)
    X_a_infer = torch.tensor(((df_a[asset_cols] - mu_a) / sd_a).values, dtype=torch.float32)
    _, Z_a, _ = train_deep_ssm(
        X_a_train, X_a_infer, latent_dim=2, gru_hidden=16, dec_hidden=32,
        lr=0.002, epochs=SSM_EPOCHS, patience=SSM_PATIENCE, verbose=False,
    )
    q_asset_specific[ticker] = project_latent_to_scalar(Z_a, df_a["x1_logv"].values, split_idx)
    print(f"  {ticker} done")

# ============================================================
# 4. Forecast ALL assets
# ============================================================
print("\n[4/4] Forecasting all assets...")

all_results = {}

for ticker in TEST_ASSETS:
    df_test = asset_dfs[ticker].copy()
    df_test["q_shared"] = pd.Series(q_shared, index=common_idx).values
    df_test["q_pca"] = pd.Series(q_pca, index=common_idx).values
    df_test["q_avg"] = pd.Series(q_avg, index=common_idx).values
    df_test["q_asset"] = pd.Series(q_asset_specific[ticker], index=common_idx).values

    results = []
    for i in range(split_idx, len(df_test) - 1):
        if i - ROLL_WINDOW < 0:
            continue
        train_i = df_test.iloc[i - ROLL_WINDOW:i]
        test_i = df_test.iloc[i:i + 1]

        skip = False
        for qcol in ["q_shared", "q_pca", "q_avg", "q_asset", "q_obs"]:
            if train_i[qcol].isna().any() or test_i[qcol].isna().any():
                skip = True
                break
        if skip:
            continue

        b = fit_har_ols(train_i)
        yh_har = float(har_predict(b, test_i)[0])
        forecasts = {"har": yh_har}

        for qcol, label in [("q_obs", "str_obs"), ("q_shared", "str_shared"),
                             ("q_pca", "str_pca"), ("q_avg", "str_avg"),
                             ("q_asset", "str_asset")]:
            try:
                p, mu, sd = fit_str2_window_robust(
                    train_i, q_col=qcol, use_basinhopping=False, n_starts=2, gamma_max=12.0
                )
                yh, _ = str2_forecast_one(p, mu, sd, test_i, q_col=qcol)
                forecasts[label] = yh
            except Exception:
                forecasts[label] = np.nan

        forecasts["y"] = float(test_i["y"].values[0])
        results.append(forecasts)

    res_df = pd.DataFrame(results)
    all_results[ticker] = res_df
    print(f"  {ticker}: {len(res_df)} OOS forecasts")

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 70)
print("QLIKE RESULTS (lower is better)")
print("=" * 70)

models = ["har", "str_obs", "str_asset", "str_shared", "str_pca", "str_avg"]
header = f"{'Asset':12s}" + "".join(f"{m:>12s}" for m in models)
print(header)
print("-" * len(header))

asset_qlike = {}
for ticker in TEST_ASSETS:
    res_df = all_results[ticker]
    row = f"{ticker:12s}"
    asset_qlike[ticker] = {}
    for m in models:
        if m in res_df.columns and res_df[m].notna().any():
            q = float(qlike(res_df["y"], res_df[m]).mean())
            asset_qlike[ticker][m] = q
            row += f"{q:12.4f}"
        else:
            row += f"{'N/A':>12s}"
    print(row)

# Panel mean
print("-" * len(header))
row = f"{'MEAN':12s}"
for m in models:
    vals = [asset_qlike[t][m] for t in TEST_ASSETS if m in asset_qlike[t]]
    row += f"{np.mean(vals):12.4f}"
print(row)

# ============================================================
# Win rates: shared vs each alternative, per asset
# ============================================================
print("\n" + "=" * 70)
print("WIN RATES: shared beats ... (per-day QLIKE comparison)")
print("=" * 70)

comparisons = ["har", "str_obs", "str_asset", "str_pca", "str_avg"]
header = f"{'Asset':12s}" + "".join(f"{m:>12s}" for m in comparisons)
print(header)
print("-" * len(header))

for ticker in TEST_ASSETS:
    res_df = all_results[ticker]
    row = f"{ticker:12s}"
    for m in comparisons:
        if m in res_df.columns and "str_shared" in res_df.columns:
            wins = (qlike(res_df["y"], res_df["str_shared"]) < qlike(res_df["y"], res_df[m])).mean()
            row += f"{wins:11.1%} "
        else:
            row += f"{'N/A':>12s}"
    print(row)

# Panel mean win rates
print("-" * len(header))
row = f"{'MEAN':12s}"
for m in comparisons:
    win_rates = []
    for ticker in TEST_ASSETS:
        res_df = all_results[ticker]
        if m in res_df.columns and "str_shared" in res_df.columns:
            wins = (qlike(res_df["y"], res_df["str_shared"]) < qlike(res_df["y"], res_df[m])).mean()
            win_rates.append(wins)
    row += f"{np.mean(win_rates):11.1%} "
print(row)

# ============================================================
# Key finding: where does shared help most?
# ============================================================
print("\n" + "=" * 70)
print("KEY FINDING: Shared vs Asset-Specific improvement")
print("=" * 70)

for ticker in TEST_ASSETS:
    q_shared_val = asset_qlike[ticker].get("str_shared", np.nan)
    q_asset_val = asset_qlike[ticker].get("str_asset", np.nan)
    diff = q_asset_val - q_shared_val  # positive = shared is better
    direction = "SHARED better" if diff > 0 else "ASSET better"
    print(f"  {ticker:12s}: shared={q_shared_val:.4f}  asset={q_asset_val:.4f}  diff={diff:+.4f}  ({direction})")

print("\n" + "=" * 70)
print("FEASIBILITY TEST COMPLETE")
print("=" * 70)
