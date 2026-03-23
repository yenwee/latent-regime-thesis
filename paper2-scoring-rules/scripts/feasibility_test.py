#!/usr/bin/env python3
"""
Paper 2 Feasibility Test: Does a multivariate VRNN shared regime
capture something different from PCA Factor 1?

Quick test with 6 diverse assets, H=5 only:
1. Download & prepare data for 6 assets
2. Stack realized volatilities into multivariate input
3. Train shared VRNN on joint RV panel
4. Extract shared latent regime q_shared
5. Compare with:
   - PCA Factor 1 of RV panel
   - Asset-specific latent q_t from Paper 1's VRNN
   - Cross-asset average RV
6. Run STR-HAR with each transition variable, compare QLIKE

Expected runtime: ~5 minutes
"""
import os
import sys
import time
import numpy as np
import pandas as pd

# Paper 1 code reuse
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

# ============================================================
# CONFIG
# ============================================================
TEST_ASSETS = ["^GSPC", "XLF", "XLE", "^TNX", "EURUSD=X", "GC=F"]  # 6 diverse assets
H = 5
ROLL_WINDOW = 600
DATA_START = "2015-01-01"
DATA_END = "2025-12-31"
SSM_EPOCHS = 400
SSM_PATIENCE = 40
LATENT_DIM_SHARED = 4
LATENT_DIM_ASSET = 2

print("=" * 60)
print("PAPER 2 FEASIBILITY TEST")
print("=" * 60)
print(f"Assets: {TEST_ASSETS}")
print(f"H={H}, window={ROLL_WINDOW}")

# ============================================================
# 1. Download & prepare data
# ============================================================
print("\n[1/6] Downloading data...")
t0 = time.perf_counter()

asset_dfs = {}
for ticker in TEST_ASSETS:
    df = download_asset_data(ticker, start=DATA_START, end=DATA_END)
    df = prepare_features(df, H, q_obs_smooth_span=10)
    asset_dfs[ticker] = df
    print(f"  {ticker}: {len(df)} days")

# Find common date range
common_idx = asset_dfs[TEST_ASSETS[0]].index
for ticker in TEST_ASSETS[1:]:
    common_idx = common_idx.intersection(asset_dfs[ticker].index)
common_idx = common_idx.sort_values()
print(f"  Common dates: {len(common_idx)} ({common_idx[0].date()} to {common_idx[-1].date()})")

# Align all assets to common dates
for ticker in TEST_ASSETS:
    asset_dfs[ticker] = asset_dfs[ticker].loc[common_idx]

dt = time.perf_counter() - t0
print(f"  Done ({dt:.1f}s)")

# ============================================================
# 2. Build multivariate RV panel
# ============================================================
print("\n[2/6] Building RV panel...")

# Stack log realized volatilities: T x N
rv_panel = pd.DataFrame(
    {ticker: asset_dfs[ticker]["logv"] for ticker in TEST_ASSETS},
    index=common_idx
)
print(f"  Panel shape: {rv_panel.shape}")

# Train/OOS split
split_idx = ROLL_WINDOW
oos_start = common_idx[split_idx]
print(f"  OOS start: {oos_start.date()}")

# ============================================================
# 3. Train shared multivariate VRNN
# ============================================================
print("\n[3/6] Training shared VRNN on multivariate RV panel...")
t0 = time.perf_counter()

# Standardize using training data
train_panel = rv_panel.iloc[:split_idx]
mu_panel = train_panel.mean()
sd_panel = train_panel.std().replace(0, 1.0)
panel_std = (rv_panel - mu_panel) / sd_panel

X_train = torch.tensor(panel_std.iloc[:split_idx].values, dtype=torch.float32)
X_infer = torch.tensor(panel_std.values, dtype=torch.float32)

_, Z_shared, elbo_shared = train_deep_ssm(
    X_train, X_infer,
    latent_dim=LATENT_DIM_SHARED,
    gru_hidden=32, dec_hidden=64,
    lr=0.001, weight_decay=1e-4,
    epochs=SSM_EPOCHS, patience=SSM_PATIENCE,
    verbose=True,
)

# Project to scalar: supervised on mean log RV
mean_logv = rv_panel.mean(axis=1).values
q_shared = project_latent_to_scalar(Z_shared, mean_logv, split_idx)

dt = time.perf_counter() - t0
print(f"  Shared VRNN: latent_dim={LATENT_DIM_SHARED}, ELBO/T={elbo_shared:.4f} ({dt:.1f}s)")

# ============================================================
# 4. Build comparison transition variables
# ============================================================
print("\n[4/6] Building comparison transition variables...")

# PCA Factor 1
pca = PCA(n_components=1)
pca.fit(panel_std.iloc[:split_idx].values)
q_pca = pca.transform(panel_std.values).flatten()
# Ensure positive correlation with mean logv
if np.corrcoef(q_pca[:split_idx], mean_logv[:split_idx])[0, 1] < 0:
    q_pca = -q_pca
print(f"  PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}")

# Cross-asset average RV (standardized)
q_avg = panel_std.mean(axis=1).values

# Asset-specific VRNN (train per asset, like Paper 1)
q_asset_specific = {}
for ticker in TEST_ASSETS:
    asset_cols = ["x1_logv", "x2_absr", "x3_logvol"]
    df_asset = asset_dfs[ticker]

    mu_a = df_asset.iloc[:split_idx][asset_cols].mean()
    sd_a = df_asset.iloc[:split_idx][asset_cols].std().replace(0, 1.0)
    X_a_train = torch.tensor(((df_asset.iloc[:split_idx][asset_cols] - mu_a) / sd_a).values, dtype=torch.float32)
    X_a_infer = torch.tensor(((df_asset[asset_cols] - mu_a) / sd_a).values, dtype=torch.float32)

    _, Z_a, _ = train_deep_ssm(
        X_a_train, X_a_infer,
        latent_dim=LATENT_DIM_ASSET, gru_hidden=16, dec_hidden=32,
        lr=0.002, epochs=SSM_EPOCHS, patience=SSM_PATIENCE, verbose=False,
    )
    q_a = project_latent_to_scalar(Z_a, df_asset["x1_logv"].values, split_idx)
    q_asset_specific[ticker] = q_a
    print(f"  {ticker} asset-specific VRNN done")

# ============================================================
# 5. Correlation analysis
# ============================================================
print("\n[5/6] Correlation analysis (OOS period)...")

oos_slice = slice(split_idx, None)
print(f"\n  Correlation matrix of transition variables (OOS):")
corr_df = pd.DataFrame({
    "q_shared": q_shared[oos_slice],
    "q_pca": q_pca[oos_slice],
    "q_avg": q_avg[oos_slice],
}, index=common_idx[oos_slice])

# Add asset-specific for first asset
corr_df["q_asset_GSPC"] = q_asset_specific["^GSPC"][oos_slice]

corr_matrix = corr_df.corr()
print(corr_matrix.to_string(float_format=lambda x: f"{x:.3f}"))

# Key question: how different is shared from PCA?
shared_pca_corr = np.corrcoef(q_shared[oos_slice], q_pca[oos_slice])[0, 1]
print(f"\n  KEY METRIC: corr(shared, PCA) = {shared_pca_corr:.3f}")
if abs(shared_pca_corr) > 0.95:
    print("  WARNING: Shared regime is nearly identical to PCA Factor 1")
    print("  Paper 2's contribution may be thin")
elif abs(shared_pca_corr) > 0.80:
    print("  Moderate overlap with PCA — shared captures some unique dynamics")
elif abs(shared_pca_corr) < 0.80:
    print("  GOOD: Shared regime captures substantially different dynamics than PCA")

# ============================================================
# 6. Forecasting comparison (rolling OOS on one asset)
# ============================================================
print("\n[6/6] Forecasting comparison (^GSPC, rolling OOS)...")

test_ticker = "^GSPC"
df_test = asset_dfs[test_ticker].copy()
df_test["q_shared"] = pd.Series(q_shared, index=common_idx).values
df_test["q_pca"] = pd.Series(q_pca, index=common_idx).values
df_test["q_avg"] = pd.Series(q_avg, index=common_idx).values
df_test["q_asset"] = pd.Series(q_asset_specific[test_ticker], index=common_idx).values

results = []
n_oos = 0
for i in range(split_idx, len(df_test) - 1):
    if i - ROLL_WINDOW < 0:
        continue

    train_i = df_test.iloc[i - ROLL_WINDOW:i]
    test_i = df_test.iloc[i:i + 1]

    # Skip if any transition variable is NaN
    skip = False
    for qcol in ["q_shared", "q_pca", "q_avg", "q_asset", "q_obs"]:
        if train_i[qcol].isna().any() or test_i[qcol].isna().any():
            skip = True
            break
    if skip:
        continue

    # HAR baseline
    b = fit_har_ols(train_i)
    yh_har = float(har_predict(b, test_i)[0])

    # STR-HAR with each transition variable
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
    forecasts["date"] = test_i.index[0]
    results.append(forecasts)
    n_oos += 1

res_df = pd.DataFrame(results).set_index("date")
print(f"  OOS forecasts: {n_oos}")

# QLIKE comparison
print(f"\n  QLIKE (lower is better):")
models = ["har", "str_obs", "str_asset", "str_shared", "str_pca", "str_avg"]
for m in models:
    if m in res_df.columns and res_df[m].notna().any():
        q = float(qlike(res_df["y"], res_df[m]).mean())
        tag = ""
        if m == "str_shared":
            tag = " <-- Paper 2"
        elif m == "str_asset":
            tag = " <-- Paper 1"
        print(f"    {m:15s}: {q:.4f}{tag}")

# Win rates: shared vs others
print(f"\n  Win rates (shared vs others, lower QLIKE wins):")
for m in ["har", "str_obs", "str_asset", "str_pca", "str_avg"]:
    if m in res_df.columns and res_df[m].notna().any():
        wins = (qlike(res_df["y"], res_df["str_shared"]) < qlike(res_df["y"], res_df[m])).mean()
        print(f"    shared beats {m:15s}: {wins:.1%}")

print("\n" + "=" * 60)
print("FEASIBILITY TEST COMPLETE")
print("=" * 60)
