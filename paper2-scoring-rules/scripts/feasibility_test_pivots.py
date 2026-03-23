#!/usr/bin/env python3
"""
Paper 2 Pivot Tests:
  Test 1: Stress-conditional — does shared beat asset-specific during crises?
  Test 2: Per-asset projection — each asset gets own w_i'z_t from shared Z

Reuses same 6-asset panel and shared VRNN from feasibility test.
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
from src.vrnn import train_deep_ssm
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

# Crisis periods (approximate)
CRISIS_PERIODS = [
    ("COVID", "2020-02-20", "2020-06-30"),
    ("Rate Shock 2022", "2022-08-01", "2022-12-31"),
    ("SVB Crisis", "2023-03-01", "2023-05-31"),
]

print("=" * 70)
print("PAPER 2 PIVOT TESTS")
print("=" * 70)

# ============================================================
# 1. Prepare data + train models (same as before)
# ============================================================
print("\n[1/5] Preparing data and training models...")
t0 = time.perf_counter()

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

# Train shared VRNN
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
print(f"  Shared VRNN: ELBO/T={elbo:.4f}")

# Original shared projection (onto mean logv)
mean_logv = rv_panel.mean(axis=1).values
Z_train_shared = Z_shared[:split_idx]
Z_mean_s = Z_train_shared.mean(axis=0, keepdims=True)
Z_std_s = Z_train_shared.std(axis=0, keepdims=True)
Z_std_s[Z_std_s < 1e-9] = 1.0
Zs_shared = (Z_shared - Z_mean_s) / Z_std_s

# Projection weights for mean logv
Xw_mean = np.column_stack([np.ones(split_idx)] + [Zs_shared[:split_idx, d] for d in range(4)])
beta_mean, *_ = np.linalg.lstsq(Xw_mean, mean_logv[:split_idx], rcond=None)
w_mean = beta_mean[1:]
q_shared_mean = Zs_shared @ w_mean
if np.corrcoef(q_shared_mean[:split_idx], mean_logv[:split_idx])[0, 1] < 0:
    q_shared_mean = -q_shared_mean

# ============================================================
# TEST 2: Per-asset projection from shared Z_t
# ============================================================
print("\n[2/5] Test 2: Per-asset projection (w_i'z_t per asset)...")

q_shared_per_asset = {}
for ticker in TEST_ASSETS:
    # Project shared Z onto THIS asset's log volatility
    y_asset = asset_dfs[ticker]["x1_logv"].values
    Xw_a = np.column_stack([np.ones(split_idx)] + [Zs_shared[:split_idx, d] for d in range(4)])
    beta_a, *_ = np.linalg.lstsq(Xw_a, y_asset[:split_idx], rcond=None)
    w_a = beta_a[1:]
    q_a = Zs_shared @ w_a
    if np.corrcoef(q_a[:split_idx], y_asset[:split_idx])[0, 1] < 0:
        q_a = -q_a
    q_shared_per_asset[ticker] = q_a
    corr_with_mean = np.corrcoef(q_a[split_idx:], q_shared_mean[split_idx:])[0, 1]
    print(f"  {ticker}: corr(per-asset proj, mean proj) = {corr_with_mean:.3f}")

# Train asset-specific VRNNs
print("\n[3/5] Training asset-specific VRNNs...")
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
    Z_a_train = Z_a[:split_idx]
    Z_a_mean = Z_a_train.mean(axis=0, keepdims=True)
    Z_a_std = Z_a_train.std(axis=0, keepdims=True)
    Z_a_std[Z_a_std < 1e-9] = 1.0
    Zs_a = (Z_a - Z_a_mean) / Z_a_std
    y_a = df_a["x1_logv"].values
    Xw_aa = np.column_stack([np.ones(split_idx)] + [Zs_a[:split_idx, d] for d in range(2)])
    beta_aa, *_ = np.linalg.lstsq(Xw_aa, y_a[:split_idx], rcond=None)
    w_aa = beta_aa[1:]
    q_aa = Zs_a @ w_aa
    if np.corrcoef(q_aa[:split_idx], y_a[:split_idx])[0, 1] < 0:
        q_aa = -q_aa
    q_asset_specific[ticker] = q_aa
    print(f"  {ticker} done")

# ============================================================
# 4. Forecast with all transition variables
# ============================================================
print("\n[4/5] Forecasting all assets with all transition variables...")

all_results = {}
for ticker in TEST_ASSETS:
    df_test = asset_dfs[ticker].copy()
    df_test["q_shared_mean"] = pd.Series(q_shared_mean, index=common_idx).values
    df_test["q_shared_asset"] = pd.Series(q_shared_per_asset[ticker], index=common_idx).values
    df_test["q_asset"] = pd.Series(q_asset_specific[ticker], index=common_idx).values

    results = []
    for i in range(split_idx, len(df_test) - 1):
        if i - ROLL_WINDOW < 0:
            continue
        train_i = df_test.iloc[i - ROLL_WINDOW:i]
        test_i = df_test.iloc[i:i + 1]

        skip = False
        for qcol in ["q_shared_mean", "q_shared_asset", "q_asset", "q_obs"]:
            if train_i[qcol].isna().any() or test_i[qcol].isna().any():
                skip = True
                break
        if skip:
            continue

        b = fit_har_ols(train_i)
        yh_har = float(har_predict(b, test_i)[0])
        forecasts = {"har": yh_har, "date": test_i.index[0]}

        for qcol, label in [("q_obs", "str_obs"),
                             ("q_shared_mean", "str_shared_mean"),
                             ("q_shared_asset", "str_shared_perasset"),
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

    all_results[ticker] = pd.DataFrame(results).set_index("date")
    print(f"  {ticker}: {len(results)} forecasts")

# ============================================================
# 5. Results
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: QLIKE WITH PER-ASSET PROJECTION (lower is better)")
print("=" * 70)

models = ["har", "str_obs", "str_asset", "str_shared_mean", "str_shared_perasset"]
labels = ["HAR", "STR-OBS", "Asset VRNN", "Shared(mean)", "Shared(per-asset)"]
header = f"{'Asset':12s}" + "".join(f"{l:>16s}" for l in labels)
print(header)
print("-" * len(header))

for ticker in TEST_ASSETS:
    res_df = all_results[ticker]
    row = f"{ticker:12s}"
    for m in models:
        if m in res_df.columns and res_df[m].notna().any():
            q = float(qlike(res_df["y"], res_df[m]).mean())
            row += f"{q:16.4f}"
        else:
            row += f"{'N/A':>16s}"
    print(row)

# Per-asset projection vs asset-specific
print("\n" + "=" * 70)
print("KEY: Shared(per-asset) vs Asset-Specific")
print("=" * 70)
for ticker in TEST_ASSETS:
    res_df = all_results[ticker]
    q_sp = float(qlike(res_df["y"], res_df["str_shared_perasset"]).mean())
    q_as = float(qlike(res_df["y"], res_df["str_asset"]).mean())
    diff = q_as - q_sp
    winner = "SHARED(per-asset) better" if diff > 0 else "ASSET better"
    print(f"  {ticker:12s}: diff={diff:+.4f}  ({winner})")

# ============================================================
# TEST 1: Stress-conditional analysis
# ============================================================
print("\n" + "=" * 70)
print("TEST 1: STRESS-CONDITIONAL ANALYSIS")
print("=" * 70)

for period_name, start, end in CRISIS_PERIODS:
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end)
    print(f"\n  --- {period_name} ({start} to {end}) ---")

    for ticker in TEST_ASSETS:
        res_df = all_results[ticker]
        mask = (res_df.index >= start_dt) & (res_df.index <= end_dt)
        crisis_df = res_df.loc[mask]

        if len(crisis_df) < 10:
            continue

        q_shared_pa = float(qlike(crisis_df["y"], crisis_df["str_shared_perasset"]).mean())
        q_asset = float(qlike(crisis_df["y"], crisis_df["str_asset"]).mean())
        q_har = float(qlike(crisis_df["y"], crisis_df["har"]).mean())
        diff = q_asset - q_shared_pa
        winner = "SHARED" if diff > 0 else "ASSET"
        print(f"    {ticker:12s}: shared={q_shared_pa:.4f}  asset={q_asset:.4f}  diff={diff:+.4f}  ({winner})  [n={len(crisis_df)}]")

# Calm periods (everything NOT crisis)
print(f"\n  --- CALM periods (non-crisis) ---")
for ticker in TEST_ASSETS:
    res_df = all_results[ticker]
    crisis_mask = pd.Series(False, index=res_df.index)
    for _, start, end in CRISIS_PERIODS:
        crisis_mask |= (res_df.index >= pd.Timestamp(start)) & (res_df.index <= pd.Timestamp(end))
    calm_df = res_df.loc[~crisis_mask]

    if len(calm_df) < 10:
        continue

    q_shared_pa = float(qlike(calm_df["y"], calm_df["str_shared_perasset"]).mean())
    q_asset = float(qlike(calm_df["y"], calm_df["str_asset"]).mean())
    diff = q_asset - q_shared_pa
    winner = "SHARED" if diff > 0 else "ASSET"
    print(f"    {ticker:12s}: shared={q_shared_pa:.4f}  asset={q_asset:.4f}  diff={diff:+.4f}  ({winner})  [n={len(calm_df)}]")

print("\n" + "=" * 70)
print("PIVOT TESTS COMPLETE")
print("=" * 70)
