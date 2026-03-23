#!/usr/bin/env python3
"""
Paper 2 Alternative: Regime Uncertainty Quantification
Quick test: does the VRNN posterior uncertainty contain useful information?

Key insight: Paper 1 uses the posterior MEAN of z_t. But the posterior
VARIANCE tells us how confident the model is about the current regime.

Tests:
1. Does posterior uncertainty correlate with forecast errors?
2. Does incorporating uncertainty improve forecast intervals?
3. Can we build a "regime confidence" indicator?

Uses existing Paper 1 infrastructure — trains VRNN and extracts full posterior.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

PAPER1_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "paper1-latent-str")
sys.path.insert(0, PAPER1_ROOT)

import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from src.data import download_asset_data, prepare_features
from src.vrnn import DeepSSM, train_deep_ssm
from src.str_har import fit_str2_window_robust, str2_forecast_one, fit_har_ols, har_predict
from src.metrics import qlike

np.random.seed(123)

print("=" * 70)
print("PAPER 2 ALT: REGIME UNCERTAINTY QUANTIFICATION")
print("=" * 70)

# ============================================================
# 1. Train VRNN and extract FULL posterior (mean AND variance)
# ============================================================
print("\n[1/4] Training VRNN on GSPC and extracting posterior uncertainty...")

df = download_asset_data("^GSPC", start="2015-01-01", end="2025-12-31")
df = prepare_features(df, 5, q_obs_smooth_span=10)

X_cols = ["x1_logv", "x2_absr", "x3_logvol"]
split_idx = 600

mu_X = df.iloc[:split_idx][X_cols].mean()
sd_X = df.iloc[:split_idx][X_cols].std().replace(0, 1.0)

X_std = ((df[X_cols] - mu_X) / sd_X).values
X_train = torch.tensor(X_std[:split_idx], dtype=torch.float32)
X_infer = torch.tensor(X_std, dtype=torch.float32)

model, Z_mean, elbo = train_deep_ssm(
    X_train, X_infer, latent_dim=2, gru_hidden=16, dec_hidden=32,
    lr=0.002, epochs=600, patience=60, verbose=True,
)
print(f"  ELBO/T={elbo:.4f}")

# Extract posterior uncertainty (std of q(z|x))
model.eval()
with torch.no_grad():
    m, s = model.infer_q(X_infer)
    Z_mean_t = m.cpu().numpy()
    Z_std_t = s.cpu().numpy()  # THIS is the posterior uncertainty

# Compute scalar regime q_t and its uncertainty
y_logv = df["x1_logv"].values
Xw = np.column_stack([np.ones(split_idx)] + [Z_mean_t[:split_idx, d] for d in range(2)])
beta_w, *_ = np.linalg.lstsq(Xw, y_logv[:split_idx], rcond=None)
w = beta_w[1:]

q_mean = Z_mean_t @ w  # regime point estimate (what Paper 1 uses)
# Regime uncertainty: propagate posterior std through linear projection
q_std = np.sqrt(np.sum((Z_std_t * w[None, :]) ** 2, axis=1))  # sqrt(w' diag(s^2) w)

if np.corrcoef(q_mean[:split_idx], y_logv[:split_idx])[0, 1] < 0:
    q_mean = -q_mean

# Add to dataframe
df["q_ssm"] = q_mean
df["q_ssm_std"] = q_std
df["q_ssm_std_rank"] = pd.Series(q_std, index=df.index).rank(pct=True)

print(f"\n  Regime q_t: mean={np.mean(q_mean):.3f}, std={np.std(q_mean):.3f}")
print(f"  Uncertainty: mean={np.mean(q_std):.4f}, std={np.std(q_std):.4f}")
print(f"  Corr(|q|, q_std): {np.corrcoef(np.abs(q_mean), q_std)[0,1]:.3f}")

# ============================================================
# 2. Does uncertainty predict forecast errors?
# ============================================================
print("\n[2/4] Uncertainty vs forecast errors...")

# Rolling forecast
results = []
for i in range(split_idx, len(df) - 1):
    if i - 600 < 0:
        continue
    train_i = df.iloc[i - 600:i].copy()
    test_i = df.iloc[i:i + 1].copy()

    if train_i["q_ssm"].isna().any() or test_i["q_ssm"].isna().any():
        continue

    b = fit_har_ols(train_i)
    yh_har = float(har_predict(b, test_i)[0])

    try:
        p, mu, sd = fit_str2_window_robust(
            train_i, q_col="q_ssm", use_basinhopping=False, n_starts=2, gamma_max=12.0
        )
        yh_ssm, g = str2_forecast_one(p, mu, sd, test_i, q_col="q_ssm")
    except Exception:
        continue

    results.append({
        "date": test_i.index[0],
        "y": float(test_i["y"].values[0]),
        "har": yh_har,
        "str_ssm": yh_ssm,
        "G_ssm": g,
        "q_std": float(test_i["q_ssm_std"].values[0]),
        "q_std_rank": float(test_i["q_ssm_std_rank"].values[0]),
    })

res_df = pd.DataFrame(results).set_index("date")
res_df["error_ssm"] = np.abs(res_df["y"] - res_df["str_ssm"])
res_df["error_har"] = np.abs(res_df["y"] - res_df["har"])
res_df["qlike_ssm"] = qlike(res_df["y"], res_df["str_ssm"])
res_df["qlike_har"] = qlike(res_df["y"], res_df["har"])
res_df["ssm_better"] = (res_df["qlike_ssm"] < res_df["qlike_har"]).astype(int)

print(f"  OOS forecasts: {len(res_df)}")

# Correlation: uncertainty vs absolute forecast error
corr_err, p_err = spearmanr(res_df["q_std"], res_df["error_ssm"])
print(f"\n  Spearman corr(uncertainty, |error|): {corr_err:.3f} (p={p_err:.4f})")
if p_err < 0.05 and corr_err > 0:
    print("  GOOD: Higher uncertainty → larger errors (uncertainty is informative)")
elif p_err < 0.05 and corr_err < 0:
    print("  INTERESTING: Higher uncertainty → smaller errors (model is cautious when uncertain)")
else:
    print("  NOT significant")

# Does SSM advantage grow when regime is certain?
high_conf = res_df["q_std_rank"] <= 0.25  # most certain 25%
low_conf = res_df["q_std_rank"] >= 0.75   # most uncertain 25%

qlike_ssm_certain = res_df.loc[high_conf, "qlike_ssm"].mean()
qlike_har_certain = res_df.loc[high_conf, "qlike_har"].mean()
qlike_ssm_uncertain = res_df.loc[low_conf, "qlike_ssm"].mean()
qlike_har_uncertain = res_df.loc[low_conf, "qlike_har"].mean()

win_certain = res_df.loc[high_conf, "ssm_better"].mean()
win_uncertain = res_df.loc[low_conf, "ssm_better"].mean()

print(f"\n  When regime is CERTAIN (low uncertainty, bottom 25%):")
print(f"    QLIKE SSM={qlike_ssm_certain:.4f}  HAR={qlike_har_certain:.4f}")
print(f"    SSM win rate: {win_certain:.1%}")

print(f"  When regime is UNCERTAIN (high uncertainty, top 25%):")
print(f"    QLIKE SSM={qlike_ssm_uncertain:.4f}  HAR={qlike_har_uncertain:.4f}")
print(f"    SSM win rate: {win_uncertain:.1%}")

if win_certain > win_uncertain:
    print(f"\n  KEY FINDING: SSM advantage is LARGER when regime is certain")
    print(f"  Difference: {win_certain - win_uncertain:+.1%}")
    print(f"  This means: regime confidence is actionable information")
else:
    print(f"\n  SSM advantage does NOT depend on regime certainty")

# ============================================================
# 3. Uncertainty-weighted forecast combination
# ============================================================
print("\n[3/4] Uncertainty-weighted forecast combination...")

# Idea: when regime is uncertain, trust HAR more; when certain, trust STR-SSM more
# weight_ssm = 1 - normalized_uncertainty
q_std_norm = (res_df["q_std"] - res_df["q_std"].min()) / (res_df["q_std"].max() - res_df["q_std"].min())
weight_ssm = 1.0 - q_std_norm  # high confidence → high weight on SSM

yhat_combined = weight_ssm * res_df["str_ssm"] + (1 - weight_ssm) * res_df["har"]
qlike_combined = qlike(res_df["y"], yhat_combined).mean()
qlike_ssm_mean = res_df["qlike_ssm"].mean()
qlike_har_mean = res_df["qlike_har"].mean()

# Fixed 50/50 combination for comparison
yhat_equal = 0.5 * res_df["str_ssm"] + 0.5 * res_df["har"]
qlike_equal = qlike(res_df["y"], yhat_equal).mean()

print(f"  HAR only:              {qlike_har_mean:.4f}")
print(f"  STR-SSM only:          {qlike_ssm_mean:.4f}")
print(f"  Equal combination:     {qlike_equal:.4f}")
print(f"  Uncertainty-weighted:  {qlike_combined:.4f}")

best = min(qlike_har_mean, qlike_ssm_mean, qlike_equal, qlike_combined)
if qlike_combined == best:
    print(f"  BEST: Uncertainty-weighted combination wins!")
elif qlike_equal == best:
    print(f"  Equal combination is best (uncertainty weighting doesn't help)")
else:
    print(f"  Single model is best (combination doesn't help)")

# ============================================================
# 4. Temporal analysis: when is uncertainty high?
# ============================================================
print("\n[4/4] Temporal uncertainty patterns...")

# Rolling 20-day mean uncertainty
res_df["q_std_rolling"] = res_df["q_std"].rolling(20).mean()

# Find top-5 uncertainty spikes
top_uncertain = res_df.nlargest(5, "q_std")
print(f"  Top 5 most uncertain days:")
for _, row in top_uncertain.iterrows():
    print(f"    {row.name.date()}: q_std={row['q_std']:.4f}, G_ssm={row['G_ssm']:.3f}")

# Uncertainty during known events
events = [
    ("COVID onset", "2020-02-20", "2020-04-01"),
    ("Rate hikes 2022", "2022-09-01", "2022-11-30"),
    ("SVB crisis", "2023-03-01", "2023-04-15"),
    ("2024 election", "2024-10-15", "2024-11-15"),
]
print(f"\n  Event-conditional uncertainty:")
for event_name, start, end in events:
    mask = (res_df.index >= pd.Timestamp(start)) & (res_df.index <= pd.Timestamp(end))
    if mask.sum() == 0:
        continue
    event_std = res_df.loc[mask, "q_std"].mean()
    overall_std = res_df["q_std"].mean()
    ratio = event_std / overall_std
    print(f"    {event_name:20s}: mean_uncertainty={event_std:.4f} ({ratio:.2f}x baseline)")

print("\n" + "=" * 70)
print("FEASIBILITY TEST COMPLETE")
print("=" * 70)
