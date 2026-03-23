#!/usr/bin/env python3
"""
Paper 2 Alternative: Regime-Dependent Tail Risk
Quick feasibility test using Paper 1's existing GSPC H=5 results.

Key question: Do latent regimes predict tail behavior beyond volatility levels?

Tests:
1. Regime-conditional distributions: are tails heavier in high-regime periods?
2. Regime-dependent nu_t: does regime-varying Student-t df improve distributional forecasts?
3. CRPS comparison: fixed-nu vs regime-dependent nu
4. Quantile coverage: does regime-dependent model improve tail calibration?

Uses existing results — no new model training needed.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, ks_2samp, norm
from scipy.optimize import minimize

PAPER1_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "paper1-latent-str")
sys.path.insert(0, PAPER1_ROOT)

from src.metrics import qlike

np.random.seed(123)

print("=" * 70)
print("PAPER 2 ALT: REGIME-DEPENDENT TAIL RISK FEASIBILITY")
print("=" * 70)

# ============================================================
# 1. Load existing Paper 1 results
# ============================================================
print("\n[1/5] Loading Paper 1 GSPC H=5 results...")

results_path = os.path.join(PAPER1_ROOT, "outputs", "test_quick", "H5", "GSPC_H5_results.csv")
df = pd.read_csv(results_path, index_col=0, parse_dates=True)
print(f"  OOS period: {df.index[0].date()} to {df.index[-1].date()} ({len(df)} days)")
print(f"  Columns: {list(df.columns)}")

# Key variables
y = df["y"].values          # log realized variance (target)
rH = df["rH"].values        # H-day forward return
yhat_ssm = df["str_ssm"].values  # STR-SSM forecast
G_ssm = df["G_ssm"].values  # regime transition weight (0=calm, 1=stress)

# Standardized residuals: eps = rH / sigma_forecast
sigma_H = np.sqrt(5 * np.exp(yhat_ssm))  # H-day vol forecast
eps = rH / (sigma_H + 1e-12)
eps_clean = eps[np.isfinite(eps)]

print(f"  Mean G_ssm: {np.nanmean(G_ssm):.3f}, Std: {np.nanstd(G_ssm):.3f}")
print(f"  Residuals: mean={np.nanmean(eps_clean):.3f}, std={np.nanstd(eps_clean):.3f}, kurtosis={pd.Series(eps_clean).kurtosis():.2f}")

# ============================================================
# 2. Regime-conditional distribution analysis
# ============================================================
print("\n[2/5] Regime-conditional distributions...")

# Split into high/low regime
threshold_high = np.nanpercentile(G_ssm, 75)
threshold_low = np.nanpercentile(G_ssm, 25)

mask_high = G_ssm >= threshold_high
mask_low = G_ssm <= threshold_low
mask_mid = (~mask_high) & (~mask_low)

eps_high = eps[mask_high & np.isfinite(eps)]
eps_low = eps[mask_low & np.isfinite(eps)]
eps_mid = eps[mask_mid & np.isfinite(eps)]

print(f"  High regime (G>={threshold_high:.2f}): n={len(eps_high)}")
print(f"  Low regime  (G<={threshold_low:.2f}): n={len(eps_low)}")
print(f"  Mid regime: n={len(eps_mid)}")

# Tail statistics per regime
for label, e in [("High", eps_high), ("Low", eps_low), ("Mid", eps_mid)]:
    kurt = pd.Series(e).kurtosis()
    skew = pd.Series(e).skew()
    tail_5 = np.mean(np.abs(e) > 1.96)  # fraction beyond 2 sigma
    tail_1 = np.mean(np.abs(e) > 2.576)  # fraction beyond ~1%
    print(f"  {label:5s}: kurtosis={kurt:6.2f}  skew={skew:+.2f}  |eps|>2sig={tail_5:.3f}  |eps|>2.6sig={tail_1:.3f}")

# KS test: are high-regime and low-regime distributions different?
ks_stat, ks_pval = ks_2samp(eps_high, eps_low)
print(f"\n  KS test (high vs low): stat={ks_stat:.3f}, p={ks_pval:.4f}")
if ks_pval < 0.05:
    print("  SIGNIFICANT: High and low regime distributions ARE different")
else:
    print("  NOT significant: distributions are similar")

# ============================================================
# 3. Fit regime-dependent nu_t
# ============================================================
print("\n[3/5] Fitting regime-dependent Student-t df...")

def fit_nu_fixed(eps):
    """Fit single fixed nu by MLE."""
    eps = eps[np.isfinite(eps)]
    def nll(log_nu):
        nu = 2.05 + np.exp(log_nu)
        s = np.sqrt(nu / (nu - 2))
        return -np.sum(student_t.logpdf(eps, df=nu, scale=1/s))
    res = minimize(nll, np.log(6.0), method='L-BFGS-B')
    return 2.05 + np.exp(res.x[0])

def fit_nu_regime_dependent(eps, G):
    """Fit nu = f(G): nu_t = nu_base + delta * G_t (more flexible tails when G is high)."""
    mask = np.isfinite(eps) & np.isfinite(G)
    eps_c = eps[mask]
    G_c = G[mask]

    def nll(params):
        log_nu_base, delta = params
        nu_base = 2.05 + np.exp(log_nu_base)
        # nu_t decreases (heavier tails) when G is high
        nu_t = np.maximum(nu_base - delta * G_c, 2.1)
        total = 0.0
        for i in range(len(eps_c)):
            s = np.sqrt(nu_t[i] / (nu_t[i] - 2))
            total -= student_t.logpdf(eps_c[i], df=nu_t[i], scale=1/s)
        return total

    res = minimize(nll, [np.log(6.0), 2.0], method='L-BFGS-B',
                   bounds=[(np.log(0.5), np.log(50)), (-10, 20)])
    nu_base = 2.05 + np.exp(res.x[0])
    delta = res.x[1]
    return nu_base, delta, res.fun

# Fixed nu
nu_fixed = fit_nu_fixed(eps)
print(f"  Fixed nu: {nu_fixed:.2f}")

# Per-regime nu
nu_high = fit_nu_fixed(eps_high)
nu_low = fit_nu_fixed(eps_low)
nu_mid = fit_nu_fixed(eps_mid)
print(f"  Per-regime: nu_high={nu_high:.2f}, nu_mid={nu_mid:.2f}, nu_low={nu_low:.2f}")

if nu_high < nu_low:
    print("  GOOD: High-regime has lower nu (heavier tails) as expected")
else:
    print("  UNEXPECTED: High-regime has higher nu (lighter tails)")

# Regime-dependent nu_t = nu_base - delta * G_t
nu_base, delta, nll_regime = fit_nu_regime_dependent(eps, G_ssm)
print(f"  Regime-dependent: nu_base={nu_base:.2f}, delta={delta:.2f}")
print(f"    -> nu when G=0 (calm): {max(nu_base, 2.1):.2f}")
print(f"    -> nu when G=1 (stress): {max(nu_base - delta, 2.1):.2f}")

# ============================================================
# 4. CRPS comparison
# ============================================================
print("\n[4/5] CRPS comparison (lower is better)...")

def crps_t(y, mu, sigma, nu, n_samples=500):
    """Monte Carlo CRPS for Student-t forecast."""
    s = np.sqrt(nu / (nu - 2))
    samples = student_t.rvs(df=nu, scale=sigma/s, size=n_samples) + mu
    crps = np.mean(np.abs(samples - y)) - 0.5 * np.mean(np.abs(samples[:, None] - samples[None, :]))
    return crps

mask = np.isfinite(rH) & np.isfinite(G_ssm) & np.isfinite(yhat_ssm)
rH_c = rH[mask]
sigma_c = sigma_H[mask]
G_c = G_ssm[mask]

# Fixed nu CRPS
crps_fixed = []
for i in range(len(rH_c)):
    crps_fixed.append(crps_t(rH_c[i], 0, sigma_c[i], nu_fixed))
crps_fixed = np.array(crps_fixed)

# Regime-dependent nu CRPS
crps_regime = []
for i in range(len(rH_c)):
    nu_i = max(nu_base - delta * G_c[i], 2.1)
    crps_regime.append(crps_t(rH_c[i], 0, sigma_c[i], nu_i))
crps_regime = np.array(crps_regime)

# Normal distribution CRPS (baseline)
crps_normal = []
for i in range(len(rH_c)):
    crps_normal.append(np.abs(rH_c[i]) * (2 * norm.cdf(np.abs(rH_c[i]) / sigma_c[i]) - 1) +
                        sigma_c[i] * 2 * norm.pdf(rH_c[i] / sigma_c[i]) - sigma_c[i] / np.sqrt(np.pi))
crps_normal = np.array(crps_normal)

print(f"  Normal:            {np.mean(crps_normal):.6f}")
print(f"  Student-t (fixed): {np.mean(crps_fixed):.6f}")
print(f"  Student-t (regime):{np.mean(crps_regime):.6f}")

improvement = (np.mean(crps_fixed) - np.mean(crps_regime)) / np.mean(crps_fixed) * 100
print(f"  Regime vs fixed improvement: {improvement:+.2f}%")

# Per-regime CRPS comparison
print(f"\n  Per-regime CRPS:")
for label, mask_r in [("High", mask_high[mask]), ("Low", mask_low[mask]), ("Mid", mask_mid[mask])]:
    if mask_r.sum() < 5:
        continue
    cf = np.mean(crps_fixed[mask_r])
    cr = np.mean(crps_regime[mask_r])
    imp = (cf - cr) / cf * 100
    print(f"    {label:5s}: fixed={cf:.6f}  regime={cr:.6f}  improvement={imp:+.2f}%")

# ============================================================
# 5. Quantile coverage test
# ============================================================
print("\n[5/5] Quantile coverage (VaR calibration)...")

for alpha in [0.01, 0.05]:
    # Fixed nu VaR
    s_fixed = np.sqrt(nu_fixed / (nu_fixed - 2))
    var_fixed = sigma_c * student_t.ppf(alpha, df=nu_fixed) / s_fixed
    violations_fixed = np.mean(rH_c < var_fixed)

    # Regime-dependent nu VaR
    violations_regime = 0
    for i in range(len(rH_c)):
        nu_i = max(nu_base - delta * G_c[i], 2.1)
        s_i = np.sqrt(nu_i / (nu_i - 2))
        var_i = sigma_c[i] * student_t.ppf(alpha, df=nu_i) / s_i
        if rH_c[i] < var_i:
            violations_regime += 1
    violations_regime /= len(rH_c)

    print(f"  alpha={alpha}: target={alpha:.3f}  fixed_nu={violations_fixed:.3f}  regime_nu={violations_regime:.3f}")
    print(f"    Fixed error:  {abs(violations_fixed - alpha):.4f}")
    print(f"    Regime error: {abs(violations_regime - alpha):.4f}")
    if abs(violations_regime - alpha) < abs(violations_fixed - alpha):
        print(f"    REGIME BETTER")
    else:
        print(f"    FIXED BETTER")

print("\n" + "=" * 70)
print("FEASIBILITY TEST COMPLETE")
print("=" * 70)
