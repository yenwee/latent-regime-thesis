#!/usr/bin/env python3
"""
Paper 2 Idea 3: Proper Scoring Rules for Regime-Switching Forecasts

Key claim: standard QLIKE doesn't account for regime identification quality.
A model that gets the regime wrong but the level right can score the same
as one that gets both right. We need a regime-aware scoring rule.

Tests:
1. Construct a regime-aware scoring rule S(y, yhat, G, G_true)
2. Show QLIKE vs regime-aware score can DISAGREE on model ranking
3. Use Paper 1's existing results: compare STR-SSM, STR-OBS, HAR
4. Check if ranking changes matter for economic decisions (VaR)

Uses Paper 1's test_quick GSPC H=5 results — no new training needed.
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PAPER1_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "paper1-latent-str")
sys.path.insert(0, PAPER1_ROOT)

from src.metrics import qlike

np.random.seed(123)

print("=" * 70)
print("PAPER 2 IDEA 3: REGIME-AWARE SCORING RULES")
print("=" * 70)

# ============================================================
# 1. Load Paper 1 results
# ============================================================
print("\n[1/6] Loading Paper 1 results...")

results_path = os.path.join(PAPER1_ROOT, "outputs", "test_quick", "H5", "GSPC_H5_results.csv")
df = pd.read_csv(results_path, index_col=0, parse_dates=True)
print(f"  {len(df)} OOS days, columns: {list(df.columns[:12])}")

y = df["y"].values
G_ssm = df["G_ssm"].values  # Latent regime indicator
G_obs = df["G_obs"].values  # Observable regime indicator

# ============================================================
# 2. Define regime-aware scoring rules
# ============================================================
print("\n[2/6] Defining scoring rules...")

def qlike_loss(y, yhat):
    """Standard QLIKE: L(y,yhat) = exp(y)/exp(yhat) - y + yhat - 1"""
    return np.exp(y) / np.exp(yhat) - y + yhat - 1

def regime_aware_qlike(y, yhat, G_pred, G_proxy, lam=0.1):
    """
    Regime-aware QLIKE: standard QLIKE + penalty for regime misidentification.

    S(y, yhat, G) = QLIKE(y, yhat) + lambda * R(G, G_proxy)

    where R(G, G_proxy) penalizes the forecast for being in the wrong regime.
    G_proxy is constructed from realized outcomes (ex-post regime indicator).

    The key insight: two forecasts with identical QLIKE can differ in regime
    identification quality. The regime-aware score breaks this tie.
    """
    base = qlike_loss(y, yhat)
    # Ex-post regime proxy: high realized variance = high regime
    # regime_penalty = (G_pred - G_proxy)^2
    regime_penalty = (G_pred - G_proxy) ** 2
    return base + lam * regime_penalty

def conditional_qlike(y, yhat, G_pred, threshold=0.5):
    """
    Conditional QLIKE: evaluate separately in each regime.

    Returns (qlike_low_regime, qlike_high_regime, weighted_total)
    """
    high = G_pred >= threshold
    low = ~high

    q_high = np.mean(qlike_loss(y[high], yhat[high])) if high.sum() > 10 else np.nan
    q_low = np.mean(qlike_loss(y[low], yhat[low])) if low.sum() > 10 else np.nan
    q_total = np.mean(qlike_loss(y, yhat))

    return q_low, q_high, q_total

def regime_transition_score(y, yhat, G_pred, window=5):
    """
    Transition-aware score: extra penalty during regime transitions.

    During regime changes (|dG/dt| is large), forecast accuracy matters more
    because that's when regime models should add the most value.

    S = sum_t w_t * QLIKE_t, where w_t = 1 + alpha * |G_t - G_{t-1}|
    """
    base = qlike_loss(y, yhat)
    # Detect transitions: rolling change in G
    dG = np.abs(np.diff(G_pred, prepend=G_pred[0]))
    # Smooth
    dG_smooth = pd.Series(dG).rolling(window, min_periods=1).mean().values
    # Weights: upweight transition periods
    weights = 1.0 + 2.0 * dG_smooth / (dG_smooth.mean() + 1e-9)
    weights /= weights.mean()  # normalize
    return np.mean(weights * base)

# ============================================================
# 3. Construct ex-post regime proxy
# ============================================================
print("\n[3/6] Constructing ex-post regime proxy...")

# Ex-post regime: based on realized variance relative to its rolling distribution
# High realized vol (top quartile of recent history) = regime=1
rv = np.exp(y)  # realized variance
rv_rolling_median = pd.Series(rv).rolling(60, min_periods=20).median().values
rv_rolling_q75 = pd.Series(rv).rolling(60, min_periods=20).quantile(0.75).values

# G_proxy: smooth indicator of whether RV is elevated
G_proxy = np.clip((rv - rv_rolling_median) / (rv_rolling_q75 - rv_rolling_median + 1e-12), 0, 1)
G_proxy = pd.Series(G_proxy).ewm(span=5).mean().values
G_proxy = np.nan_to_num(G_proxy, nan=0.5)

print(f"  G_proxy: mean={np.mean(G_proxy):.3f}, std={np.std(G_proxy):.3f}")
print(f"  Corr(G_ssm, G_proxy): {np.corrcoef(G_ssm, G_proxy)[0,1]:.3f}")
print(f"  Corr(G_obs, G_proxy): {np.corrcoef(G_obs, G_proxy)[0,1]:.3f}")

# ============================================================
# 4. Compare model rankings under different scoring rules
# ============================================================
print("\n[4/6] Model ranking comparison...")

models = {
    "HAR": (df["har"].values, np.full(len(df), 0.5)),  # HAR has no regime, use 0.5
    "STR-OBS": (df["str_obs"].values, G_obs),
    "STR-SSM": (df["str_ssm"].values, G_ssm),
}

print(f"\n  {'Scoring Rule':<30s} {'HAR':>10s} {'STR-OBS':>10s} {'STR-SSM':>10s} {'Best':>10s}")
print(f"  {'-'*70}")

rankings = {}

# Standard QLIKE
scores = {}
for name, (yhat, G) in models.items():
    scores[name] = np.mean(qlike_loss(y, yhat))
best = min(scores, key=scores.get)
rankings["QLIKE"] = best
print(f"  {'Standard QLIKE':<30s} {scores['HAR']:10.6f} {scores['STR-OBS']:10.6f} {scores['STR-SSM']:10.6f} {best:>10s}")

# Regime-aware QLIKE (various lambda)
for lam in [0.01, 0.05, 0.1, 0.5]:
    scores = {}
    for name, (yhat, G) in models.items():
        scores[name] = np.mean(regime_aware_qlike(y, yhat, G, G_proxy, lam=lam))
    best = min(scores, key=scores.get)
    rankings[f"Regime-aware (λ={lam})"] = best
    print(f"  {f'Regime-aware QLIKE (λ={lam})':<30s} {scores['HAR']:10.6f} {scores['STR-OBS']:10.6f} {scores['STR-SSM']:10.6f} {best:>10s}")

# Transition-aware score
scores = {}
for name, (yhat, G) in models.items():
    scores[name] = regime_transition_score(y, yhat, G)
best = min(scores, key=scores.get)
rankings["Transition-aware"] = best
print(f"  {'Transition-aware':<30s} {scores['HAR']:10.6f} {scores['STR-OBS']:10.6f} {scores['STR-SSM']:10.6f} {best:>10s}")

# Conditional QLIKE (high regime)
scores_high = {}
for name, (yhat, G) in models.items():
    _, q_high, _ = conditional_qlike(y, yhat, G)
    scores_high[name] = q_high
best_high = min(scores_high, key=scores_high.get)
rankings["Conditional (high regime)"] = best_high
print(f"  {'Conditional QLIKE (high G)':<30s} {scores_high['HAR']:10.6f} {scores_high['STR-OBS']:10.6f} {scores_high['STR-SSM']:10.6f} {best_high:>10s}")

# Conditional QLIKE (low regime)
scores_low = {}
for name, (yhat, G) in models.items():
    q_low, _, _ = conditional_qlike(y, yhat, G)
    scores_low[name] = q_low
best_low = min(scores_low, key=scores_low.get)
rankings["Conditional (low regime)"] = best_low
print(f"  {'Conditional QLIKE (low G)':<30s} {scores_low['HAR']:10.6f} {scores_low['STR-OBS']:10.6f} {scores_low['STR-SSM']:10.6f} {best_low:>10s}")

# ============================================================
# 5. KEY TEST: Does ranking change?
# ============================================================
print(f"\n[5/6] Ranking analysis...")
print(f"\n  Rankings by scoring rule:")
for rule, winner in rankings.items():
    print(f"    {rule:<35s}: {winner}")

unique_winners = set(rankings.values())
print(f"\n  Unique winners: {unique_winners}")

if len(unique_winners) > 1:
    print("  *** RANKING REVERSAL DETECTED ***")
    print("  Different scoring rules select different best models!")
    print("  This is the key finding for the paper.")
else:
    print("  No ranking reversal — same model wins under all rules.")
    print("  Weaker finding: regime-awareness doesn't change conclusions.")

# ============================================================
# 6. Simulation: construct case where QLIKE misranks
# ============================================================
print(f"\n[6/6] Simulation: demonstrating QLIKE misranking...")

# Create two synthetic forecasters:
# Model A: good at level, bad at regime timing
# Model B: slightly worse at level, good at regime timing
# QLIKE prefers A, regime-aware score prefers B

T = 1000
np.random.seed(42)

# True DGP: regime-switching
true_regime = np.zeros(T)
in_stress = False
for t in range(1, T):
    if not in_stress and np.random.random() < 0.02:
        in_stress = True
    elif in_stress and np.random.random() < 0.05:
        in_stress = False
    true_regime[t] = 1.0 if in_stress else 0.0

# True log variance
y_sim = np.where(true_regime > 0.5, -6.0 + np.random.normal(0, 0.3, T),
                                      -9.0 + np.random.normal(0, 0.2, T))

# Model A: good level, bad regime (uses noisy lagged regime)
G_A = pd.Series(true_regime).shift(5).fillna(0).values + np.random.normal(0, 0.3, T)
G_A = np.clip(G_A, 0, 1)
yhat_A = np.where(G_A > 0.5, -6.1 + np.random.normal(0, 0.05, T),
                                -9.05 + np.random.normal(0, 0.05, T))

# Model B: slightly worse level, good regime (uses true regime with small noise)
G_B = true_regime + np.random.normal(0, 0.1, T)
G_B = np.clip(G_B, 0, 1)
yhat_B = np.where(G_B > 0.5, -5.9 + np.random.normal(0, 0.08, T),
                                -9.1 + np.random.normal(0, 0.08, T))

qlike_A = np.mean(qlike_loss(y_sim, yhat_A))
qlike_B = np.mean(qlike_loss(y_sim, yhat_B))

ra_A = np.mean(regime_aware_qlike(y_sim, yhat_A, G_A, true_regime, lam=0.3))
ra_B = np.mean(regime_aware_qlike(y_sim, yhat_B, G_B, true_regime, lam=0.3))

print(f"\n  Simulated scenario:")
print(f"    Model A (good level, bad regime): QLIKE={qlike_A:.4f}  Regime-aware={ra_A:.4f}")
print(f"    Model B (ok level, good regime):  QLIKE={qlike_B:.4f}  Regime-aware={ra_B:.4f}")
print(f"    QLIKE prefers: {'A' if qlike_A < qlike_B else 'B'}")
print(f"    Regime-aware prefers: {'A' if ra_A < ra_B else 'B'}")

if (qlike_A < qlike_B) != (ra_A < ra_B):
    print(f"\n  *** RANKING REVERSAL IN SIMULATION ***")
    print(f"  QLIKE and regime-aware score DISAGREE on best model!")
    print(f"  This proves standard scoring rules can misrank regime models.")
else:
    print(f"\n  No reversal in this simulation.")

# Economic significance: VaR comparison during transitions
transition_mask = np.abs(np.diff(true_regime, prepend=0)) > 0.5
transition_idx = np.where(transition_mask)[0]
if len(transition_idx) > 0:
    # Expand to ±3 days around transitions
    expanded = set()
    for idx in transition_idx:
        for offset in range(-3, 4):
            if 0 <= idx + offset < T:
                expanded.add(idx + offset)
    trans_mask = np.zeros(T, dtype=bool)
    trans_mask[list(expanded)] = True

    qlike_A_trans = np.mean(qlike_loss(y_sim[trans_mask], yhat_A[trans_mask]))
    qlike_B_trans = np.mean(qlike_loss(y_sim[trans_mask], yhat_B[trans_mask]))

    print(f"\n  During regime transitions only (n={trans_mask.sum()}):")
    print(f"    Model A: QLIKE={qlike_A_trans:.4f}")
    print(f"    Model B: QLIKE={qlike_B_trans:.4f}")
    print(f"    Even standard QLIKE prefers {'A' if qlike_A_trans < qlike_B_trans else 'B'} during transitions")

print("\n" + "=" * 70)
print("FEASIBILITY TEST COMPLETE")
print("=" * 70)
