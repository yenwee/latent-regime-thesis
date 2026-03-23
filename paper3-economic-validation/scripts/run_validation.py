"""
Paper 3: Economic Validation — Main Analysis Runner

Loads pre-computed regime indicators from Paper 1, fetches external
stress proxies, and runs all four validation analyses:
  A. Lead-lag (cross-correlation + Granger causality)
  B. Distributional comparison (KS, Mann-Whitney, Levene)
  C. Crisis event studies (4 episodes)
  D. Predictive regressions (regime -> stress outcomes)

Usage:
    cd paper3-economic-validation
    python -u scripts/run_validation.py
"""

import logging
import os
import pickle
import sys
import time
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.external_data import load_stress_proxies
from src.regime_loader import (
    load_regime_panel,
    load_regime_series,
    align_regime_and_external,
)
from src.lead_lag import (
    cross_correlation,
    granger_causality,
    panel_cross_correlation,
    peak_lag,
)
from src.distributional import (
    full_distributional_comparison,
)
from src.event_study import (
    define_episodes,
    episode_summary,
    compare_obs_vs_latent,
)
from src.predictive_regression import (
    regime_predicts_stress,
    panel_predictive_regression,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PAPER1_EXP_DIR = str(
    Path(__file__).resolve().parent.parent.parent
    / "paper1-latent-str" / "outputs" / "exp_v1"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"

PRIMARY_ASSET = "GSPC"
PRIMARY_HORIZON = 1

STRESS_VARIABLES = ["VIX", "VVIX", "SKEW", "HY_OAS", "BBB_OAS", "TERM_SPREAD", "TED"]
PANEL_VARIABLES = ["VIX", "HY_OAS"]
PREDICTIVE_TARGETS = ["VIX", "HY_OAS", "TERM_SPREAD"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_validation")


def stars(p):
    """Return significance stars for a p-value."""
    if p is None or np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# ===================================================================
# MAIN
# ===================================================================
def main():
    t0 = time.time()

    # Create output directories
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output tables directory: %s", TABLE_DIR)

    # ------------------------------------------------------------------
    # 1. Load regime panel from Paper 1 (H=1 primary)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Loading regime panel from Paper 1 (H=%d)", PRIMARY_HORIZON)
    logger.info("=" * 60)

    regime_panel = load_regime_panel(exp_dir=PAPER1_EXP_DIR, horizon=PRIMARY_HORIZON)
    logger.info("Loaded %d assets: %s", len(regime_panel), sorted(regime_panel.keys()))

    # Quick summary of GSPC regimes
    if PRIMARY_ASSET in regime_panel:
        gspc = regime_panel[PRIMARY_ASSET]
        logger.info(
            "GSPC: %d obs, G_ssm mean=%.3f std=%.3f, G_obs mean=%.3f std=%.3f",
            len(gspc), gspc["G_ssm"].mean(), gspc["G_ssm"].std(),
            gspc["G_obs"].mean(), gspc["G_obs"].std(),
        )

    # ------------------------------------------------------------------
    # 2. Fetch external stress proxies
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Fetching external stress proxies")
    logger.info("=" * 60)

    stress_df = load_stress_proxies()
    logger.info(
        "Stress proxies: %d obs, %d variables (%s to %s)",
        len(stress_df), len(stress_df.columns),
        stress_df.index.min().strftime("%Y-%m-%d"),
        stress_df.index.max().strftime("%Y-%m-%d"),
    )
    for col in stress_df.columns:
        n_valid = stress_df[col].notna().sum()
        logger.info("  %s: %d valid, mean=%.2f, std=%.2f",
                     col, n_valid,
                     stress_df[col].mean(), stress_df[col].std())

    # ------------------------------------------------------------------
    # 3. Align dates (for GSPC as primary)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Aligning dates")
    logger.info("=" * 60)

    gspc_aligned = align_regime_and_external(regime_panel[PRIMARY_ASSET], stress_df)
    logger.info("GSPC aligned: %d observations", len(gspc_aligned))

    # Collect all results
    results = {}

    # ==================================================================
    # A. LEAD-LAG ANALYSIS
    # ==================================================================
    logger.info("=" * 60)
    logger.info("ANALYSIS A: Lead-Lag (Cross-Correlation + Granger)")
    logger.info("=" * 60)

    # A1: Cross-correlation of G_ssm vs each stress variable for GSPC
    ccf_rows = []
    available_stress = [v for v in STRESS_VARIABLES if v in gspc_aligned.columns]

    for var in available_stress:
        ext_series = gspc_aligned[var].dropna()
        g_ssm = gspc_aligned["G_ssm"]

        ccf = cross_correlation(g_ssm, ext_series, max_lag=20)
        pk_lag, pk_rho, pk_p = peak_lag(ccf)

        ccf_rows.append({
            "variable": var,
            "peak_lag": pk_lag,
            "peak_rho": pk_rho,
            "peak_p": pk_p,
            "rho_lag0": ccf.loc[ccf["lag"] == 0, "rho"].values[0] if 0 in ccf["lag"].values else np.nan,
            "rho_lag1": ccf.loc[ccf["lag"] == 1, "rho"].values[0] if 1 in ccf["lag"].values else np.nan,
            "rho_lag5": ccf.loc[ccf["lag"] == 5, "rho"].values[0] if 5 in ccf["lag"].values else np.nan,
        })

        logger.info(
            "  G_ssm vs %s: peak_lag=%d, rho=%.4f (p=%.4f)%s",
            var, pk_lag, pk_rho, pk_p, stars(pk_p)
        )

    ccf_df = pd.DataFrame(ccf_rows)
    ccf_df.to_csv(TABLE_DIR / "lead_lag_ccf.csv", index=False)
    results["lead_lag_ccf"] = ccf_df

    # A2: Panel cross-correlation for VIX and HY_OAS
    panel_ccf_results = {}
    for var in PANEL_VARIABLES:
        if var not in stress_df.columns:
            continue
        panel_result = panel_cross_correlation(
            regime_panel, stress_df, variable=var,
            regime_type="G_ssm", max_lag=20
        )
        panel_ccf_results[var] = panel_result
        if panel_result["n_assets"] > 0:
            pk_lag_panel, pk_rho_panel, pk_p_panel = peak_lag(panel_result["mean_ccf"])
            logger.info(
                "  Panel G_ssm vs %s (%d assets): peak_lag=%d, mean_rho=%.4f",
                var, panel_result["n_assets"], pk_lag_panel, pk_rho_panel
            )
    results["panel_ccf"] = panel_ccf_results

    # A3: Granger causality for GSPC: G_ssm -> each stress variable
    granger_rows = []
    for var in available_stress:
        ext_series = gspc_aligned[var].dropna()
        g_ssm = gspc_aligned["G_ssm"]

        gc = granger_causality(g_ssm, ext_series, max_order=5)
        for lag_order, gc_result in gc.items():
            granger_rows.append({
                "variable": var,
                "lag_order": lag_order,
                "F_stat": gc_result["F_stat"],
                "p_value": gc_result["p_value"],
                "regime_differenced": gc_result["regime_differenced"],
                "external_differenced": gc_result["external_differenced"],
            })

        # Report best lag
        best_lag = min(gc.keys(), key=lambda k: gc[k]["p_value"] if np.isfinite(gc[k]["p_value"]) else 999)
        logger.info(
            "  Granger G_ssm -> %s: best lag=%d, F=%.2f, p=%.4f%s",
            var, best_lag, gc[best_lag]["F_stat"], gc[best_lag]["p_value"],
            stars(gc[best_lag]["p_value"])
        )

    granger_df = pd.DataFrame(granger_rows)
    granger_df.to_csv(TABLE_DIR / "granger_causality.csv", index=False)
    results["granger_causality"] = granger_df

    # ==================================================================
    # B. DISTRIBUTIONAL COMPARISON
    # ==================================================================
    logger.info("=" * 60)
    logger.info("ANALYSIS B: Distributional Comparison")
    logger.info("=" * 60)

    # B1: G_ssm with quantile-based classification (top/bottom 25%)
    logger.info("  G_ssm (quantile-based: top/bottom 25%%)")
    dist_ssm = full_distributional_comparison(
        external_df=gspc_aligned,
        G_series=gspc_aligned["G_ssm"],
        variables=available_stress,
        use_quantile=True,
        q_high=0.75,
        q_low=0.25,
        n_boot=1000,
        seed=42,
    )
    dist_ssm.to_csv(TABLE_DIR / "distributional_gsm.csv", index=False)
    results["distributional_gsm"] = dist_ssm

    for _, row in dist_ssm.iterrows():
        logger.info(
            "    %s: mean_high=%.3f, mean_low=%.3f, diff=%.3f, "
            "Cohen's d=%.3f, KS p=%.4f%s, MW p=%.4f%s",
            row["variable"],
            row["mean_high"], row["mean_low"], row["diff"],
            row["cohens_d"],
            row["ks_p"], stars(row["ks_p"]),
            row["mw_p"], stars(row["mw_p"]),
        )

    # B2: G_obs with 0.5 threshold
    logger.info("  G_obs (threshold=0.5)")
    dist_obs = full_distributional_comparison(
        external_df=gspc_aligned,
        G_series=gspc_aligned["G_obs"],
        variables=available_stress,
        use_quantile=False,
        threshold=0.5,
        n_boot=1000,
        seed=42,
    )
    dist_obs.to_csv(TABLE_DIR / "distributional_gobs.csv", index=False)
    results["distributional_gobs"] = dist_obs

    for _, row in dist_obs.iterrows():
        logger.info(
            "    %s: mean_high=%.3f, mean_low=%.3f, diff=%.3f, "
            "Cohen's d=%.3f, KS p=%.4f%s",
            row["variable"],
            row["mean_high"], row["mean_low"], row["diff"],
            row["cohens_d"],
            row["ks_p"], stars(row["ks_p"]),
        )

    # ==================================================================
    # C. CRISIS EVENT STUDIES
    # ==================================================================
    logger.info("=" * 60)
    logger.info("ANALYSIS C: Crisis Event Studies")
    logger.info("=" * 60)

    episodes = define_episodes()
    logger.info("  Episodes: %s", [ep["name"] for ep in episodes])

    # C1: Episode summary for G_ssm
    crisis_ssm = episode_summary(
        regime_panel, episodes, regime_type="G_ssm", primary_asset=PRIMARY_ASSET
    )
    crisis_ssm.to_csv(TABLE_DIR / "crisis_timing.csv", index=False)
    results["crisis_timing"] = crisis_ssm

    logger.info("  G_ssm crisis timing (GSPC):")
    gspc_rows = crisis_ssm[crisis_ssm["asset"] == PRIMARY_ASSET]
    for _, row in gspc_rows.iterrows():
        days_str = str(row["days_before_peak"]) if row["days_before_peak"] is not None else "N/A"
        logger.info(
            "    %s: days_before_peak=%s, peak_G=%.4f, days_elevated=%s",
            row["episode"], days_str,
            row["peak_G"] if row["peak_G"] is not None else float("nan"),
            str(row["days_elevated"]) if row["days_elevated"] is not None else "N/A",
        )

    # C2: Observable vs Latent comparison
    obs_vs_latent = compare_obs_vs_latent(
        regime_panel, episodes, primary_asset=PRIMARY_ASSET
    )
    obs_vs_latent.to_csv(TABLE_DIR / "obs_vs_latent_timing.csv", index=False)
    results["obs_vs_latent_timing"] = obs_vs_latent

    logger.info("  G_obs vs G_ssm comparison (GSPC):")
    gspc_timing = obs_vs_latent[obs_vs_latent["scope"] == PRIMARY_ASSET]
    for _, row in gspc_timing.iterrows():
        logger.info(
            "    %s | %s: G_obs=%s, G_ssm=%s, diff=%s",
            row["episode"], row["metric"],
            f"{row['G_obs']:.2f}" if row["G_obs"] is not None and np.isfinite(row["G_obs"]) else "N/A",
            f"{row['G_ssm']:.2f}" if row["G_ssm"] is not None and np.isfinite(row["G_ssm"]) else "N/A",
            f"{row['difference']:.2f}" if row["difference"] is not None and np.isfinite(row["difference"]) else "N/A",
        )

    # ==================================================================
    # D. PREDICTIVE REGRESSIONS
    # ==================================================================
    logger.info("=" * 60)
    logger.info("ANALYSIS D: Predictive Regressions")
    logger.info("=" * 60)

    # D1: GSPC: G_ssm predicting VIX, HY_OAS, TERM_SPREAD at h=1,5,22
    pred_rows = []
    for target in PREDICTIVE_TARGETS:
        if target not in stress_df.columns:
            logger.warning("  Skipping %s (not in stress data)", target)
            continue

        pred_result = regime_predicts_stress(
            regime_df=regime_panel[PRIMARY_ASSET],
            external_df=stress_df,
            outcome_var=target,
            regime_type="G_ssm",
            horizons=[1, 5, 22],
        )

        for _, row in pred_result.iterrows():
            pred_rows.append({
                "target": target,
                "horizon": int(row["horizon"]),
                "coef_regime": row["coef_regime"],
                "se_regime": row["se_regime"],
                "t_regime": row["t_regime"],
                "p_regime": row["p_regime"],
                "R2": row["R2"],
                "incremental_R2": row["incremental_R2"],
                "nobs": int(row["nobs"]),
            })

            logger.info(
                "  GSPC G_ssm -> %s (h=%d): coef=%.4f, t=%.2f, p=%.4f%s, "
                "R2=%.4f, incr_R2=%.4f",
                target, int(row["horizon"]),
                row["coef_regime"], row["t_regime"], row["p_regime"],
                stars(row["p_regime"]),
                row["R2"], row["incremental_R2"],
            )

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(TABLE_DIR / "predictive_regression.csv", index=False)
    results["predictive_regression"] = pred_df

    # D2: Incremental R2 summary (extract from pred_df)
    incr_r2_df = pred_df[["target", "horizon", "incremental_R2", "R2"]].copy()
    incr_r2_df.to_csv(TABLE_DIR / "incremental_r2.csv", index=False)
    results["incremental_r2"] = incr_r2_df

    # D3: Panel predictive regressions for VIX and HY_OAS
    panel_pred_rows = []
    for target in PANEL_VARIABLES:
        if target not in stress_df.columns:
            continue
        panel_pred = panel_predictive_regression(
            regime_dict=regime_panel,
            external_df=stress_df,
            outcome_var=target,
            regime_type="G_ssm",
            horizons=[1, 5, 22],
        )
        for _, row in panel_pred.iterrows():
            panel_pred_rows.append({
                "target": target,
                "horizon": int(row["horizon"]),
                "mean_coef": row["mean_coef"],
                "frac_significant": row["frac_significant"],
                "frac_positive": row["frac_positive"],
                "mean_R2": row["mean_R2"],
                "mean_incremental_R2": row["mean_incremental_R2"],
                "n_assets": int(row["n_assets"]),
            })

            logger.info(
                "  Panel G_ssm -> %s (h=%d): mean_coef=%.4f, "
                "frac_sig=%.1f%%, frac_pos=%.1f%%, n=%d",
                target, int(row["horizon"]),
                row["mean_coef"],
                row["frac_significant"] * 100,
                row["frac_positive"] * 100,
                int(row["n_assets"]),
            )

    panel_pred_df = pd.DataFrame(panel_pred_rows)
    panel_pred_df.to_csv(TABLE_DIR / "panel_predictive.csv", index=False)
    results["panel_predictive"] = panel_pred_df

    # ==================================================================
    # 5. Save complete results
    # ==================================================================
    results_path = OUTPUT_DIR / "results.pkl"
    with open(results_path, "wb") as f:
        pickle.dump(results, f)
    logger.info("Saved complete results to %s", results_path)

    # ==================================================================
    # 6. Print summary
    # ==================================================================
    elapsed = time.time() - t0
    print("\n")
    print("=" * 70)
    print("  PAPER 3: ECONOMIC VALIDATION -- RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nTotal runtime: {elapsed:.1f} seconds")
    print(f"Assets in panel: {len(regime_panel)}")
    print(f"Primary asset: {PRIMARY_ASSET}")
    print(f"Aligned observations (GSPC): {len(gspc_aligned)}")

    print("\n--- A. LEAD-LAG ANALYSIS ---")
    print(f"\nCross-correlation (GSPC, G_ssm vs stress proxies):")
    print(ccf_df.to_string(index=False, float_format="%.4f"))

    print(f"\nGranger causality (best lag per variable):")
    best_granger = granger_df.loc[
        granger_df.groupby("variable")["p_value"].idxmin()
    ][["variable", "lag_order", "F_stat", "p_value"]]
    print(best_granger.to_string(index=False, float_format="%.4f"))

    print("\n--- B. DISTRIBUTIONAL COMPARISON ---")
    print(f"\nG_ssm (quantile-based, GSPC):")
    cols_to_show = ["variable", "mean_high", "mean_low", "diff", "cohens_d", "ks_p", "mw_p"]
    cols_available = [c for c in cols_to_show if c in dist_ssm.columns]
    print(dist_ssm[cols_available].to_string(index=False, float_format="%.4f"))

    print(f"\nG_obs (threshold=0.5, GSPC):")
    cols_available_obs = [c for c in cols_to_show if c in dist_obs.columns]
    print(dist_obs[cols_available_obs].to_string(index=False, float_format="%.4f"))

    print("\n--- C. CRISIS EVENT STUDIES ---")
    print(f"\nG_ssm timing (GSPC):")
    gspc_crisis = crisis_ssm[crisis_ssm["asset"] == PRIMARY_ASSET]
    print(gspc_crisis[["episode", "days_before_peak", "peak_G", "days_elevated"]].to_string(
        index=False, float_format="%.4f"))

    print(f"\nG_obs vs G_ssm comparison (GSPC):")
    gspc_compare = obs_vs_latent[obs_vs_latent["scope"] == PRIMARY_ASSET]
    print(gspc_compare[["episode", "metric", "G_obs", "G_ssm", "difference"]].to_string(
        index=False, float_format="%.4f"))

    print("\n--- D. PREDICTIVE REGRESSIONS ---")
    print(f"\nGSPC: G_ssm predicting stress variables:")
    print(pred_df.to_string(index=False, float_format="%.4f"))

    print(f"\nPanel predictive regressions:")
    print(panel_pred_df.to_string(index=False, float_format="%.4f"))

    print("\n" + "=" * 70)
    print("  Output files saved to:", TABLE_DIR)
    print("  Complete results pickle:", results_path)
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
