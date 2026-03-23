#!/usr/bin/env python3
"""
Generate publication-quality figures for Paper 3: Economic Validation.

Produces 8 PDF figures for the manuscript, saved to manuscript/figures/.
Reads pre-computed analysis results from outputs/tables/ and loads raw
regime data from Paper 1 for time-series and crisis-episode plots.

Usage:
    cd paper3-economic-validation
    python -u scripts/generate_figures.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure src is importable
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.regime_loader import load_regime_series, load_regime_panel, align_regime_and_external
from src.external_data import load_stress_proxies
from src.lead_lag import cross_correlation, peak_lag
from src.distributional import (
    full_distributional_comparison,
    classify_regimes_quantile,
    conditional_means,
)
from src.event_study import define_episodes, extract_episode_window, _nearest_trading_day

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TABLE_DIR = PROJECT_DIR / "outputs" / "tables"
FIGURE_DIR = PROJECT_DIR / "manuscript" / "figures"
PAPER1_EXP_DIR = PROJECT_DIR.parent / "paper1-latent-str" / "outputs" / "exp_v1"

# ---------------------------------------------------------------------------
# Publication style — match Paper 1 (JFEC compliant)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.labelsize": 11,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette — consistent with Paper 1
COLOR_SSM = "#2ca02c"     # Green for G_ssm / Deep-LSTR
COLOR_OBS = "#1f77b4"     # Blue for G_obs / STR-HAR
COLOR_GRAY = "#888888"    # Gray for reference lines
COLOR_CRISIS = "#ffcccc"  # Light red for crisis shading
COLOR_THRESHOLD = "#D55E00"  # Vermillion for threshold lines

# Stress variable display labels
VARIABLE_LABELS = {
    "VIX": "VIX",
    "VVIX": "VVIX",
    "SKEW": "SKEW",
    "HY_OAS": "HY OAS",
    "BBB_OAS": "BBB OAS",
    "TERM_SPREAD": "Term Spread",
    "TED": "TED Spread",
}

# Crisis episode definitions (matching event_study.py)
CRISIS_EPISODES = [
    {"name": "Volmageddon", "start": "2018-02-01", "peak": "2018-02-05", "end": "2018-02-28"},
    {"name": "COVID Crash", "start": "2020-02-20", "peak": "2020-03-16", "end": "2020-04-30"},
    {"name": "Rate Shock 2022", "start": "2022-09-01", "peak": "2022-10-15", "end": "2022-11-30"},
    {"name": "SVB Crisis", "start": "2023-03-08", "peak": "2023-03-13", "end": "2023-03-31"},
]


# ===================================================================
# Data loading (cached after first call)
# ===================================================================
_CACHE = {}


def _load_gspc_regime():
    """Load GSPC H1 regime data from Paper 1."""
    if "gspc_regime" not in _CACHE:
        _CACHE["gspc_regime"] = load_regime_series(
            exp_dir=str(PAPER1_EXP_DIR), ticker="GSPC", horizon=1
        )
    return _CACHE["gspc_regime"]


def _load_stress():
    """Load external stress proxies."""
    if "stress" not in _CACHE:
        _CACHE["stress"] = load_stress_proxies()
    return _CACHE["stress"]


def _load_aligned():
    """Load GSPC regime + stress data aligned on common dates."""
    if "aligned" not in _CACHE:
        regime = _load_gspc_regime()
        stress = _load_stress()
        _CACHE["aligned"] = align_regime_and_external(regime, stress)
    return _CACHE["aligned"]


# ===================================================================
# Figure 1: Cross-Correlation Functions (2x2 panel)
# ===================================================================
def generate_fig1_cross_correlation():
    """CCF of G_ssm vs VIX, HY_OAS, SKEW, TERM_SPREAD for GSPC."""
    print("Generating Figure 1: Cross-Correlation Functions...")

    aligned = _load_aligned()
    variables = ["VIX", "HY_OAS", "SKEW", "TERM_SPREAD"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        ax = axes[i]
        ext = aligned[var].dropna()
        g_ssm = aligned["G_ssm"]

        ccf = cross_correlation(g_ssm, ext, max_lag=20)
        pk_lag_val, pk_rho_val, pk_p_val = peak_lag(ccf)

        # Plot CCF bars
        lags = ccf["lag"].values
        rhos = ccf["rho"].values
        ax.bar(lags, rhos, width=0.8, color=COLOR_SSM, alpha=0.6, edgecolor="none")

        # 95% CI band (approximate: +/- 1.96/sqrt(T))
        T = len(ext.index.intersection(g_ssm.dropna().index))
        ci_bound = 1.96 / np.sqrt(T)
        ax.axhspan(-ci_bound, ci_bound, color=COLOR_GRAY, alpha=0.15, label="95% CI")

        # Vertical dashed line at peak lag
        ax.axvline(pk_lag_val, color=COLOR_THRESHOLD, linestyle="--",
                    linewidth=1.5, alpha=0.8)

        # Annotate peak lag and rho
        sign_str = "+" if pk_lag_val >= 0 else ""
        ax.annotate(
            f"Peak: lag={sign_str}{int(pk_lag_val)}\n$\\rho$={pk_rho_val:.3f}",
            xy=(pk_lag_val, pk_rho_val),
            xytext=(0.97, 0.97),
            textcoords="axes fraction",
            ha="right", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=COLOR_GRAY, alpha=0.9),
        )

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Lag (trading days)")
        ax.set_ylabel("Correlation")
        label = VARIABLE_LABELS.get(var, var)
        panel_letter = chr(65 + i)  # A, B, C, D
        ax.set_title(f"({panel_letter}) $G_{{ssm}}$ vs. {label}", fontweight="bold")
        ax.set_xlim(-21, 21)

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig1_cross_correlation.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ===================================================================
# Figure 2: Regime vs Stress Time Series (dual-axis)
# ===================================================================
def generate_fig2_regime_vs_stress_timeseries():
    """Dual-axis time series: G_ssm (green), VIX (blue), crisis shading."""
    print("Generating Figure 2: Regime vs Stress Time Series...")

    aligned = _load_aligned()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Left axis: regime indicators
    ax1.plot(aligned.index, aligned["G_ssm"], color=COLOR_SSM,
             linewidth=0.9, alpha=0.9, label="$G_{ssm}$ (Deep-LSTR)")
    ax1.plot(aligned.index, aligned["G_obs"], color=COLOR_GRAY,
             linewidth=0.5, alpha=0.5, linestyle="--", label="$G_{obs}$ (STR-HAR)")
    ax1.set_ylabel("Transition Function $G(s_t)$", color="black")
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y")

    # Right axis: VIX
    ax2 = ax1.twinx()
    ax2.plot(aligned.index, aligned["VIX"], color=COLOR_OBS,
             linewidth=0.7, alpha=0.7, label="VIX")
    ax2.set_ylabel("VIX Index", color=COLOR_OBS)
    ax2.tick_params(axis="y", labelcolor=COLOR_OBS)
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    # Shade crisis episodes
    for ep in CRISIS_EPISODES:
        start = pd.Timestamp(ep["start"])
        end = pd.Timestamp(ep["end"])
        ax1.axvspan(start, end, color=COLOR_CRISIS, alpha=0.4, zorder=0)
        # Label at top
        mid = start + (end - start) / 2
        ax1.text(mid, 1.02, ep["name"], ha="center", va="bottom",
                 fontsize=7, rotation=0, transform=ax1.get_xaxis_transform())

    # Format x-axis
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
               framealpha=0.9)

    fig.tight_layout()
    out_path = FIGURE_DIR / "fig2_regime_vs_stress_timeseries.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ===================================================================
# Figure 3: Distributional Comparison (grouped bar of Cohen's d)
# ===================================================================
def generate_fig3_distributional_comparison():
    """Grouped bar chart: Cohen's d for G_ssm vs G_obs, by stress variable."""
    print("Generating Figure 3: Distributional Comparison (Cohen's d)...")

    dist_ssm = pd.read_csv(TABLE_DIR / "distributional_gsm.csv")
    dist_obs = pd.read_csv(TABLE_DIR / "distributional_gobs.csv")

    # Merge on variable
    merged = dist_ssm[["variable", "cohens_d"]].merge(
        dist_obs[["variable", "cohens_d"]],
        on="variable", suffixes=("_ssm", "_obs"),
    )

    # Sort by SSM effect size (descending absolute)
    merged["abs_d_ssm"] = merged["cohens_d_ssm"].abs()
    merged = merged.sort_values("abs_d_ssm", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(merged))
    width = 0.35

    bars_ssm = ax.bar(x - width / 2, merged["cohens_d_ssm"].abs(), width,
                       color=COLOR_SSM, alpha=0.8, label="$G_{ssm}$ (Deep-LSTR)",
                       edgecolor="white", linewidth=0.5)
    bars_obs = ax.bar(x + width / 2, merged["cohens_d_obs"].abs(), width,
                       color=COLOR_OBS, alpha=0.8, label="$G_{obs}$ (STR-HAR)",
                       edgecolor="white", linewidth=0.5)

    # Large effect threshold
    ax.axhline(0.8, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.2,
               alpha=0.7, label="Large effect (d=0.8)")

    # X-axis labels
    labels = [VARIABLE_LABELS.get(v, v) for v in merged["variable"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("|Cohen's d|")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig3_distributional_comparison.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ===================================================================
# Figure 4: Crisis Episodes (2x2 panel)
# ===================================================================
def generate_fig4_crisis_episodes():
    """2x2 panel: G_ssm and G_obs trajectories around each crisis episode."""
    print("Generating Figure 4: Crisis Episodes...")

    gspc = _load_gspc_regime()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    episodes = define_episodes()

    for i, ep in enumerate(episodes):
        ax = axes[i]

        # Extract episode window (raw, NOT normalized — we want raw G values)
        trading_dates = gspc.index.sort_values()
        peak_date = _nearest_trading_day(
            pd.Timestamp(ep["peak"]), trading_dates, direction="backward"
        )
        start_date = _nearest_trading_day(
            pd.Timestamp(ep["start"]), trading_dates, direction="backward"
        )
        end_date = _nearest_trading_day(
            pd.Timestamp(ep["end"]), trading_dates, direction="backward"
        )

        # Window: 20 days before start to 40 days after peak
        peak_idx = trading_dates.get_loc(peak_date)
        start_idx = trading_dates.get_loc(start_date)
        window_start = max(0, start_idx - 20)
        window_end = min(len(trading_dates) - 1, peak_idx + 40)

        window = gspc.iloc[window_start:window_end + 1].copy()

        # Compute relative trading-day index (t=0 at peak)
        peak_pos = window.index.get_loc(peak_date)
        t_values = np.arange(len(window)) - peak_pos

        # Plot G_ssm and G_obs
        ax.plot(t_values, window["G_ssm"].values, color=COLOR_SSM,
                linewidth=1.5, label="$G_{ssm}$ (Deep-LSTR)")
        ax.plot(t_values, window["G_obs"].values, color=COLOR_OBS,
                linewidth=1.0, linestyle="--", alpha=0.8,
                label="$G_{obs}$ (STR-HAR)")

        # Vertical lines for episode start and end
        start_pos_in_window = window.index.get_loc(start_date) if start_date in window.index else None
        end_pos_in_window = window.index.get_loc(end_date) if end_date in window.index else None

        if start_pos_in_window is not None:
            t_start = start_pos_in_window - peak_pos
            ax.axvline(t_start, color=COLOR_GRAY, linestyle=":", linewidth=1.0,
                       alpha=0.7)
            ax.text(t_start, ax.get_ylim()[1] * 0.02 + ax.get_ylim()[0] * 0.98,
                    " Start", fontsize=7, color=COLOR_GRAY, va="bottom")

        if end_pos_in_window is not None:
            t_end = end_pos_in_window - peak_pos
            ax.axvline(t_end, color=COLOR_GRAY, linestyle=":", linewidth=1.0,
                       alpha=0.7)
            ax.text(t_end, ax.get_ylim()[1] * 0.02 + ax.get_ylim()[0] * 0.98,
                    " End", fontsize=7, color=COLOR_GRAY, va="bottom")

        # t=0 reference
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)

        ax.axhline(0.5, color=COLOR_GRAY, linestyle="--", linewidth=0.5, alpha=0.4)
        ax.set_xlabel("Trading days relative to peak (t=0)")
        ax.set_ylabel("$G(s_t)$")
        panel_letter = chr(65 + i)
        ax.set_title(f"({panel_letter}) {ep['name']}", fontweight="bold")
        ax.set_ylim(-0.05, 1.05)

        if i == 0:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig4_crisis_episodes.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ===================================================================
# Figure 5: Predictive Incremental R-squared
# ===================================================================
def generate_fig5_predictive_r2():
    """Grouped bar chart: incremental R2 by target variable and horizon."""
    print("Generating Figure 5: Predictive Incremental R-squared...")

    incr_df = pd.read_csv(TABLE_DIR / "incremental_r2.csv")

    targets = incr_df["target"].unique()
    horizons = sorted(incr_df["horizon"].unique())
    horizon_labels = {1: "h=1", 5: "h=5", 22: "h=22"}

    fig, ax = plt.subplots(figsize=(10, 5))

    n_targets = len(targets)
    n_horizons = len(horizons)
    group_width = 0.7
    bar_width = group_width / n_horizons
    x = np.arange(n_targets)

    colors_h = ["#2ca02c", "#66c266", "#b3e0b3"]  # Decreasing green intensity

    for j, h in enumerate(horizons):
        vals = []
        for t in targets:
            row = incr_df[(incr_df["target"] == t) & (incr_df["horizon"] == h)]
            if len(row) > 0:
                vals.append(row["incremental_R2"].values[0])
            else:
                vals.append(0.0)

        offset = (j - (n_horizons - 1) / 2) * bar_width
        ax.bar(x + offset, vals, bar_width, color=colors_h[j], alpha=0.85,
               edgecolor="white", linewidth=0.5, label=horizon_labels[h])

    # Format
    target_labels = [VARIABLE_LABELS.get(t, t) for t in targets]
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels)
    ax.set_ylabel("Incremental $R^2$")
    ax.legend(loc="upper right", framealpha=0.9)

    # Scientific notation for y-axis
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(-3, -3))

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig5_predictive_r2.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ===================================================================
# Figure 6: Observable vs Latent Alignment
# ===================================================================
def generate_fig6_obs_vs_latent_alignment():
    """Grouped bar: peak CCF |rho| for G_obs vs G_ssm, by stress variable."""
    print("Generating Figure 6: Observable vs Latent Alignment...")

    # Compute CCFs for both G_ssm and G_obs
    aligned = _load_aligned()
    variables = ["VIX", "VVIX", "SKEW", "HY_OAS", "BBB_OAS", "TERM_SPREAD", "TED"]

    rows = []
    for var in variables:
        if var not in aligned.columns:
            continue
        ext = aligned[var].dropna()

        # G_ssm CCF
        ccf_ssm = cross_correlation(aligned["G_ssm"], ext, max_lag=20)
        pk_lag_ssm, pk_rho_ssm, _ = peak_lag(ccf_ssm)

        # G_obs CCF
        ccf_obs = cross_correlation(aligned["G_obs"], ext, max_lag=20)
        pk_lag_obs, pk_rho_obs, _ = peak_lag(ccf_obs)

        rows.append({
            "variable": var,
            "abs_rho_ssm": abs(pk_rho_ssm),
            "abs_rho_obs": abs(pk_rho_obs),
            "peak_lag_ssm": pk_lag_ssm,
            "peak_lag_obs": pk_lag_obs,
        })

    df = pd.DataFrame(rows)
    # Sort by SSM rho descending
    df = df.sort_values("abs_rho_ssm", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width / 2, df["abs_rho_ssm"], width, color=COLOR_SSM,
           alpha=0.8, label="$G_{ssm}$ (Deep-LSTR)", edgecolor="white",
           linewidth=0.5)
    ax.bar(x + width / 2, df["abs_rho_obs"], width, color=COLOR_OBS,
           alpha=0.8, label="$G_{obs}$ (STR-HAR)", edgecolor="white",
           linewidth=0.5)

    labels = [VARIABLE_LABELS.get(v, v) for v in df["variable"]]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Peak |$\\rho$| (CCF)")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig6_obs_vs_latent_alignment.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ===================================================================
# Figure 7: Granger Causality Heatmap
# ===================================================================
def generate_fig7_granger_heatmap():
    """Heatmap: -log10(p-value) for Granger causality by variable x lag."""
    print("Generating Figure 7: Granger Causality Heatmap...")

    granger_df = pd.read_csv(TABLE_DIR / "granger_causality.csv")

    # Pivot to matrix form
    variables = granger_df["variable"].unique()
    lag_orders = sorted(granger_df["lag_order"].unique())

    # Build matrix
    matrix = np.full((len(variables), len(lag_orders)), np.nan)
    sig_matrix = np.full((len(variables), len(lag_orders)), "", dtype=object)

    for i, var in enumerate(variables):
        for j, lag in enumerate(lag_orders):
            row = granger_df[(granger_df["variable"] == var) &
                             (granger_df["lag_order"] == lag)]
            if len(row) > 0:
                p = row["p_value"].values[0]
                if np.isfinite(p) and p > 0:
                    matrix[i, j] = -np.log10(p)
                    if p < 0.01:
                        sig_matrix[i, j] = "***"
                    elif p < 0.05:
                        sig_matrix[i, j] = "**"
                    elif p < 0.10:
                        sig_matrix[i, j] = "*"

    fig, ax = plt.subplots(figsize=(8, 6))

    # Custom colormap: white -> light green -> dark green
    cmap_colors = ["#ffffff", "#c7e9c0", "#74c476", "#31a354", "#006d2c"]
    cmap = mcolors.LinearSegmentedColormap.from_list("granger", cmap_colors)

    # Significance threshold: -log10(0.05) = 1.301
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=0, vmax=max(3.5, np.nanmax(matrix)))

    # Annotate with asterisks
    for i in range(len(variables)):
        for j in range(len(lag_orders)):
            text = sig_matrix[i, j]
            if text:
                val = matrix[i, j]
                text_color = "white" if val > 2.0 else "black"
                ax.text(j, i, text, ha="center", va="center",
                        fontsize=11, fontweight="bold", color=text_color)

    # Axis labels
    var_labels = [VARIABLE_LABELS.get(v, v) for v in variables]
    ax.set_xticks(np.arange(len(lag_orders)))
    ax.set_xticklabels([str(l) for l in lag_orders])
    ax.set_yticks(np.arange(len(variables)))
    ax.set_yticklabels(var_labels)
    ax.set_xlabel("Lag Order")
    ax.set_ylabel("Stress Variable")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="$-\\log_{10}(p)$")
    # Add significance reference line
    cbar.ax.axhline(-np.log10(0.05), color="black", linestyle="--",
                     linewidth=1.0, alpha=0.7)
    cbar.ax.text(1.1, -np.log10(0.05), "p=0.05", va="center", fontsize=8,
                 transform=cbar.ax.get_yaxis_transform())

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig7_granger_heatmap.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ===================================================================
# Figure 8: Threshold Sensitivity
# ===================================================================
def generate_fig8_threshold_sensitivity():
    """Line plot: mean Cohen's d across variables as threshold changes."""
    print("Generating Figure 8: Threshold Sensitivity...")

    aligned = _load_aligned()
    g_ssm = aligned["G_ssm"]
    variables = ["VIX", "VVIX", "SKEW", "HY_OAS", "BBB_OAS", "TERM_SPREAD"]
    available_vars = [v for v in variables if v in aligned.columns]

    # Quantile pairs to test: (q_low, q_high)
    # These correspond to increasingly aggressive splits
    quantile_pairs = [
        (0.10, 0.90),
        (0.20, 0.80),
        (0.25, 0.75),
        (0.30, 0.70),
        (0.40, 0.60),
    ]

    pair_labels = ["10/90", "20/80", "25/75", "30/70", "40/60"]

    # Also compute for G_obs with threshold=0.5 as reference
    results_ssm = {var: [] for var in available_vars}
    mean_d_ssm = []

    for q_low, q_high in quantile_pairs:
        dist_result = full_distributional_comparison(
            external_df=aligned,
            G_series=g_ssm,
            variables=available_vars,
            use_quantile=True,
            q_high=q_high,
            q_low=q_low,
            n_boot=200,  # Fewer bootstraps for speed
            seed=42,
        )

        d_values = []
        for _, row in dist_result.iterrows():
            var = row["variable"]
            d = abs(row["cohens_d"])
            results_ssm[var].append(d)
            if np.isfinite(d):
                d_values.append(d)

        mean_d_ssm.append(np.mean(d_values) if d_values else np.nan)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot mean across all variables
    x = np.arange(len(quantile_pairs))
    ax.plot(x, mean_d_ssm, "o-", color=COLOR_SSM, linewidth=2.0,
            markersize=8, label="Mean |Cohen's d|", zorder=5)

    # Plot individual variable lines (lighter)
    for var in available_vars:
        label = VARIABLE_LABELS.get(var, var)
        ax.plot(x, results_ssm[var], ".-", alpha=0.35, linewidth=0.8,
                markersize=4, label=label)

    # Large effect threshold
    ax.axhline(0.8, color=COLOR_THRESHOLD, linestyle="--", linewidth=1.2,
               alpha=0.7, label="Large effect (d=0.8)")

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels)
    ax.set_xlabel("Quantile Split (low/high)")
    ax.set_ylabel("|Cohen's d|")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)

    plt.tight_layout()
    out_path = FIGURE_DIR / "fig8_threshold_sensitivity.pdf"
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


# ===================================================================
# Main
# ===================================================================
def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Paper 3: Generating Publication Figures")
    print(f"  Output directory: {FIGURE_DIR}")
    print("=" * 60)

    generate_fig1_cross_correlation()
    generate_fig2_regime_vs_stress_timeseries()
    generate_fig3_distributional_comparison()
    generate_fig4_crisis_episodes()
    generate_fig5_predictive_r2()
    generate_fig6_obs_vs_latent_alignment()
    generate_fig7_granger_heatmap()
    generate_fig8_threshold_sensitivity()

    print("=" * 60)
    print("  All 8 figures generated successfully.")
    print(f"  Output: {FIGURE_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
