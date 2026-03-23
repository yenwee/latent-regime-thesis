#!/usr/bin/env python3
"""
Generate publication-quality figures for the Deep-LSTR JFEC paper.

Usage:
    python scripts/generate_paper_figures.py --exp-dir data/exp_20260115_124219

Outputs figures to: {exp_dir}/figures/
"""

import argparse
import os
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# Publication style settings - JFEC compliant
plt.rcParams.update(
    {
        "font.size": 10,
        "font.family": "serif",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Model display names
MODEL_NAMES = {
    "har": "HAR",
    "str_obs": "STR-HAR",  # Observable STR-HAR
    "str_ssm": "Deep-LSTR",
    "garch": "GARCH-t",
    "egarch": "EGARCH-t",
    "msgarch": "MS-GARCH-t",
}

# Colorblind-friendly palette (IBM Design / Wong palette)
# These colors are distinguishable for all forms of color blindness
MODEL_COLORS = {
    "HAR": "#888888",           # Gray
    "STR-HAR": "#0072B2",       # Blue (accessible)
    "Deep-LSTR": "#009E73",     # Teal/Green (accessible)
    "GARCH-t": "#E69F00",       # Orange/Amber
    "EGARCH-t": "#CC79A7",      # Pink/Magenta
    "MS-GARCH-t": "#56B4E9",    # Light blue
}

# Colorblind-friendly colors for positive/negative (avoid red-green)
COLOR_POSITIVE = "#009E73"  # Teal - for "better" / positive
COLOR_NEGATIVE = "#D55E00"  # Vermillion - for "worse" / negative

# Asset class mapping
ASSET_CLASSES = {
    "GSPC": "Equity Index",
    "DJI": "Equity Index",
    "NDX": "Equity Index",
    "RUT": "Equity Index",
    "XLF": "Equity Sector",
    "XLK": "Equity Sector",
    "XLE": "Equity Sector",
    "XLU": "Equity Sector",
    "TLT": "Fixed Income",
    "IEF": "Fixed Income",
    "TNX": "Fixed Income",
    "TYX": "Fixed Income",
    "IRX": "Fixed Income",
    "EURUSDX": "Currency",
    "GBPUSDX": "Currency",
    "USDJPYX": "Currency",
    "AUDUSDX": "Currency",
    "CLF": "Commodity",
    "GCF": "Commodity",
}

CLASS_ORDER = ["Equity Index", "Equity Sector", "Fixed Income", "Currency", "Commodity"]


def get_assets(exp_dir: Path, horizon: str = "H1") -> list:
    """Get list of assets with accuracy files."""
    assets = []
    horizon_dir = exp_dir / horizon
    if horizon_dir.exists():
        for f in horizon_dir.iterdir():
            if f.name.endswith("_accuracy.csv"):
                assets.append(f.name.replace("_accuracy.csv", ""))
    return sorted(assets)


def generate_figure1_regime_smoothing(exp_dir: Path, output_dir: Path):
    """
    Figure 1: Regime Smoothing Mechanism
    Shows G_obs vs G_ssm time series, distributions, and statistics.
    """
    print("Generating Figure 1: Regime Smoothing Mechanism...")

    df = pd.read_csv(exp_dir / "H1" / "GSPC_oos_overlapping.csv", index_col=0, parse_dates=True)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Panel A: Time series
    ax = axes[0, 0]
    ax.plot(df.index, df["G_obs"], alpha=0.7, linewidth=0.5, label="STR-HAR", color=MODEL_COLORS["STR-HAR"])
    ax.plot(df.index, df["G_ssm"], alpha=0.9, linewidth=0.8, label="Deep-LSTR", color=MODEL_COLORS["Deep-LSTR"])
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(0.8, color="gray", linestyle=":", alpha=0.3)
    ax.set_ylabel("Transition Function $G(s_t)$")
    ax.set_title("(A) Transition Function Time Series", fontweight="bold")
    ax.legend(loc="upper right")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(-0.05, 1.05)

    # Panel B: Distribution
    ax = axes[0, 1]
    bins = np.linspace(0, 1, 31)
    ax.hist(df["G_obs"], bins=bins, alpha=0.6, label="STR-HAR", color=MODEL_COLORS["STR-HAR"], density=True)
    ax.hist(df["G_ssm"], bins=bins, alpha=0.6, label="Deep-LSTR", color=MODEL_COLORS["Deep-LSTR"], density=True)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Transition Function Value")
    ax.set_ylabel("Density")
    ax.set_title("(B) Distribution of Transition Values", fontweight="bold")
    ax.legend()

    # Panel C: Box plot
    ax = axes[1, 0]
    box_data = [df["G_obs"].values, df["G_ssm"].values]
    bp = ax.boxplot(box_data, tick_labels=["STR-HAR", "Deep-LSTR"], patch_artist=True)
    bp["boxes"][0].set_facecolor(MODEL_COLORS["STR-HAR"])
    bp["boxes"][0].set_alpha(0.6)
    bp["boxes"][1].set_facecolor(MODEL_COLORS["Deep-LSTR"])
    bp["boxes"][1].set_alpha(0.6)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.2, color="gray", linestyle=":", alpha=0.3)
    ax.axhline(0.8, color="gray", linestyle=":", alpha=0.3)
    ax.set_ylabel("Transition Function Value")
    ax.set_title("(C) Distributional Comparison", fontweight="bold")

    obs_extreme = ((df["G_obs"] < 0.2) | (df["G_obs"] > 0.8)).mean() * 100
    ssm_extreme = ((df["G_ssm"] < 0.2) | (df["G_ssm"] > 0.8)).mean() * 100
    ax.text(
        0.95,
        0.95,
        f"Extreme days:\nSTR-HAR: {obs_extreme:.1f}%\nDeep-LSTR: {ssm_extreme:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Panel D: Stats table
    ax = axes[1, 1]
    ax.axis("off")
    stats_data = [
        ["Property", "STR-HAR", "Deep-LSTR"],
        ["Standard Deviation", f'{df["G_obs"].std():.3f}', f'{df["G_ssm"].std():.3f}'],
        [
            "IQR",
            f'[{df["G_obs"].quantile(0.25):.2f}, {df["G_obs"].quantile(0.75):.2f}]',
            f'[{df["G_ssm"].quantile(0.25):.2f}, {df["G_ssm"].quantile(0.75):.2f}]',
        ],
        ["% Extreme (<0.2 or >0.8)", f"{obs_extreme:.1f}%", f"{ssm_extreme:.1f}%"],
        ["Autocorr (lag-1)", f'{df["G_obs"].autocorr(1):.3f}', f'{df["G_ssm"].autocorr(1):.3f}'],
    ]

    table = ax.table(cellText=stats_data, loc="center", cellLoc="center", colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    for j in range(3):
        table[(0, j)].set_facecolor("#e6e6e6")
        table[(0, j)].set_text_props(weight="bold")
    ax.set_title("(D) Summary Statistics", y=0.95, fontweight="bold")

    # No suptitle - figure number/title goes in caption per JFEC guidelines
    plt.tight_layout()
    plt.savefig(output_dir / "fig1_regime_smoothing.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig1_regime_smoothing.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig1_regime_smoothing.png/pdf")


def generate_figure2_cumulative_qlike(exp_dir: Path, output_dir: Path):
    """
    Figure 2: Cumulative Forecast Performance
    Shows cumulative QLIKE difference over time.
    """
    print("Generating Figure 2: Cumulative Forecast Performance...")

    df = pd.read_csv(exp_dir / "H1" / "GSPC_oos_overlapping.csv", index_col=0, parse_dates=True)

    def qlike(y_true, y_pred):
        rv = np.exp(y_true)
        h = np.exp(y_pred)
        return np.log(h) + rv / h

    qlike_har = qlike(df["y"], df["har"])
    qlike_ssm = qlike(df["y"], df["str_ssm"])
    qlike_obs = qlike(df["y"], df["str_obs"])

    cum_diff_har = np.cumsum(qlike_har - qlike_ssm)
    cum_diff_obs = np.cumsum(qlike_obs - qlike_ssm)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Panel A: vs HAR (colorblind-friendly palette)
    ax = axes[0]
    ax.fill_between(
        df.index, 0, cum_diff_har, where=cum_diff_har > 0, alpha=0.6, color=COLOR_POSITIVE, label="Deep-LSTR better"
    )
    ax.fill_between(df.index, 0, cum_diff_har, where=cum_diff_har <= 0, alpha=0.6, color=COLOR_NEGATIVE, label="HAR better")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative QLIKE Difference")
    ax.set_title("(A) Deep-LSTR vs. HAR", fontweight="bold")
    ax.legend(loc="upper left")

    # Panel B: vs STR-HAR (colorblind-friendly palette)
    ax = axes[1]
    ax.fill_between(
        df.index, 0, cum_diff_obs, where=cum_diff_obs > 0, alpha=0.6, color=COLOR_POSITIVE, label="Deep-LSTR better"
    )
    ax.fill_between(
        df.index, 0, cum_diff_obs, where=cum_diff_obs <= 0, alpha=0.6, color=COLOR_NEGATIVE, label="STR-HAR better"
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Cumulative QLIKE Difference")
    ax.set_xlabel("Date")
    ax.set_title("(B) Deep-LSTR vs. STR-HAR", fontweight="bold")
    ax.legend(loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # No suptitle - figure number/title goes in caption per JFEC guidelines
    plt.tight_layout()
    plt.savefig(output_dir / "fig2_cumulative_qlike.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig2_cumulative_qlike.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig2_cumulative_qlike.png/pdf")


def generate_figure3_winner_heatmap(exp_dir: Path, output_dir: Path):
    """
    Figure 3: Best Model by Asset and Horizon
    Heatmap showing winner for each asset/horizon (RV-based models only).
    """
    print("Generating Figure 3: Winner Heatmap...")

    assets = get_assets(exp_dir, "H1")
    horizons = ["H1", "H5", "H22"]
    models = ["har", "str_obs", "str_ssm"]  # RV-based models only

    winner_matrix = np.full((len(assets), len(horizons)), -1, dtype=int)

    for i, asset in enumerate(assets):
        for j, horizon in enumerate(horizons):
            try:
                acc = pd.read_csv(exp_dir / horizon / f"{asset}_accuracy.csv")
                qlike_vals = {}
                for _, row in acc.iterrows():
                    model = row["model"]
                    if model in models:
                        qlike_vals[model] = row["QLIKE"]

                if qlike_vals:
                    best = min(qlike_vals, key=qlike_vals.get)
                    model_map = {"har": 0, "str_obs": 1, "str_ssm": 2}
                    winner_matrix[i, j] = model_map.get(best, -1)
            except Exception:
                pass

    # Sort by asset class
    sorted_indices = sorted(
        range(len(assets)),
        key=lambda i: (
            CLASS_ORDER.index(ASSET_CLASSES.get(assets[i], "Other")) if ASSET_CLASSES.get(assets[i], "Other") in CLASS_ORDER else 99,
            assets[i],
        ),
    )
    assets_sorted = [assets[i] for i in sorted_indices]
    winner_matrix_sorted = winner_matrix[sorted_indices]

    fig, ax = plt.subplots(figsize=(8, 8))

    # Colorblind-friendly colors for heatmap
    colors = [MODEL_COLORS["HAR"], MODEL_COLORS["STR-HAR"], MODEL_COLORS["Deep-LSTR"], "#ffffff"]
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    ax.imshow(winner_matrix_sorted, cmap=cmap, aspect="auto", vmin=0, vmax=3)

    ax.set_xticks(range(len(horizons)))
    ax.set_xticklabels(["H=1\n(Daily)", "H=5\n(Weekly)", "H=22\n(Monthly)"])
    ax.set_yticks(range(len(assets_sorted)))
    ax.set_yticklabels(assets_sorted)

    # Class separators
    prev_class = None
    for i, asset in enumerate(assets_sorted):
        curr_class = ASSET_CLASSES.get(asset, "Other")
        if prev_class and curr_class != prev_class:
            ax.axhline(i - 0.5, color="white", linewidth=2)
        prev_class = curr_class

    ax.set_xlabel("Forecast Horizon")
    # No suptitle - figure number/title goes in caption per JFEC guidelines

    legend_elements = [
        Patch(facecolor=colors[0], label="HAR", edgecolor="black", linewidth=0.5),
        Patch(facecolor=colors[1], label="STR-HAR", edgecolor="black", linewidth=0.5),
        Patch(facecolor=colors[2], label="Deep-LSTR", edgecolor="black", linewidth=0.5),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    plt.savefig(output_dir / "fig3_winner_heatmap.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig3_winner_heatmap.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig3_winner_heatmap.png/pdf")


def generate_figure4_mcs_dumbbell(exp_dir: Path, output_dir: Path):
    """
    Figure 4: Model Confidence Set Inclusion by Horizon
    Dumbbell plot showing MCS inclusion rates.
    """
    print("Generating Figure 4: MCS Dumbbell Plot...")

    t2 = pd.read_csv(exp_dir / "tables" / "table_2_win_mcs.csv")

    models = t2["Model"].tolist()
    mcs_h1 = t2["MCS H=1"].tolist()
    mcs_h5 = t2["MCS H=5"].tolist()
    mcs_h22 = t2["MCS H=22"].tolist()

    # Remove MS-GARCH (broken)
    if "MS-GARCH-t" in models:
        idx = models.index("MS-GARCH-t")
        models.pop(idx)
        mcs_h1.pop(idx)
        mcs_h5.pop(idx)
        mcs_h22.pop(idx)

    # Rename STR-OBS to STR-HAR
    models = ["STR-HAR" if m == "STR-OBS" else m for m in models]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = [MODEL_COLORS.get(m, "#808080") for m in models]

    for i, model in enumerate(models):
        ax.plot([mcs_h1[i], mcs_h22[i]], [i, i], "k-", linewidth=1, alpha=0.3)
        ax.scatter(mcs_h1[i], i, s=120, c=colors[i], marker="o", zorder=5)
        ax.scatter(mcs_h5[i], i, s=80, c=colors[i], marker="s", alpha=0.7, zorder=5)
        ax.scatter(mcs_h22[i], i, s=120, c=colors[i], marker="^", zorder=5)

    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("MCS Inclusion Rate (90% Confidence)")
    ax.set_xlim(-0.05, 1.05)
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="o", color="gray", markersize=10, linestyle="", label="H=1 (Daily)"),
        Line2D([0], [0], marker="s", color="gray", markersize=8, linestyle="", label="H=5 (Weekly)"),
        Line2D([0], [0], marker="^", color="gray", markersize=10, linestyle="", label="H=22 (Monthly)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    # No suptitle - figure number/title goes in caption per JFEC guidelines
    ax.grid(axis="x", alpha=0.3)

    # Highlight Deep-LSTR row (colorblind-friendly)
    dlstr_idx = models.index("Deep-LSTR") if "Deep-LSTR" in models else -1
    if dlstr_idx >= 0:
        ax.axhspan(dlstr_idx - 0.5, dlstr_idx + 0.5, alpha=0.15, color=MODEL_COLORS["Deep-LSTR"])

    plt.tight_layout()
    plt.savefig(output_dir / "fig4_mcs_dumbbell.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig4_mcs_dumbbell.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig4_mcs_dumbbell.png/pdf")


def generate_figure5_fz0_loss(exp_dir: Path, output_dir: Path):
    """
    Figure 5: Tail Risk Evaluation (FZ0 Loss)
    Bar chart comparing FZ0 loss across models.
    """
    print("Generating Figure 5: FZ0 Tail Risk...")

    t4a = pd.read_csv(exp_dir / "tables" / "table_4a_fz0.csv")

    def parse_val_se(s):
        parts = s.split(" ")
        val = float(parts[0])
        se = float(parts[1].strip("()"))
        return val, se

    models = t4a["Model"].tolist()
    # Rename STR-OBS to STR-HAR
    models = ["STR-HAR" if m == "STR-OBS" else m for m in models]

    fz_1pct = [parse_val_se(t4a["1%"].iloc[i]) for i in range(len(t4a))]
    fz_5pct = [parse_val_se(t4a["5%"].iloc[i]) for i in range(len(t4a))]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    colors = [MODEL_COLORS.get(m, "#808080") for m in models]

    # Panel A: 1% VaR level
    ax = axes[0]
    x = np.arange(len(models))
    vals = [v[0] for v in fz_1pct]
    ses = [v[1] for v in fz_1pct]
    ax.bar(x, vals, yerr=ses, capsize=5, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("FZ0 Loss")
    ax.set_title("(A) 1% VaR Level", fontweight="bold")
    ax.axhline(min(vals), color=MODEL_COLORS["Deep-LSTR"], linestyle="--", alpha=0.7, linewidth=1.5)

    # Panel B: 5% VaR level
    ax = axes[1]
    vals = [v[0] for v in fz_5pct]
    ses = [v[1] for v in fz_5pct]
    ax.bar(x, vals, yerr=ses, capsize=5, color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("FZ0 Loss")
    ax.set_title("(B) 5% VaR Level", fontweight="bold")
    ax.axhline(min(vals), color=MODEL_COLORS["Deep-LSTR"], linestyle="--", alpha=0.7, linewidth=1.5)

    # No suptitle - figure number/title goes in caption per JFEC guidelines
    plt.tight_layout()
    plt.savefig(output_dir / "fig5_fz0_loss.png", dpi=300, bbox_inches="tight")
    plt.savefig(output_dir / "fig5_fz0_loss.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved: fig5_fz0_loss.png/pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from experiment results")
    parser.add_argument("--exp-dir", type=str, required=True, help="Path to experiment directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: {exp_dir}/figures)")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment directory not found: {exp_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else exp_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment directory: {exp_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Generate all figures
    generate_figure1_regime_smoothing(exp_dir, output_dir)
    generate_figure2_cumulative_qlike(exp_dir, output_dir)
    generate_figure3_winner_heatmap(exp_dir, output_dir)
    generate_figure4_mcs_dumbbell(exp_dir, output_dir)
    generate_figure5_fz0_loss(exp_dir, output_dir)

    print("=" * 60)
    print(f"All figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
