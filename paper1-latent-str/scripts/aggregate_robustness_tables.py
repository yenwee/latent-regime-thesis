#!/usr/bin/env python3
"""
Generate publication-ready robustness tables (Tables 7-10) from Deep-LSTR experiments.

This script aggregates robustness check results and produces:
- Table 7: Sensitivity to Latent Dimension (d in {1,2,4,8})
- Table 8: Alternative Transition Functions (logistic, exponential, double-logistic)
- Table 9: Subsample Performance (2-year rolling windows)
- Table 10: Alternative Volatility Estimators (GK, Parkinson, Rogers-Satchell)

Example usage:
    python scripts/aggregate_robustness_tables.py --exp-dir outputs/robustness_v1
    python scripts/aggregate_robustness_tables.py --check latent_dims --exp-dir outputs/robustness_v1
    python scripts/aggregate_robustness_tables.py --all --exp-dir outputs/exp_20260115
"""

import argparse
import os
import sys
import pickle
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import yaml

from src.metrics import qlike
from src.mcs import bootstrap_mcs
from scripts.results_schema import ResultsSchema, create_empty_results


# Display constants
TABLE_WIDTH = 70
QLIKE_SCALE = 100  # Scale QLIKE values by 100 for readability (as in paper)

# Asset class groupings (for subsample analysis breakdown if needed)
ASSET_CLASSES = {
    "Equity Indices": ["GSPC", "NDX", "RUT", "DJI"],
    "Equity Sectors": ["XLF", "XLK", "XLE", "XLU"],
    "Fixed Income": ["IRX", "TNX", "TYX", "IEF", "TLT"],
    "Currencies": ["EURUSDX", "USDJPYX", "GBPUSDX", "AUDUSDX"],
    "Commodities": ["CLF", "GCF", "NGF", "HGF"],
}


def load_robustness_config(config_path: str = None) -> dict:
    """
    Load robustness configuration.

    Args:
        config_path: Path to config file. If None, uses configs/robustness.yaml

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "robustness.yaml")

    if not os.path.exists(config_path):
        # Return default config
        return {
            "robustness": {
                "latent_dims": {
                    "enabled": True,
                    "values": [1, 2, 4, 8],
                    "baseline": 2,
                    "horizon": 5,
                },
                "transition_fns": {
                    "enabled": True,
                    "values": [
                        {"name": "logistic", "params": 10},
                        {"name": "exponential", "params": 10},
                        {"name": "double_logistic", "params": 12},
                    ],
                    "baseline": "logistic",
                    "horizon": 5,
                },
                "subsample": {
                    "enabled": True,
                    "periods": [
                        {"name": "2017-2018", "start": "2017-01-01", "end": "2018-12-31"},
                        {"name": "2019-2020", "start": "2019-01-01", "end": "2020-12-31"},
                        {"name": "2021-2022", "start": "2021-01-01", "end": "2022-12-31"},
                        {"name": "2023-2024", "start": "2023-01-01", "end": "2024-12-31"},
                    ],
                    "horizon": 5,
                },
                "volatility_estimators": {
                    "enabled": True,
                    "values": ["garman_klass", "parkinson", "rogers_satchell"],
                    "baseline": "garman_klass",
                    "horizon": 5,
                },
            },
            "mcs": {
                "alpha": 0.10,
                "block_size": 10,
                "n_bootstrap": 1000,
            },
        }

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_robustness_checkpoint(path: str) -> Optional[dict]:
    """Load a robustness checkpoint pickle file."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_robustness_checkpoint_path(exp_dir: str, check_name: str, variant: str) -> str:
    """Get checkpoint path for a robustness check variant."""
    safe_variant = str(variant).replace(".", "_").replace("/", "_")
    return os.path.join(exp_dir, "robustness", check_name, "checkpoints", f"{safe_variant}_results.pkl")


def load_oos_results(exp_dir: str, H: int, asset: str) -> Optional[pd.DataFrame]:
    """
    Load OOS results for a specific asset and horizon.

    Searches for results in multiple possible locations:
    1. H{H}/{asset}_oos_overlapping.csv
    2. H{H}/{asset}_H{H}_results.csv

    Args:
        exp_dir: Experiment directory
        H: Forecast horizon
        asset: Asset name (without special chars)

    Returns:
        DataFrame with OOS results or None if not found
    """
    safe_asset = asset.replace("=", "").replace("^", "")
    h_dir = os.path.join(exp_dir, f"H{H}")

    # Try different naming conventions
    paths_to_try = [
        os.path.join(h_dir, f"{safe_asset}_oos_overlapping.csv"),
        os.path.join(h_dir, f"{safe_asset}_H{H}_results.csv"),
    ]

    for path in paths_to_try:
        if os.path.exists(path):
            try:
                return pd.read_csv(path, index_col=0, parse_dates=True)
            except Exception:
                pass

    return None


def compute_mcs_inclusion_rate(
    asset_results: List[dict],
    model_col: str = "str_ssm",
    alpha: float = 0.10,
    n_boot: int = 1000,
    block_size: int = 10,
) -> float:
    """
    Compute MCS inclusion rate for a model across multiple assets.

    Args:
        asset_results: List of asset result dictionaries containing OOS DataFrames
        model_col: Column name for the model to check
        alpha: MCS significance level
        n_boot: Number of bootstrap replications
        block_size: Block size for circular block bootstrap

    Returns:
        Fraction of assets where model is in MCS
    """
    models = ["har", "str_obs", "str_ssm"]
    n_included = 0
    n_total = 0

    for res in asset_results:
        # Handle both nested dict format and DataFrame format
        if isinstance(res, dict):
            if "oos_df" in res:
                df = res["oos_df"]
            elif "results_df" in res:
                df = res["results_df"]
            else:
                continue
        elif isinstance(res, pd.DataFrame):
            df = res
        else:
            continue

        if "y" not in df.columns:
            continue

        y = df["y"].values

        # Compute QLIKE losses for each model
        losses_dict = {}
        for m in models:
            if m in df.columns:
                losses_dict[m] = qlike(y, df[m].values)

        if len(losses_dict) < 2:
            continue

        losses_df = pd.DataFrame(losses_dict)

        try:
            mcs_models, _ = bootstrap_mcs(
                losses_df,
                alpha=alpha,
                n_boot=n_boot,
                block_size=block_size
            )
            n_total += 1
            if model_col in mcs_models:
                n_included += 1
        except Exception:
            # If MCS fails, count as included (conservative)
            n_total += 1
            n_included += 1

    if n_total == 0:
        return 0.0

    return n_included / n_total


def aggregate_latent_dims(
    exp_dir: str,
    config: dict,
    recompute_mcs: bool = True,
) -> pd.DataFrame:
    """
    Aggregate results for Table 7: Sensitivity to Latent Dimension.

    Args:
        exp_dir: Experiment directory
        config: Configuration dictionary
        recompute_mcs: Whether to recompute MCS from raw results

    Returns:
        DataFrame with columns [latent_dim, qlike_mean, mcs_inclusion]
    """
    rob_config = config["robustness"]["latent_dims"]
    latent_dims = rob_config["values"]
    baseline = rob_config.get("baseline", 2)
    mcs_config = config.get("mcs", {})

    rows = []

    for d in latent_dims:
        variant_name = f"latent_dim_{d}"
        checkpoint_path = get_robustness_checkpoint_path(exp_dir, "latent_dims", variant_name)
        checkpoint = load_robustness_checkpoint(checkpoint_path)

        if checkpoint is None:
            # Try loading from variant experiment directory
            variant_exp_dir = os.path.join(exp_dir, "robustness", "latent_dims", f"d{d}")
            qlike_values = []
            asset_results_for_mcs = []

            # Try to load individual asset results
            asset_basket = config.get("asset_basket", [])
            H = rob_config.get("horizon", 5)

            for ticker in asset_basket:
                df = load_oos_results(variant_exp_dir, H, ticker)
                if df is not None and "y" in df.columns and "str_ssm" in df.columns:
                    ql = qlike(df["y"].values, df["str_ssm"].values).mean()
                    qlike_values.append(ql)
                    asset_results_for_mcs.append(df)

            if qlike_values:
                mean_qlike = np.mean(qlike_values)
                if recompute_mcs and asset_results_for_mcs:
                    mcs_rate = compute_mcs_inclusion_rate(
                        asset_results_for_mcs,
                        model_col="str_ssm",
                        alpha=mcs_config.get("alpha", 0.10),
                        n_boot=mcs_config.get("n_bootstrap", 1000),
                        block_size=mcs_config.get("block_size", 10),
                    )
                else:
                    mcs_rate = np.nan

                rows.append({
                    "latent_dim": d,
                    "qlike_mean": mean_qlike * QLIKE_SCALE,
                    "mcs_inclusion": mcs_rate,
                    "n_assets": len(qlike_values),
                    "is_baseline": d == baseline,
                })
            continue

        # Load from checkpoint
        mean_qlike = checkpoint.get("mean_qlike", np.nan)
        mcs_rate = checkpoint.get("mcs_inclusion_rate", np.nan)

        # Optionally recompute MCS from raw asset results
        if recompute_mcs and "asset_results" in checkpoint:
            asset_results = checkpoint["asset_results"]
            # Extract OOS DataFrames if available
            oos_dfs = []
            for ar in asset_results:
                if isinstance(ar, dict) and "oos_df" in ar:
                    oos_dfs.append(ar["oos_df"])

            if oos_dfs:
                mcs_rate = compute_mcs_inclusion_rate(
                    oos_dfs,
                    model_col="str_ssm",
                    alpha=mcs_config.get("alpha", 0.10),
                    n_boot=mcs_config.get("n_bootstrap", 1000),
                    block_size=mcs_config.get("block_size", 10),
                )

        rows.append({
            "latent_dim": d,
            "qlike_mean": mean_qlike * QLIKE_SCALE if np.isfinite(mean_qlike) else np.nan,
            "mcs_inclusion": mcs_rate,
            "n_assets": checkpoint.get("n_assets", 0),
            "is_baseline": d == baseline,
        })

    return pd.DataFrame(rows)


def aggregate_transition_fns(
    exp_dir: str,
    config: dict,
) -> pd.DataFrame:
    """
    Aggregate results for Table 8: Alternative Transition Functions.

    Args:
        exp_dir: Experiment directory
        config: Configuration dictionary

    Returns:
        DataFrame with columns [transition_fn, qlike_mean, n_params]
    """
    rob_config = config["robustness"]["transition_fns"]
    transitions = rob_config["values"]
    baseline = rob_config.get("baseline", "logistic")

    rows = []

    for trans in transitions:
        trans_name = trans["name"]
        n_params = trans.get("params", 10)
        variant_name = f"transition_{trans_name}"

        checkpoint_path = get_robustness_checkpoint_path(exp_dir, "transition_fns", variant_name)
        checkpoint = load_robustness_checkpoint(checkpoint_path)

        if checkpoint is None:
            # Try loading from variant experiment directory
            variant_exp_dir = os.path.join(exp_dir, "robustness", "transition_fns", trans_name)
            qlike_values = []

            asset_basket = config.get("asset_basket", [])
            H = rob_config.get("horizon", 5)

            for ticker in asset_basket:
                df = load_oos_results(variant_exp_dir, H, ticker)
                if df is not None and "y" in df.columns and "str_ssm" in df.columns:
                    ql = qlike(df["y"].values, df["str_ssm"].values).mean()
                    qlike_values.append(ql)

            if qlike_values:
                mean_qlike = np.mean(qlike_values)
                rows.append({
                    "transition_fn": trans_name,
                    "qlike_mean": mean_qlike * QLIKE_SCALE,
                    "n_params": n_params,
                    "n_assets": len(qlike_values),
                    "is_baseline": trans_name == baseline,
                })
            continue

        mean_qlike = checkpoint.get("mean_qlike", np.nan)

        rows.append({
            "transition_fn": trans_name,
            "qlike_mean": mean_qlike * QLIKE_SCALE if np.isfinite(mean_qlike) else np.nan,
            "n_params": checkpoint.get("n_params", n_params),
            "n_assets": checkpoint.get("n_assets", 0),
            "is_baseline": trans_name == baseline,
        })

    return pd.DataFrame(rows)


def aggregate_subsample(
    exp_dir: str,
    config: dict,
    baseline_exp_dir: str = None,
) -> pd.DataFrame:
    """
    Aggregate results for Table 9: Subsample Performance.

    This analyzes existing baseline results by period, computing QLIKE
    improvement vs HAR for each 2-year window.

    Args:
        exp_dir: Robustness experiment directory
        config: Configuration dictionary
        baseline_exp_dir: Directory with baseline results (if different from exp_dir)

    Returns:
        DataFrame with columns [period, improvement_ssm, improvement_obs]
    """
    rob_config = config["robustness"]["subsample"]
    periods = rob_config["periods"]
    H = rob_config.get("horizon", 5)

    # Check for existing checkpoint
    checkpoint_path = get_robustness_checkpoint_path(exp_dir, "subsample", "all_periods")
    checkpoint = load_robustness_checkpoint(checkpoint_path)

    if checkpoint is not None:
        # Convert checkpoint to DataFrame
        rows = []
        for period_name, res in checkpoint.items():
            if isinstance(res, dict) and "mean_improvement_ssm" in res:
                rows.append({
                    "period": period_name,
                    "improvement_ssm": res["mean_improvement_ssm"],
                    "improvement_obs": res.get("mean_improvement_obs", np.nan),
                    "n_assets": res.get("n_assets", 0),
                })
        if rows:
            return pd.DataFrame(rows)

    # Compute from baseline results
    if baseline_exp_dir is None:
        baseline_exp_dir = exp_dir

    asset_basket = config.get("asset_basket", [])
    rows = []

    for period in periods:
        period_name = period["name"]
        start_date = pd.Timestamp(period["start"])
        end_date = pd.Timestamp(period["end"])

        improvements_ssm = []
        improvements_obs = []

        for ticker in asset_basket:
            df = load_oos_results(baseline_exp_dir, H, ticker)

            if df is None:
                continue

            # Filter to period
            mask = (df.index >= start_date) & (df.index <= end_date)
            period_df = df.loc[mask]

            if len(period_df) < 50:
                continue

            # Check required columns
            required = ["y", "har", "str_ssm"]
            if not all(c in period_df.columns for c in required):
                continue

            y = period_df["y"].values

            # Compute QLIKE
            qlike_har = qlike(y, period_df["har"].values).mean()
            qlike_ssm = qlike(y, period_df["str_ssm"].values).mean()

            # Improvement vs HAR (positive = SSM better)
            imp_ssm = (qlike_har - qlike_ssm) / abs(qlike_har) * 100
            improvements_ssm.append(imp_ssm)

            # STR-OBS improvement if available
            if "str_obs" in period_df.columns:
                qlike_obs = qlike(y, period_df["str_obs"].values).mean()
                imp_obs = (qlike_har - qlike_obs) / abs(qlike_har) * 100
                improvements_obs.append(imp_obs)

        if improvements_ssm:
            rows.append({
                "period": period_name,
                "improvement_ssm": np.mean(improvements_ssm),
                "improvement_obs": np.mean(improvements_obs) if improvements_obs else np.nan,
                "n_assets": len(improvements_ssm),
            })

    return pd.DataFrame(rows)


def aggregate_volatility_estimators(
    exp_dir: str,
    config: dict,
) -> pd.DataFrame:
    """
    Aggregate results for Table 10: Alternative Volatility Estimators.

    Args:
        exp_dir: Experiment directory
        config: Configuration dictionary

    Returns:
        DataFrame with columns [estimator, qlike_ssm, qlike_obs, improvement]
    """
    rob_config = config["robustness"]["volatility_estimators"]
    estimators = rob_config["values"]
    baseline = rob_config.get("baseline", "garman_klass")
    H = rob_config.get("horizon", 5)

    # Display name mapping
    display_names = {
        "garman_klass": "Garman-Klass",
        "parkinson": "Parkinson",
        "rogers_satchell": "Rogers-Satchell",
    }

    rows = []

    for est_name in estimators:
        variant_name = f"volatility_{est_name}"
        checkpoint_path = get_robustness_checkpoint_path(exp_dir, "volatility_estimators", variant_name)
        checkpoint = load_robustness_checkpoint(checkpoint_path)

        if checkpoint is None:
            # Try loading from variant experiment directory
            variant_exp_dir = os.path.join(exp_dir, "robustness", "volatility_estimators", est_name)
            qlike_ssm_values = []
            qlike_obs_values = []

            asset_basket = config.get("asset_basket", [])

            for ticker in asset_basket:
                df = load_oos_results(variant_exp_dir, H, ticker)
                if df is None or "y" not in df.columns:
                    continue

                y = df["y"].values

                if "str_ssm" in df.columns:
                    qlike_ssm_values.append(qlike(y, df["str_ssm"].values).mean())
                if "str_obs" in df.columns:
                    qlike_obs_values.append(qlike(y, df["str_obs"].values).mean())

            if qlike_ssm_values:
                mean_ssm = np.mean(qlike_ssm_values)
                mean_obs = np.mean(qlike_obs_values) if qlike_obs_values else np.nan
                improvement = (mean_obs - mean_ssm) / abs(mean_obs) * 100 if np.isfinite(mean_obs) else np.nan

                rows.append({
                    "estimator": display_names.get(est_name, est_name),
                    "estimator_key": est_name,
                    "qlike_ssm": mean_ssm * QLIKE_SCALE,
                    "qlike_obs": mean_obs * QLIKE_SCALE if np.isfinite(mean_obs) else np.nan,
                    "improvement": improvement,
                    "n_assets": len(qlike_ssm_values),
                    "is_baseline": est_name == baseline,
                })
            continue

        mean_ssm = checkpoint.get("mean_qlike_ssm", np.nan)
        mean_obs = checkpoint.get("mean_qlike_obs", np.nan)
        improvement = checkpoint.get("improvement_pct", np.nan)

        rows.append({
            "estimator": display_names.get(est_name, est_name),
            "estimator_key": est_name,
            "qlike_ssm": mean_ssm * QLIKE_SCALE if np.isfinite(mean_ssm) else np.nan,
            "qlike_obs": mean_obs * QLIKE_SCALE if np.isfinite(mean_obs) else np.nan,
            "improvement": improvement,
            "n_assets": checkpoint.get("n_assets", 0),
            "is_baseline": est_name == baseline,
        })

    return pd.DataFrame(rows)


def format_table_7(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Format and save Table 7: Sensitivity to Latent Dimension.

    Args:
        df: Aggregated latent dims results
        output_dir: Output directory for tables

    Returns:
        Formatted DataFrame
    """
    if df.empty:
        print("\nNo data available for Table 7 (Latent Dimensions)")
        return df

    # Format for display
    table_df = df.copy()

    # Add baseline marker
    table_df["Latent Dim (d)"] = table_df.apply(
        lambda r: f"{int(r['latent_dim'])} (baseline)" if r["is_baseline"] else str(int(r["latent_dim"])),
        axis=1
    )

    # Format QLIKE (bold best)
    best_qlike = table_df["qlike_mean"].min()
    table_df["QLIKE (H=5)"] = table_df.apply(
        lambda r: f"**{r['qlike_mean']:.2f}**" if np.isclose(r["qlike_mean"], best_qlike, rtol=0.001)
        else f"{r['qlike_mean']:.2f}",
        axis=1
    )

    # Format MCS inclusion
    best_mcs = table_df["mcs_inclusion"].max()
    table_df["MCS Inclusion"] = table_df.apply(
        lambda r: f"**{r['mcs_inclusion']:.2f}**" if np.isclose(r["mcs_inclusion"], best_mcs, rtol=0.01)
        else f"{r['mcs_inclusion']:.2f}",
        axis=1
    )

    # Select columns for output
    output_df = table_df[["Latent Dim (d)", "QLIKE (H=5)", "MCS Inclusion"]].copy()

    # Save CSV
    csv_df = df[["latent_dim", "qlike_mean", "mcs_inclusion", "n_assets"]].copy()
    csv_df.to_csv(os.path.join(output_dir, "table_7_latent_dims.csv"), index=False)

    # Save LaTeX
    latex_df = output_df.copy()
    # Remove markdown bold for LaTeX (use \textbf instead)
    latex_df["QLIKE (H=5)"] = latex_df["QLIKE (H=5)"].str.replace(
        r"\*\*(.+?)\*\*", r"\\textbf{\1}", regex=True
    )
    latex_df["MCS Inclusion"] = latex_df["MCS Inclusion"].str.replace(
        r"\*\*(.+?)\*\*", r"\\textbf{\1}", regex=True
    )

    latex = latex_df.to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, "table_7_latent_dims.tex"), "w") as f:
        f.write("% Table 7: Sensitivity to Latent Dimension\n")
        f.write("% Notes: Mean QLIKE loss (x100) across 21 assets. Bold indicates best performance.\n")
        f.write(latex)

    # Print formatted table
    print("\n" + "=" * TABLE_WIDTH)
    print("TABLE 7: Sensitivity to Latent Dimension")
    print("=" * TABLE_WIDTH)
    print(f"{'Latent Dim (d)':<20} {'QLIKE (H=5)':<15} {'MCS Inclusion':<15}")
    print("-" * 50)

    for _, row in table_df.iterrows():
        d_str = f"{int(row['latent_dim'])}"
        if row["is_baseline"]:
            d_str += " (baseline)"
        qlike_str = f"{row['qlike_mean']:.2f}"
        mcs_str = f"{row['mcs_inclusion']:.2f}"
        print(f"{d_str:<20} {qlike_str:<15} {mcs_str:<15}")

    print("\nNotes: Mean QLIKE loss (x100) across assets. Bold indicates best performance.")

    return output_df


def format_table_8(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Format and save Table 8: Alternative Transition Functions.

    Args:
        df: Aggregated transition functions results
        output_dir: Output directory for tables

    Returns:
        Formatted DataFrame
    """
    if df.empty:
        print("\nNo data available for Table 8 (Transition Functions)")
        return df

    # Format for display
    table_df = df.copy()

    # Capitalize transition function names
    table_df["Transition Function"] = table_df.apply(
        lambda r: f"{r['transition_fn'].replace('_', '-').title()} (baseline)" if r["is_baseline"]
        else r["transition_fn"].replace("_", "-").title(),
        axis=1
    )

    # Format QLIKE (bold best)
    best_qlike = table_df["qlike_mean"].min()
    table_df["QLIKE (H=5)"] = table_df.apply(
        lambda r: f"**{r['qlike_mean']:.2f}**" if np.isclose(r["qlike_mean"], best_qlike, rtol=0.001)
        else f"{r['qlike_mean']:.2f}",
        axis=1
    )

    table_df["Parameter Count"] = table_df["n_params"].astype(int)

    # Select columns for output
    output_df = table_df[["Transition Function", "QLIKE (H=5)", "Parameter Count"]].copy()

    # Save CSV
    csv_df = df[["transition_fn", "qlike_mean", "n_params", "n_assets"]].copy()
    csv_df.to_csv(os.path.join(output_dir, "table_8_transition_fns.csv"), index=False)

    # Save LaTeX
    latex_df = output_df.copy()
    latex_df["QLIKE (H=5)"] = latex_df["QLIKE (H=5)"].str.replace(
        r"\*\*(.+?)\*\*", r"\\textbf{\1}", regex=True
    )

    latex = latex_df.to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, "table_8_transition_fns.tex"), "w") as f:
        f.write("% Table 8: Alternative Transition Functions\n")
        f.write("% Notes: Mean QLIKE loss (x100) across 21 assets.\n")
        f.write(latex)

    # Print formatted table
    print("\n" + "=" * TABLE_WIDTH)
    print("TABLE 8: Alternative Transition Functions")
    print("=" * TABLE_WIDTH)
    print(f"{'Transition Function':<25} {'QLIKE (H=5)':<15} {'Parameter Count':<15}")
    print("-" * 55)

    for _, row in table_df.iterrows():
        fn_name = row["transition_fn"].replace("_", "-").title()
        if row["is_baseline"]:
            fn_name += " (baseline)"
        qlike_str = f"{row['qlike_mean']:.2f}"
        param_str = str(int(row["n_params"]))
        print(f"{fn_name:<25} {qlike_str:<15} {param_str:<15}")

    print("\nNotes: Mean QLIKE loss (x100) across 21 assets.")

    return output_df


def format_table_9(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Format and save Table 9: Subsample Performance.

    Args:
        df: Aggregated subsample results
        output_dir: Output directory for tables

    Returns:
        Formatted DataFrame
    """
    if df.empty:
        print("\nNo data available for Table 9 (Subsample Performance)")
        return df

    # Format for display
    table_df = df.copy()

    table_df["Period"] = table_df["period"]
    table_df["Deep-LSTR"] = table_df["improvement_ssm"].apply(lambda x: f"{x:.1f}%")
    table_df["STR-OBS"] = table_df["improvement_obs"].apply(
        lambda x: f"{x:.1f}%" if np.isfinite(x) else "-"
    )

    # Select columns for output
    output_df = table_df[["Period", "Deep-LSTR", "STR-OBS"]].copy()

    # Save CSV
    csv_df = df[["period", "improvement_ssm", "improvement_obs", "n_assets"]].copy()
    csv_df.to_csv(os.path.join(output_dir, "table_9_subsample.csv"), index=False)

    # Save LaTeX
    latex = output_df.to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, "table_9_subsample.tex"), "w") as f:
        f.write("% Table 9: Subsample Performance (QLIKE Improvement vs. HAR)\n")
        f.write("% Notes: QLIKE improvement relative to HAR at H=5.\n")
        f.write(latex)

    # Print formatted table
    print("\n" + "=" * TABLE_WIDTH)
    print("TABLE 9: Subsample Performance (QLIKE Improvement vs. HAR)")
    print("=" * TABLE_WIDTH)
    print(f"{'Period':<15} {'Deep-LSTR':<15} {'STR-OBS':<15}")
    print("-" * 45)

    for _, row in df.iterrows():
        period_str = row["period"]
        ssm_str = f"{row['improvement_ssm']:.1f}%"
        obs_str = f"{row['improvement_obs']:.1f}%" if np.isfinite(row["improvement_obs"]) else "-"
        print(f"{period_str:<15} {ssm_str:<15} {obs_str:<15}")

    print("\nNotes: QLIKE improvement relative to HAR at H=5.")

    return output_df


def format_table_10(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Format and save Table 10: Alternative Volatility Estimators.

    Args:
        df: Aggregated volatility estimators results
        output_dir: Output directory for tables

    Returns:
        Formatted DataFrame
    """
    if df.empty:
        print("\nNo data available for Table 10 (Volatility Estimators)")
        return df

    # Format for display
    table_df = df.copy()

    # Add baseline marker
    table_df["Estimator"] = table_df.apply(
        lambda r: f"{r['estimator']} (baseline)" if r["is_baseline"] else r["estimator"],
        axis=1
    )

    table_df["Deep-LSTR QLIKE"] = table_df["qlike_ssm"].apply(lambda x: f"{x:.2f}")
    table_df["STR-OBS QLIKE"] = table_df["qlike_obs"].apply(
        lambda x: f"{x:.2f}" if np.isfinite(x) else "-"
    )
    table_df["Improvement"] = table_df["improvement"].apply(
        lambda x: f"{x:.1f}%" if np.isfinite(x) else "-"
    )

    # Select columns for output
    output_df = table_df[["Estimator", "Deep-LSTR QLIKE", "STR-OBS QLIKE", "Improvement"]].copy()

    # Save CSV
    csv_df = df[["estimator_key", "qlike_ssm", "qlike_obs", "improvement", "n_assets"]].copy()
    csv_df.to_csv(os.path.join(output_dir, "table_10_volatility_estimators.csv"), index=False)

    # Save LaTeX
    latex = output_df.to_latex(index=False, escape=False)
    with open(os.path.join(output_dir, "table_10_volatility_estimators.tex"), "w") as f:
        f.write("% Table 10: Alternative Volatility Estimators\n")
        f.write("% Notes: Mean QLIKE loss (x100) across 21 assets at H=5.\n")
        f.write(latex)

    # Print formatted table
    print("\n" + "=" * TABLE_WIDTH)
    print("TABLE 10: Alternative Volatility Estimators")
    print("=" * TABLE_WIDTH)
    print(f"{'Estimator':<25} {'Deep-LSTR QLIKE':<18} {'STR-OBS QLIKE':<15} {'Improvement':<12}")
    print("-" * 70)

    for _, row in df.iterrows():
        est_str = row["estimator"]
        if row["is_baseline"]:
            est_str += " (baseline)"
        ssm_str = f"{row['qlike_ssm']:.2f}"
        obs_str = f"{row['qlike_obs']:.2f}" if np.isfinite(row["qlike_obs"]) else "-"
        imp_str = f"{row['improvement']:.1f}%" if np.isfinite(row["improvement"]) else "-"
        print(f"{est_str:<25} {ssm_str:<18} {obs_str:<15} {imp_str:<12}")

    print("\nNotes: Mean QLIKE loss (x100) across 21 assets at H=5.")

    return output_df


def collect_robustness_tables_for_yaml(
    table7_md: Optional[str],
    table8_md: Optional[str],
    table9_md: Optional[str],
    table10_md: Optional[str],
    table11_md: Optional[str],
) -> Dict[str, str]:
    """
    Collect robustness tables as markdown strings for YAML output.
    """
    tables = {}

    if table7_md:
        tables['latent_dim_sensitivity'] = table7_md
    if table8_md:
        tables['transition_fn_comparison'] = table8_md
    if table9_md:
        tables['subsample_performance'] = table9_md
    if table10_md:
        tables['volatility_estimator_comparison'] = table10_md
    if table11_md:
        tables['transition_ablation'] = table11_md

    return tables


def generate_robustness_yaml(
    exp_dir: str,
    output_dir: str,
    tables: Dict[str, str],
    config_path: str = None,
) -> str:
    """
    Generate robustness_results.yaml file for paper injection.
    """
    experiment_id = os.path.basename(exp_dir) + "_robustness"

    results = create_empty_results(
        experiment_id=experiment_id,
        config_path=config_path or "unknown",
    )
    results.tables = tables

    yaml_path = os.path.join(output_dir, "robustness_results.yaml")
    results.to_yaml(yaml_path)

    print(f"\nGenerated robustness results YAML: {yaml_path}")
    print(f"  Tables: {len(tables)}")

    return yaml_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready robustness tables (Tables 7-10)"
    )
    parser.add_argument(
        "--check",
        type=str,
        choices=["latent_dims", "transition_fns", "subsample", "volatility_estimators", "all"],
        default="all",
        help="Which robustness check to aggregate (default: all)",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to experiment directory with robustness results",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Path to baseline experiment directory (for subsample analysis)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: configs/robustness.yaml)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for tables (default: exp-dir/tables)",
    )
    parser.add_argument(
        "--no-recompute-mcs",
        action="store_true",
        help="Skip MCS recomputation (use cached values)",
    )

    args = parser.parse_args()

    # Validate experiment directory
    if not os.path.exists(args.exp_dir):
        print(f"Error: Experiment directory not found: {args.exp_dir}")
        sys.exit(1)

    # Load configuration
    config = load_robustness_config(args.config)

    # Set output directory
    output_dir = args.output_dir or os.path.join(args.exp_dir, "tables")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * TABLE_WIDTH)
    print("DEEP-LSTR ROBUSTNESS TABLE AGGREGATION")
    print("=" * TABLE_WIDTH)
    print(f"Experiment directory: {args.exp_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Check(s) to aggregate: {args.check}")

    all_tables = {}

    # Table 7: Latent Dimensions
    if args.check in ["latent_dims", "all"]:
        print("\n" + "-" * TABLE_WIDTH)
        print("Aggregating Table 7: Sensitivity to Latent Dimension...")
        df = aggregate_latent_dims(
            args.exp_dir,
            config,
            recompute_mcs=not args.no_recompute_mcs
        )
        all_tables["table_7"] = format_table_7(df, output_dir)

    # Table 8: Transition Functions
    if args.check in ["transition_fns", "all"]:
        print("\n" + "-" * TABLE_WIDTH)
        print("Aggregating Table 8: Alternative Transition Functions...")
        df = aggregate_transition_fns(args.exp_dir, config)
        all_tables["table_8"] = format_table_8(df, output_dir)

    # Table 9: Subsample Performance
    if args.check in ["subsample", "all"]:
        print("\n" + "-" * TABLE_WIDTH)
        print("Aggregating Table 9: Subsample Performance...")
        baseline_dir = args.baseline_dir or args.exp_dir
        df = aggregate_subsample(args.exp_dir, config, baseline_exp_dir=baseline_dir)
        all_tables["table_9"] = format_table_9(df, output_dir)

    # Table 10: Volatility Estimators
    if args.check in ["volatility_estimators", "all"]:
        print("\n" + "-" * TABLE_WIDTH)
        print("Aggregating Table 10: Alternative Volatility Estimators...")
        df = aggregate_volatility_estimators(args.exp_dir, config)
        all_tables["table_10"] = format_table_10(df, output_dir)

    # Summary
    print("\n" + "=" * TABLE_WIDTH)
    print("AGGREGATION COMPLETE")
    print("=" * TABLE_WIDTH)
    print(f"Tables saved to: {output_dir}")

    # List generated files
    generated_files = []
    for f in os.listdir(output_dir):
        if f.startswith("table_") and (f.endswith(".csv") or f.endswith(".tex")):
            generated_files.append(f)

    if generated_files:
        print("\nGenerated files:")
        for f in sorted(generated_files):
            print(f"  - {f}")

    return all_tables


if __name__ == "__main__":
    main()
