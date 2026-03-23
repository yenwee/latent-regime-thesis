#!/usr/bin/env python3
"""
Risk evaluation script for VaR/ES backtesting.

This script evaluates risk forecasts (VaR and ES) using the complete pipeline
- Rolling k_t volatility scaling
- Rolling nu (degrees of freedom) estimation via MLE
- Mean adjustment (muH)
- Non-overlapping H-day blocks for evaluation
- FZ0 loss (proper scoring rule for joint VaR+ES)
- Kupiec test for VaR coverage
- DM tests on FZ0 losses

The script can either:
1. Load existing results from a CSV file
2. Run the full forecasting pipeline first (via run_single_asset)

Example usage:
    # Evaluate risk from existing results
    python scripts/run_risk_evaluation.py --results results/GSPC_H5_results.csv --horizon 5

    # Run full pipeline including risk evaluation
    python scripts/run_risk_evaluation.py --ticker ^GSPC --horizon 5 --run-forecasts

    # Customize risk parameters
    python scripts/run_risk_evaluation.py --results results.csv --horizon 5 \\
        --alphas 0.01 0.025 0.05 --no-rolling-k --overlapping
"""

import argparse
import os
import sys
import importlib.util

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import yaml


def _load_module_directly(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_risk_module = _load_module_directly(
    "src.risk",
    os.path.join(PROJECT_ROOT, "src", "risk.py")
)

run_risk_evaluation = _risk_module.run_risk_evaluation
risk_series_var_es_dynamic = _risk_module.risk_series_var_es_dynamic
fz0_loss = _risk_module.fz0_loss
compute_rolling_k = _risk_module.compute_rolling_k
compute_rolling_nu = _risk_module.compute_rolling_nu
DEFAULT_ALPHAS = _risk_module.DEFAULT_ALPHAS
EPS = _risk_module.EPS


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "default.yaml")

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # Return minimal default config
    return {
        "output": {"base_dir": "."},
        "data": {"start": "2005-01-01", "end": "2024-12-31", "interval": "1d"},
    }


def ensure_writable_dir(preferred_dir):
    """Ensure directory exists and is writable."""
    preferred_dir = preferred_dir or "."
    try:
        os.makedirs(preferred_dir, exist_ok=True)
        test_path = os.path.join(preferred_dir, "__write_test__.tmp")
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        return preferred_dir
    except Exception:
        fallback = "/tmp/results"
        os.makedirs(fallback, exist_ok=True)
        print(f"\n[WARN] OUTPUT_DIR='{preferred_dir}' not writable. Falling back to '{fallback}'.")
        return fallback


def load_results_with_risk_params(
    results_path,
    H,
    use_rolling_k=True,
    rolling_nu=True,
    mean_adjust=True,
):
    """
    Load results CSV and prepare risk parameters.

    If the CSV already has k_*, nu_*, and muH columns, use them.
    Otherwise, compute them from available data (limited capability).

    Args:
        results_path: Path to results CSV
        H: Forecast horizon
        use_rolling_k: Whether to use rolling k_t
        rolling_nu: Whether to use rolling nu
        mean_adjust: Whether to use mean adjustment

    Returns:
        DataFrame ready for risk evaluation
    """
    df = pd.read_csv(results_path, index_col=0, parse_dates=True)

    # Check for required columns
    required = ["rH"]
    model_cols = []
    for col in ["har", "str_obs", "str_ssm", "garch", "egarch", "msgarch"]:
        if col in df.columns:
            model_cols.append(col)

    if not model_cols:
        raise ValueError(f"No model forecast columns found in {results_path}")

    if "rH" not in df.columns:
        # Try to compute rH from r_fwd_sum_{H} if available
        rH_col = f"r_fwd_sum_{H}"
        if rH_col in df.columns:
            df["rH"] = df[rH_col]
        elif "y" in df.columns:
            # Cannot compute rH from log variance target
            raise ValueError(
                f"Results file missing 'rH' column and cannot be derived. "
                f"Expected columns: rH (H-day return) or r_fwd_sum_{H}"
            )

    return df


def print_risk_summary(risk_tables, dm_tests, verbose=True):
    """Print formatted risk evaluation summary."""
    if verbose:
        print("\n" + "=" * 70)
        print("RISK EVALUATION SUMMARY")
        print("=" * 70)

        if not risk_tables.empty:
            print("\nFZ0 Loss and Coverage by Model and Alpha:")
            print("-" * 70)

            # Pivot for nice display
            for alpha in risk_tables["alpha"].unique():
                subset = risk_tables[risk_tables["alpha"] == alpha]
                print(f"\n  Alpha = {alpha}:")
                print(f"  {'Model':<10} {'Violations':<12} {'Rate':<10} {'Kupiec p':<12} {'FZ0':<12}")
                print(f"  {'-'*10} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
                for _, row in subset.iterrows():
                    print(
                        f"  {row['model']:<10} "
                        f"{row['violations']:>5}/{row['T']:<5} "
                        f"{row['viol_rate']:>8.4f}  "
                        f"{row['kupiec_p']:>10.4f}  "
                        f"{row['mean_FZ0']:>10.4f}"
                    )

        if not dm_tests.empty:
            print("\n\nDiebold-Mariano Tests on FZ0 Loss:")
            print("-" * 70)
            print(f"  {'Model A':<12} {'Model B':<12} {'Alpha':<8} {'DM Stat':<12} {'p-value':<12}")
            print(f"  {'-'*12} {'-'*12} {'-'*8} {'-'*12} {'-'*12}")
            for _, row in dm_tests.iterrows():
                sig = ""
                if row["dm_pval"] < 0.01:
                    sig = "***"
                elif row["dm_pval"] < 0.05:
                    sig = "**"
                elif row["dm_pval"] < 0.10:
                    sig = "*"
                print(
                    f"  {row['model_a']:<12} "
                    f"{row['model_b']:<12} "
                    f"{row['alpha']:<8.3f} "
                    f"{row['dm_stat']:>10.3f}  "
                    f"{row['dm_pval']:>10.4f} {sig}"
                )

        print("\n" + "=" * 70)


def run_risk_pipeline(
    results_path=None,
    ticker=None,
    H=5,
    config=None,
    output_dir=None,
    alphas=None,
    use_rolling_k=True,
    rolling_nu=True,
    mean_adjust=True,
    nonoverlapping=True,
    run_forecasts=False,
    verbose=True,
):
    """
    Run complete risk evaluation pipeline.

    Args:
        results_path: Path to existing results CSV
        ticker: Asset ticker (if running forecasts)
        H: Forecast horizon
        config: Configuration dictionary
        output_dir: Output directory
        alphas: List of VaR/ES alpha levels
        use_rolling_k: Use rolling volatility scaling
        rolling_nu: Use rolling nu estimation
        mean_adjust: Use mean adjustment
        nonoverlapping: Use non-overlapping H-day blocks
        run_forecasts: Run forecasting pipeline first
        verbose: Print progress

    Returns:
        Dictionary with risk evaluation results
    """
    if config is None:
        config = load_config()

    if alphas is None:
        alphas = DEFAULT_ALPHAS

    # Ensure output directory
    if output_dir is None:
        output_dir = config.get("output", {}).get("base_dir", ".")
    save_dir = ensure_writable_dir(output_dir)

    # Load or generate results
    if run_forecasts:
        if ticker is None:
            raise ValueError("--ticker required when --run-forecasts is set")

        if verbose:
            print(f"Running forecast pipeline for {ticker} (H={H})...")

        # Import and run single asset pipeline
        from scripts.run_single_asset import run_single_asset

        output = run_single_asset(
            ticker=ticker,
            H=H,
            config=config,
            output_dir=save_dir,
            verbose=verbose,
        )
        df_res = output["results"]

        # The run_single_asset doesn't compute rH, k_t, nu_t by default
        # We need to check if those columns exist
        if "rH" not in df_res.columns:
            if verbose:
                print("\nWarning: rH column not in results. Risk evaluation requires rH.")
                print("Please ensure the results include H-day returns for risk evaluation.")
            return None

    else:
        if results_path is None:
            raise ValueError("Either --results or --run-forecasts must be specified")

        if verbose:
            print(f"Loading results from {results_path}...")

        df_res = load_results_with_risk_params(
            results_path, H, use_rolling_k, rolling_nu, mean_adjust
        )

    # Identify available models
    model_map = {
        "har": "har",
        "obs": "str_obs",
        "ssm": "str_ssm",
        "garch": "garch",
        "egarch": "egarch",
        "msgarch": "msgarch",
    }
    models = []
    for short_name, col_name in model_map.items():
        if col_name in df_res.columns:
            models.append(short_name)

    if not models:
        raise ValueError("No model forecast columns found in results")

    if verbose:
        print(f"\nFound models: {models}")
        print(f"Total observations: {len(df_res)}")
        print(f"Alphas: {alphas}")
        print(f"Settings: rolling_k={use_rolling_k}, rolling_nu={rolling_nu}, "
              f"mean_adjust={mean_adjust}, nonoverlapping={nonoverlapping}")

    # DM lag
    dm_lag = max(20, 2 * H)

    # Run risk evaluation
    if verbose:
        print("\nRunning risk evaluation...")

    results = run_risk_evaluation(
        df_res=df_res,
        models=models,
        H=H,
        alphas=alphas,
        use_rolling_k=use_rolling_k,
        rolling_nu=rolling_nu,
        mean_adjust=mean_adjust,
        nonoverlapping=nonoverlapping,
        dm_lag=dm_lag,
    )

    # Print summary
    print_risk_summary(results["risk_tables"], results["dm_tests"], verbose)

    # Save results
    if not results["risk_tables"].empty:
        risk_path = os.path.join(save_dir, f"risk_tables_H{H}.csv")
        results["risk_tables"].to_csv(risk_path, index=False)
        if verbose:
            print(f"\nRisk tables saved to: {risk_path}")

    if not results["dm_tests"].empty:
        dm_path = os.path.join(save_dir, f"risk_dm_tests_H{H}.csv")
        results["dm_tests"].to_csv(dm_path, index=False)
        if verbose:
            print(f"DM tests saved to: {dm_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Risk evaluation for VaR/ES forecasts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate from existing results
  python scripts/run_risk_evaluation.py --results results/GSPC_H5_results.csv --horizon 5

  # Custom alpha levels
  python scripts/run_risk_evaluation.py --results results.csv --horizon 5 --alphas 0.01 0.025 0.05

  # Disable non-overlapping (use all observations)
  python scripts/run_risk_evaluation.py --results results.csv --horizon 5 --overlapping
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--results",
        type=str,
        help="Path to results CSV file",
    )
    input_group.add_argument(
        "--run-forecasts",
        action="store_true",
        help="Run forecasting pipeline first (requires --ticker)",
    )

    parser.add_argument(
        "--ticker",
        type=str,
        help="Asset ticker (required with --run-forecasts)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        required=True,
        help="Forecast horizon in days",
    )

    # Risk parameters
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help="VaR/ES alpha levels (default: 0.01 0.05)",
    )
    parser.add_argument(
        "--no-rolling-k",
        action="store_true",
        help="Disable rolling k_t scaling",
    )
    parser.add_argument(
        "--no-rolling-nu",
        action="store_true",
        help="Disable rolling nu estimation",
    )
    parser.add_argument(
        "--no-mean-adjust",
        action="store_true",
        help="Disable mean adjustment",
    )
    parser.add_argument(
        "--overlapping",
        action="store_true",
        help="Use overlapping observations (default: non-overlapping H-day blocks)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.run_forecasts and args.ticker is None:
        parser.error("--ticker is required when using --run-forecasts")

    if not args.run_forecasts and args.results is None:
        parser.error("Either --results or --run-forecasts must be specified")

    # Load config
    config = load_config(args.config) if args.config else load_config()

    # Run pipeline
    run_risk_pipeline(
        results_path=args.results,
        ticker=args.ticker,
        H=args.horizon,
        config=config,
        output_dir=args.output_dir,
        alphas=args.alphas,
        use_rolling_k=not args.no_rolling_k,
        rolling_nu=not args.no_rolling_nu,
        mean_adjust=not args.no_mean_adjust,
        nonoverlapping=not args.overlapping,
        run_forecasts=args.run_forecasts,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
