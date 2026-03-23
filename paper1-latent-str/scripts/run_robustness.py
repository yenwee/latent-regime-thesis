#!/usr/bin/env python3
"""
Run Deep-LSTR robustness checks with checkpoint resumability.

Implements the four robustness checks from Section 6 of the JFEC paper:
- 6.1 Alternative Latent Dimensions (d=1,2,4,8)
- 6.2 Alternative Transition Functions (logistic, exponential, double-logistic)
- 6.3 Subsample Stability (rolling 2-year windows)
- 6.4 Alternative Volatility Estimators (Garman-Klass, Parkinson, Rogers-Satchell)

Example usage:
    python scripts/run_robustness.py --check latent_dims
    python scripts/run_robustness.py --check volatility_estimators --n-jobs 4
    python scripts/run_robustness.py --all
"""

import argparse
import os
import sys
import datetime
import pickle

# Prevent thread oversubscription when using joblib parallelism.
# Each worker gets 1 thread; parallelism comes from joblib instead.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import yaml

from src.utils import (
    set_seeds,
    load_config,
    get_experiment_dir,
    check_segment_checkpoint,
    get_completed_segments,
)
from src.data import download_asset_data, prepare_features, VOLATILITY_ESTIMATORS
from src.str_har import (
    TRANSITION_FUNCTIONS,
    fit_har_ols,
    har_predict,
    fit_str2_window_robust,
    str2_forecast_one,
)
from src.metrics import qlike
from scripts.run_single_asset import run_single_asset
from scripts.run_panel import pretrain_vrnn_for_asset

import subprocess as _subprocess
from concurrent.futures import ThreadPoolExecutor


def _run_single_asset_subprocess(ticker, H, config_path, exp_dir):
    """Run forecasting as isolated subprocess to avoid PyTorch VSIZE on Apple Silicon.

    Uses the config file at config_path (which may be a variant-specific temp config).
    Output is tagged with RESULT_JSON: prefix to avoid parsing issues from verbose output.
    """
    import json as _json
    # Use a unique marker so we can find the JSON line reliably
    marker = "RESULT_JSON:"
    cmd = [
        sys.executable, "-u", "-c",
        f"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
import sys, json, traceback
sys.path.insert(0, '{PROJECT_ROOT}')
from scripts.run_single_asset import run_single_asset
from src.utils import load_config
config = load_config('{config_path}')
try:
    result = run_single_asset('{ticker}', H={H}, config=config, exp_dir='{exp_dir}', verbose=False)
    qlike = result.get('qlike', {{}})
    mcs = result.get('mcs', {{}})
    print('{marker}' + json.dumps({{'ticker': '{ticker}', 'H': {H}, 'qlike': qlike, 'mcs': mcs}}))
except Exception as e:
    traceback.print_exc(file=sys.stderr)
    print('{marker}' + json.dumps({{'ticker': '{ticker}', 'H': {H}, 'error': str(e)}}))
"""
    ]
    try:
        result = _subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except _subprocess.TimeoutExpired:
        return {"ticker": ticker, "H": H, "error": "Subprocess timeout (1800s)"}

    if result.returncode != 0:
        return {"ticker": ticker, "H": H, "error": f"rc={result.returncode}: {result.stderr[-500:]}"}

    # Find the tagged result line
    for line in reversed(result.stdout.strip().split('\n')):
        if line.startswith(marker):
            try:
                return _json.loads(line[len(marker):])
            except Exception:
                pass

    return {"ticker": ticker, "H": H, "error": f"No result line found. stdout[-500:]: {result.stdout[-500:]}"}


def _get_asset_checkpoint_path(variant_exp_dir, ticker, H):
    """Get per-asset checkpoint path within a variant directory."""
    safe_ticker = ticker.replace("=", "").replace("^", "").replace("/", "")
    ckpt_dir = os.path.join(variant_exp_dir, "asset_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f"{safe_ticker}_H{H}_result.pkl")


def _load_completed_assets(variant_exp_dir, assets, H):
    """Load per-asset results that already completed. Returns (completed_results, remaining_assets)."""
    completed = []
    remaining = []
    for ticker in assets:
        ckpt = _get_asset_checkpoint_path(variant_exp_dir, ticker, H)
        if os.path.exists(ckpt):
            try:
                with open(ckpt, "rb") as f:
                    result = pickle.load(f)
                if "error" not in result:
                    completed.append(result)
                    continue
            except Exception:
                pass
        remaining.append(ticker)
    return completed, remaining


def _save_asset_checkpoint(variant_exp_dir, ticker, H, result):
    """Save a single asset result for crash recovery."""
    ckpt = _get_asset_checkpoint_path(variant_exp_dir, ticker, H)
    with open(ckpt, "wb") as f:
        pickle.dump(result, f)


def _pipelined_run_variant(assets, H, variant_config, variant_exp_dir, config_path, n_jobs=10, verbose=False):
    """
    Pipelined execution for a robustness variant with per-asset checkpointing:
    1. Load any previously completed assets (crash recovery)
    2. Train VRNNs sequentially for remaining assets (safe, imports torch)
    3. Run forecasts in parallel subprocesses (no torch)
    4. Save each asset result as it completes
    """
    import json

    # Load previously completed assets
    completed_results, remaining_assets = _load_completed_assets(variant_exp_dir, assets, H)
    if completed_results:
        print(f"    [RESUME] {len(completed_results)}/{len(assets)} assets already done, "
              f"{len(remaining_assets)} remaining")

    if not remaining_assets:
        return completed_results

    # Write variant config to temp file for subprocesses
    tmp_config = os.path.join(variant_exp_dir, "_variant_config.yaml")
    import yaml as _yaml
    with open(tmp_config, 'w') as f:
        _yaml.dump(variant_config, f)

    # Phase 1: VRNN training sequentially for remaining assets only
    for ticker in remaining_assets:
        try:
            pretrain_vrnn_for_asset(ticker, H, variant_config, variant_exp_dir, verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"    VRNN ERROR {ticker}: {e}")

    # Phase 2: Forecast in parallel subprocesses for remaining assets only
    new_results = []
    failed_assets = []
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        futures = {
            pool.submit(_run_single_asset_subprocess, ticker, H, tmp_config, variant_exp_dir): ticker
            for ticker in remaining_assets
        }
        for future in futures:
            ticker = futures[future]
            try:
                r = future.result(timeout=1800)
                if "error" not in r:
                    new_results.append(r)
                    _save_asset_checkpoint(variant_exp_dir, ticker, H, r)
                else:
                    failed_assets.append((ticker, r.get("error", "unknown")))
            except Exception as e:
                failed_assets.append((ticker, str(e)))

    # Retry failed assets sequentially (often fixes transient issues)
    if failed_assets:
        print(f"    [RETRY] {len(failed_assets)} assets failed, retrying sequentially...")
        for ticker, prev_error in failed_assets:
            try:
                r = _run_single_asset_subprocess(ticker, H, tmp_config, variant_exp_dir)
                if "error" not in r:
                    new_results.append(r)
                    _save_asset_checkpoint(variant_exp_dir, ticker, H, r)
                    print(f"    [RETRY] {ticker}: SUCCESS")
                else:
                    print(f"    [RETRY] {ticker}: FAILED again - {r.get('error', '')[:100]}")
            except Exception as e:
                print(f"    [RETRY] {ticker}: EXCEPTION - {e}")

    # Cleanup temp config
    if os.path.exists(tmp_config):
        os.remove(tmp_config)

    all_results = completed_results + [r for r in new_results if "error" not in r]
    return all_results


def load_baseline_q_ssm(exp_dir: str, ticker: str, H: int) -> tuple:
    """
    Load q_ssm series from baseline segment checkpoints.

    Args:
        exp_dir: Baseline experiment directory
        ticker: Asset ticker
        H: Forecast horizon

    Returns:
        Tuple of (q_ssm_full pd.Series, segment_bounds list) or (None, None) if not found
    """
    # Get completed segments from baseline
    completed_segments = get_completed_segments(exp_dir, ticker, H)
    if not completed_segments:
        return None, None

    # Load all segment q_ssm values
    q_ssm_parts = []
    segment_bounds = []

    for seg_id in sorted(completed_segments):
        seg_checkpoint = check_segment_checkpoint(exp_dir, ticker, H, seg_id)
        if seg_checkpoint is None:
            continue

        q_series = seg_checkpoint["q_ssm_segment"]
        q_ssm_parts.append(q_series)
        segment_bounds.append((seg_checkpoint["seg_start"], seg_checkpoint["seg_end"]))

    if not q_ssm_parts:
        return None, None

    # Combine all segments into one series
    q_ssm_full = pd.concat(q_ssm_parts)
    # Remove duplicates (overlapping segments), keep last value
    q_ssm_full = q_ssm_full[~q_ssm_full.index.duplicated(keep="last")]
    q_ssm_full = q_ssm_full.sort_index()

    return q_ssm_full, segment_bounds


def run_str_har_with_transition(
    ticker: str,
    H: int,
    transition_fn: str,
    config: dict,
    baseline_exp_dir: str,
    verbose: bool = False,
) -> dict:
    """
    Run STR-HAR forecasting with a specific transition function, reusing SSM from baseline.

    This function skips SSM training entirely and only runs the STR-HAR fitting
    with the specified transition function. This is much faster than a full run.

    Args:
        ticker: Asset ticker
        H: Forecast horizon
        transition_fn: Transition function name
        config: Configuration dictionary
        baseline_exp_dir: Experiment directory with baseline SSM checkpoints
        verbose: Print progress

    Returns:
        Dictionary with QLIKE results
    """
    from src.garch import Garch11_t, Egarch11_t, MSGarch2_t

    # Load q_ssm from baseline checkpoints
    q_ssm_full, segment_bounds = load_baseline_q_ssm(baseline_exp_dir, ticker, H)
    if q_ssm_full is None:
        if verbose:
            print(f"    No baseline checkpoints found for {ticker}")
        return None

    # Download and prepare data (same as baseline)
    df = download_asset_data(
        ticker,
        start=config["data"]["start"],
        end=config["data"]["end"],
        interval=config["data"]["interval"],
    )

    volatility_estimator = config.get("data", {}).get("volatility_estimator", "garman_klass")
    df = prepare_features(
        df, H,
        q_obs_smooth_span=config["smoothing"]["q_obs_span"],
        volatility_estimator=volatility_estimator,
    )

    # Add q_ssm to dataframe
    df["q_ssm"] = q_ssm_full.reindex(df.index)

    # Define rolling window parameters
    roll_window = config["rolling"]["window"]
    min_start = config["rolling"]["min_start"]
    str_config = config["str_har"]
    seed = config.get("seed", 123)

    # Find OOS start
    start_base = max(min_start, roll_window)

    results = []

    # Run forecasting loop (only STR-HAR, skip GARCH for speed)
    for i in range(start_base, len(df) - 1):
        if i - roll_window < 0:
            continue

        train_i = df.iloc[i - roll_window:i].copy()
        test_i = df.iloc[i:i + 1].copy()

        # Skip if q_ssm is missing
        if train_i["q_ssm"].isna().any() or test_i["q_ssm"].isna().any():
            continue

        # Fit HAR (same for all transition functions)
        b = fit_har_ols(train_i)
        yh_har = float(har_predict(b, test_i)[0])

        # Fit STR-HAR with observable transition
        p_obs, mu_obs, sd_obs = fit_str2_window_robust(
            train_i,
            q_col="q_obs",
            use_basinhopping=str_config["use_basinhopping"],
            bh_niter=str_config["basinhopping_niter"],
            n_starts=str_config["n_starts"],
            gamma_max=str_config["gamma_max"],
            gamma_lam=str_config["gamma_lambda"],
            transition_fn=transition_fn,
        )
        yh_obs, g_obs = str2_forecast_one(
            p_obs, mu_obs, sd_obs, test_i, q_col="q_obs", transition_fn=transition_fn
        )

        # Fit STR-HAR with latent transition (using baseline q_ssm)
        p_ssm, mu_ssm, sd_ssm = fit_str2_window_robust(
            train_i,
            q_col="q_ssm",
            use_basinhopping=str_config["use_basinhopping"],
            bh_niter=str_config["basinhopping_niter"],
            n_starts=str_config["n_starts"],
            gamma_max=str_config["gamma_max"],
            gamma_lam=str_config["gamma_lambda"],
            transition_fn=transition_fn,
        )
        yh_ssm, g_ssm = str2_forecast_one(
            p_ssm, mu_ssm, sd_ssm, test_i, q_col="q_ssm", transition_fn=transition_fn
        )

        results.append({
            "date": test_i.index[0],
            "y": float(test_i["y"].values[0]),
            "har": yh_har,
            "str_obs": yh_obs,
            "str_ssm": yh_ssm,
            "G_obs": g_obs,
            "G_ssm": g_ssm,
        })

    if not results:
        return None

    res_df = pd.DataFrame(results).set_index("date")

    # Compute QLIKE
    qlike_har = float(qlike(res_df["y"], res_df["har"]).mean())
    qlike_obs = float(qlike(res_df["y"], res_df["str_obs"]).mean())
    qlike_ssm = float(qlike(res_df["y"], res_df["str_ssm"]).mean())

    return {
        "ticker": ticker,
        "transition_fn": transition_fn,
        "n_forecasts": len(results),
        "qlike_har": qlike_har,
        "qlike_obs": qlike_obs,
        "qlike_ssm": qlike_ssm,
    }


def get_robustness_checkpoint_path(exp_dir: str, check_name: str, variant: str) -> str:
    """Get checkpoint path for a robustness check variant."""
    checkpoint_dir = os.path.join(exp_dir, "robustness", check_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    safe_variant = str(variant).replace(".", "_").replace("/", "_")
    return os.path.join(checkpoint_dir, f"{safe_variant}_results.pkl")


def save_robustness_checkpoint(path: str, data: dict, verbose: bool = False):
    """Save robustness check results."""
    with open(path, "wb") as f:
        pickle.dump(data, f)
    if verbose:
        print(f"  [CHECKPOINT] Saved: {path}")


def load_robustness_checkpoint(path: str, verbose: bool = False) -> dict:
    """Load robustness check results if exists."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    if verbose:
        print(f"  [CHECKPOINT] Loaded: {path}")
    return data


def is_variant_complete(exp_dir: str, check_name: str, variant: str) -> bool:
    """Check if a variant run is complete."""
    path = get_robustness_checkpoint_path(exp_dir, check_name, variant)
    return os.path.exists(path)


def run_latent_dims_check(
    config: dict,
    exp_dir: str,
    n_jobs: int = 3,
    resume: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run robustness check 6.1: Alternative Latent Dimensions.

    Tests d ∈ {1, 2, 4, 8} and reports QLIKE and MCS inclusion rates.
    """
    rob_config = config["robustness"]["latent_dims"]
    if not rob_config.get("enabled", True):
        print("  Latent dims check disabled in config")
        return {}

    latent_dims = rob_config["values"]
    H = rob_config.get("horizon", 5)

    print(f"\n{'='*60}")
    print(f"ROBUSTNESS CHECK 6.1: Alternative Latent Dimensions")
    print(f"{'='*60}")
    print(f"Testing d ∈ {latent_dims} at H={H}")

    results = {}

    for d in latent_dims:
        variant_name = f"latent_dim_{d}"

        # Check if already complete
        if resume and is_variant_complete(exp_dir, "latent_dims", variant_name):
            checkpoint = load_robustness_checkpoint(
                get_robustness_checkpoint_path(exp_dir, "latent_dims", variant_name),
                verbose=verbose
            )
            results[d] = checkpoint
            print(f"\n  d={d}: RESUMED from checkpoint")
            continue

        print(f"\n  Running d={d}... (pipelined: sequential VRNN + parallel forecast)")

        # Create variant config
        variant_config = config.copy()
        variant_config["ssm"] = config["ssm"].copy()
        variant_config["ssm"]["latent_dim"] = d

        # Create variant experiment directory
        variant_exp_dir = os.path.join(exp_dir, "robustness", "latent_dims", f"d{d}")
        os.makedirs(variant_exp_dir, exist_ok=True)

        # Pipelined: sequential VRNN training + parallel subprocess forecasting
        config_path = os.path.join(PROJECT_ROOT, "configs", "robustness.yaml")
        asset_results = _pipelined_run_variant(
            config["asset_basket"], H, variant_config, variant_exp_dir,
            config_path, n_jobs=n_jobs, verbose=verbose,
        )

        if len(asset_results) > 0:
            qlike_values = [r["qlike"]["str_ssm"] for r in asset_results if "qlike" in r]
            mcs_included = [1 for r in asset_results if "mcs" in r and "str_ssm" in r["mcs"].get("models", [])]

            variant_result = {
                "latent_dim": d,
                "n_assets": len(asset_results),
                "mean_qlike": np.median(qlike_values) if qlike_values else np.nan,
                "mcs_inclusion_rate": len(mcs_included) / len(asset_results) if asset_results else 0,
                "asset_results": asset_results,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        else:
            variant_result = {"latent_dim": d, "error": "No successful runs"}

        results[d] = variant_result

        # Save checkpoint
        if resume:
            save_robustness_checkpoint(
                get_robustness_checkpoint_path(exp_dir, "latent_dims", variant_name),
                variant_result,
                verbose=verbose
            )

        print(f"    QLIKE: {variant_result.get('mean_qlike', np.nan):.4f}, "
              f"MCS: {variant_result.get('mcs_inclusion_rate', 0):.2%}")

    # Generate summary table
    _print_latent_dims_table(results, rob_config["baseline"])

    return results


def run_transition_fns_check(
    config: dict,
    exp_dir: str,
    baseline_exp_dir: str = None,
    n_jobs: int = 3,
    resume: bool = True,
    verbose: bool = False,
    share_ssm: bool = True,
) -> dict:
    """
    Run robustness check 6.2: Alternative Transition Functions.

    Tests logistic, exponential, and double-logistic transitions.

    OPTIMIZATION: When share_ssm=True (default), reuses SSM models from baseline
    instead of retraining. This is valid because the latent state inference
    (VRNN architecture, latent_dim, etc.) doesn't change - only the STR-HAR
    transition function changes.

    Args:
        config: Configuration dictionary
        exp_dir: Robustness output directory
        baseline_exp_dir: Baseline experiment directory with SSM checkpoints.
                          If None, looks for H{horizon}/ in exp_dir parent.
        n_jobs: Number of parallel jobs
        resume: Enable checkpoint resumability
        verbose: Print progress
        share_ssm: If True, reuse SSM from baseline (faster). If False, retrain SSM.

    Returns:
        Dictionary of results per transition function
    """
    rob_config = config["robustness"]["transition_fns"]
    if not rob_config.get("enabled", True):
        print("  Transition functions check disabled in config")
        return {}

    transitions = rob_config["values"]
    H = rob_config.get("horizon", 5)

    print(f"\n{'='*60}")
    print(f"ROBUSTNESS CHECK 6.2: Alternative Transition Functions")
    print(f"{'='*60}")
    print(f"Testing: {[t['name'] for t in transitions]} at H={H}")

    # Determine baseline directory for SSM sharing
    if share_ssm:
        if baseline_exp_dir is None:
            # Try to find baseline in same output directory
            baseline_exp_dir = exp_dir

        # Check if baseline checkpoints exist
        test_ticker = config["asset_basket"][0]
        q_ssm_test, _ = load_baseline_q_ssm(baseline_exp_dir, test_ticker, H)

        if q_ssm_test is None:
            print(f"\n  [INFO] No baseline SSM checkpoints found in {baseline_exp_dir}")
            print(f"         Falling back to full SSM retraining (share_ssm=False)")
            share_ssm = False
        else:
            print(f"\n  [OPTIMIZATION] Reusing SSM models from baseline")
            print(f"                 Baseline dir: {baseline_exp_dir}")

    results = {}

    for trans in transitions:
        trans_name = trans["name"]
        variant_name = f"transition_{trans_name}"

        # Check if already complete
        if resume and is_variant_complete(exp_dir, "transition_fns", variant_name):
            checkpoint = load_robustness_checkpoint(
                get_robustness_checkpoint_path(exp_dir, "transition_fns", variant_name),
                verbose=verbose
            )
            results[trans_name] = checkpoint
            print(f"\n  {trans_name}: RESUMED from checkpoint")
            continue

        # Per-asset checkpoint dir for this transition variant
        variant_ckpt_dir = os.path.join(exp_dir, "robustness", "transition_fns", trans_name)
        os.makedirs(variant_ckpt_dir, exist_ok=True)

        # Load previously completed assets
        completed_results, remaining_assets = _load_completed_assets(
            variant_ckpt_dir, config["asset_basket"], H
        )
        if completed_results:
            print(f"\n  Running {trans_name}... [{len(completed_results)}/{len(config['asset_basket'])} resumed]")
        else:
            print(f"\n  Running {trans_name}...")

        if not remaining_assets:
            asset_results = completed_results
        elif share_ssm:
            # FAST PATH: Reuse SSM from baseline, only run STR-HAR with new transition
            new_results = []
            for ticker in remaining_assets:
                try:
                    r = run_str_har_with_transition(
                        ticker=ticker,
                        H=H,
                        transition_fn=trans_name,
                        config=config,
                        baseline_exp_dir=baseline_exp_dir,
                        verbose=False,
                    )
                    if r is not None:
                        new_results.append(r)
                        _save_asset_checkpoint(variant_ckpt_dir, ticker, H, r)
                except Exception as e:
                    if verbose:
                        print(f"    ERROR on {ticker}: {e}")
            asset_results = completed_results + new_results
        else:
            # SLOW PATH: Full SSM retraining per asset (pipelined with per-asset checkpoints)
            variant_config = config.copy()
            variant_config["str_har"] = config["str_har"].copy()
            variant_config["str_har"]["transition_fn"] = trans_name

            config_path = os.path.join(PROJECT_ROOT, "configs", "robustness.yaml")
            new_results = _pipelined_run_variant(
                remaining_assets, H, variant_config, variant_ckpt_dir,
                config_path, n_jobs=n_jobs, verbose=verbose,
            )
            asset_results = completed_results + new_results

        asset_results = [r for r in asset_results if r is not None]

        if len(asset_results) > 0:
            # Handle both shared (simple dict) and full (nested dict) result formats
            if share_ssm:
                qlike_values = [r["qlike_ssm"] for r in asset_results if "qlike_ssm" in r]
            else:
                qlike_values = [r["qlike"]["str_ssm"] for r in asset_results if "qlike" in r]

            variant_result = {
                "transition_fn": trans_name,
                "n_params": trans["params"],
                "n_assets": len(asset_results),
                "mean_qlike": np.median(qlike_values) if qlike_values else np.nan,
                "asset_results": asset_results,
                "ssm_shared": share_ssm,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        else:
            variant_result = {"transition_fn": trans_name, "error": "No successful runs"}

        results[trans_name] = variant_result

        if resume:
            save_robustness_checkpoint(
                get_robustness_checkpoint_path(exp_dir, "transition_fns", variant_name),
                variant_result,
                verbose=verbose
            )

        print(f"    QLIKE: {variant_result.get('mean_qlike', np.nan):.4f}")

    _print_transition_fns_table(results, rob_config["baseline"])

    return results


def run_volatility_estimators_check(
    config: dict,
    exp_dir: str,
    n_jobs: int = 3,
    resume: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run robustness check 6.4: Alternative Volatility Estimators.

    Tests Garman-Klass, Parkinson, and Rogers-Satchell estimators.
    """
    rob_config = config["robustness"]["volatility_estimators"]
    if not rob_config.get("enabled", True):
        print("  Volatility estimators check disabled in config")
        return {}

    estimators = rob_config["values"]
    H = rob_config.get("horizon", 5)

    print(f"\n{'='*60}")
    print(f"ROBUSTNESS CHECK 6.4: Alternative Volatility Estimators")
    print(f"{'='*60}")
    print(f"Testing: {estimators} at H={H}")

    results = {}

    for est_name in estimators:
        variant_name = f"volatility_{est_name}"

        # Check if already complete
        if resume and is_variant_complete(exp_dir, "volatility_estimators", variant_name):
            checkpoint = load_robustness_checkpoint(
                get_robustness_checkpoint_path(exp_dir, "volatility_estimators", variant_name),
                verbose=verbose
            )
            results[est_name] = checkpoint
            print(f"\n  {est_name}: RESUMED from checkpoint")
            continue

        print(f"\n  Running {est_name}... (pipelined: sequential VRNN + parallel forecast)")

        # Create variant config
        variant_config = config.copy()
        variant_config["data"] = config["data"].copy()
        variant_config["data"]["volatility_estimator"] = est_name

        # Create variant experiment directory
        variant_exp_dir = os.path.join(exp_dir, "robustness", "volatility_estimators", est_name)
        os.makedirs(variant_exp_dir, exist_ok=True)

        # Pipelined: sequential VRNN training + parallel subprocess forecasting
        config_path = os.path.join(PROJECT_ROOT, "configs", "robustness.yaml")
        asset_results = _pipelined_run_variant(
            config["asset_basket"], H, variant_config, variant_exp_dir,
            config_path, n_jobs=n_jobs, verbose=verbose,
        )

        if len(asset_results) > 0:
            qlike_ssm = [r["qlike"]["str_ssm"] for r in asset_results if "qlike" in r]
            qlike_obs = [r["qlike"]["str_obs"] for r in asset_results if "qlike" in r]

            improvement = 0
            if qlike_obs and qlike_ssm:
                improvement = (np.mean(qlike_obs) - np.mean(qlike_ssm)) / np.mean(qlike_obs) * 100

            variant_result = {
                "estimator": est_name,
                "n_assets": len(asset_results),
                "mean_qlike_ssm": np.median(qlike_ssm) if qlike_ssm else np.nan,
                "mean_qlike_obs": np.median(qlike_obs) if qlike_obs else np.nan,
                "improvement_pct": improvement,
                "asset_results": asset_results,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        else:
            variant_result = {"estimator": est_name, "error": "No successful runs"}

        results[est_name] = variant_result

        if resume:
            save_robustness_checkpoint(
                get_robustness_checkpoint_path(exp_dir, "volatility_estimators", variant_name),
                variant_result,
                verbose=verbose
            )

        print(f"    Deep-LSTR QLIKE: {variant_result.get('mean_qlike_ssm', np.nan):.4f}, "
              f"Improvement: {variant_result.get('improvement_pct', 0):.1f}%")

    _print_volatility_estimators_table(results, rob_config["baseline"])

    return results


def run_subsample_stability_check(
    config: dict,
    exp_dir: str,
    resume: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Run robustness check 6.3: Subsample Stability.

    Analyzes performance across rolling 2-year windows.
    Note: This is a post-processing step on baseline results.
    """
    rob_config = config["robustness"]["subsample"]
    if not rob_config.get("enabled", True):
        print("  Subsample stability check disabled in config")
        return {}

    periods = rob_config["periods"]
    H = rob_config.get("horizon", 5)

    print(f"\n{'='*60}")
    print(f"ROBUSTNESS CHECK 6.3: Subsample Stability")
    print(f"{'='*60}")
    print(f"Analyzing {len(periods)} periods at H={H}")

    # Check for existing checkpoint
    checkpoint_path = get_robustness_checkpoint_path(exp_dir, "subsample", "all_periods")
    if resume and os.path.exists(checkpoint_path):
        results = load_robustness_checkpoint(checkpoint_path, verbose=verbose)
        print("  RESUMED from checkpoint")
        _print_subsample_table(results)
        return results

    # Load baseline results
    baseline_dir = os.path.join(exp_dir, f"H{H}")
    if not os.path.exists(baseline_dir):
        print(f"  ERROR: Baseline results not found at {baseline_dir}")
        print("  Please run the baseline experiment first.")
        return {}

    results = {}

    for period in periods:
        period_name = period["name"]
        start_date = pd.Timestamp(period["start"])
        end_date = pd.Timestamp(period["end"])

        print(f"\n  Analyzing {period_name}...")

        period_results = []

        for ticker in config["asset_basket"]:
            safe_ticker = ticker.replace("=", "").replace("^", "")
            results_path = os.path.join(baseline_dir, f"{safe_ticker}_H{H}_results.csv")

            if not os.path.exists(results_path):
                continue

            try:
                res_df = pd.read_csv(results_path, index_col=0, parse_dates=True)

                # Filter to period
                mask = (res_df.index >= start_date) & (res_df.index <= end_date)
                period_df = res_df.loc[mask]

                if len(period_df) < 50:
                    continue

                # Compute QLIKE for period
                from src.metrics import qlike

                qlike_har = qlike(period_df["y"], period_df["har"]).mean()
                qlike_obs = qlike(period_df["y"], period_df["str_obs"]).mean()
                qlike_ssm = qlike(period_df["y"], period_df["str_ssm"]).mean()

                # Improvements vs HAR
                imp_ssm = (qlike_har - qlike_ssm) / qlike_har * 100
                imp_obs = (qlike_har - qlike_obs) / qlike_har * 100

                period_results.append({
                    "ticker": ticker,
                    "n_obs": len(period_df),
                    "qlike_har": qlike_har,
                    "qlike_obs": qlike_obs,
                    "qlike_ssm": qlike_ssm,
                    "improvement_ssm": imp_ssm,
                    "improvement_obs": imp_obs,
                })
            except Exception as e:
                if verbose:
                    print(f"    Error processing {ticker}: {e}")

        if period_results:
            mean_imp_ssm = np.mean([r["improvement_ssm"] for r in period_results])
            mean_imp_obs = np.mean([r["improvement_obs"] for r in period_results])

            results[period_name] = {
                "period": period_name,
                "start": period["start"],
                "end": period["end"],
                "n_assets": len(period_results),
                "mean_improvement_ssm": mean_imp_ssm,
                "mean_improvement_obs": mean_imp_obs,
                "asset_results": period_results,
            }

            print(f"    Deep-LSTR improvement: {mean_imp_ssm:.1f}%, "
                  f"STR-OBS improvement: {mean_imp_obs:.1f}%")

    # Save checkpoint
    if resume and results:
        save_robustness_checkpoint(checkpoint_path, results, verbose=verbose)

    _print_subsample_table(results)

    return results


def _print_latent_dims_table(results: dict, baseline: int):
    """Print Table 7: Sensitivity to Latent Dimension."""
    print(f"\n{'='*60}")
    print("Table 7: Sensitivity to Latent Dimension")
    print(f"{'='*60}")
    print(f"{'Latent Dim (d)':<18} {'QLIKE (H=5)':<15} {'MCS Inclusion':<15}")
    print("-" * 48)

    for d, res in sorted(results.items()):
        if "error" in res:
            continue
        qlike = res.get("mean_qlike", np.nan)
        mcs = res.get("mcs_inclusion_rate", 0)
        marker = " (baseline)" if d == baseline else ""
        bold = "**" if d == baseline else ""
        print(f"{d}{marker:<14} {bold}{qlike:.4f}{bold:<10} {mcs:.2%}")


def _print_transition_fns_table(results: dict, baseline: str):
    """Print Table 8: Alternative Transition Functions."""
    print(f"\n{'='*60}")
    print("Table 8: Alternative Transition Functions")
    print(f"{'='*60}")
    print(f"{'Transition Function':<22} {'QLIKE (H=5)':<15} {'Param Count':<12}")
    print("-" * 49)

    for name, res in results.items():
        if "error" in res:
            continue
        qlike = res.get("mean_qlike", np.nan)
        n_params = res.get("n_params", 0)
        marker = " (baseline)" if name == baseline else ""
        print(f"{name}{marker:<14} {qlike:.4f}         {n_params}")


def _print_volatility_estimators_table(results: dict, baseline: str):
    """Print Table 10: Alternative Volatility Estimators."""
    print(f"\n{'='*60}")
    print("Table 10: Alternative Volatility Estimators")
    print(f"{'='*60}")
    print(f"{'Estimator':<20} {'Deep-LSTR QLIKE':<18} {'STR-OBS QLIKE':<15} {'Improvement':<12}")
    print("-" * 65)

    for name, res in results.items():
        if "error" in res:
            continue
        qlike_ssm = res.get("mean_qlike_ssm", np.nan)
        qlike_obs = res.get("mean_qlike_obs", np.nan)
        imp = res.get("improvement_pct", 0)
        marker = " (baseline)" if name == baseline else ""
        print(f"{name}{marker:<8} {qlike_ssm:.4f}            {qlike_obs:.4f}         {imp:.1f}%")


def _print_subsample_table(results: dict):
    """Print Table 9: Subsample Performance."""
    print(f"\n{'='*60}")
    print("Table 9: Subsample Performance (QLIKE Improvement vs. HAR)")
    print(f"{'='*60}")
    print(f"{'Period':<15} {'Deep-LSTR':<12} {'STR-OBS':<12}")
    print("-" * 39)

    for period_name, res in results.items():
        if "error" in res:
            continue
        imp_ssm = res.get("mean_improvement_ssm", 0)
        imp_obs = res.get("mean_improvement_obs", 0)
        print(f"{period_name:<15} {imp_ssm:.1f}%        {imp_obs:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run Deep-LSTR robustness checks"
    )
    parser.add_argument(
        "--check",
        type=str,
        choices=["latent_dims", "transition_fns", "volatility_estimators", "subsample", "all"],
        default="all",
        help="Which robustness check to run (default: all)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (default: configs/robustness.yaml)",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Experiment directory",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=10,
        help="Number of parallel forecast workers (default: 10)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resumability",
    )
    parser.add_argument(
        "--no-share-ssm",
        action="store_true",
        help="Disable SSM model sharing for transition function check (slower, retrains SSM)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Baseline experiment directory for SSM sharing (default: same as exp-dir)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config_path = os.path.join(PROJECT_ROOT, "configs", "robustness.yaml")
        if os.path.exists(config_path):
            config = load_config(config_path)
        else:
            # Fall back to default config with robustness defaults
            config = load_config()
            config["robustness"] = {
                "latent_dims": {"enabled": True, "values": [1, 2, 4, 8], "baseline": 2, "horizon": 5},
                "transition_fns": {"enabled": True, "values": [
                    {"name": "logistic", "params": 9},
                    {"name": "exponential", "params": 9},
                    {"name": "double_logistic", "params": 10},
                ], "baseline": "logistic", "horizon": 5},
                "volatility_estimators": {"enabled": True, "values": ["garman_klass", "parkinson", "rogers_satchell"], "baseline": "garman_klass", "horizon": 5},
                "subsample": {"enabled": True, "periods": [
                    {"name": "2017-2018", "start": "2017-01-01", "end": "2018-12-31"},
                    {"name": "2019-2020", "start": "2019-01-01", "end": "2020-12-31"},
                    {"name": "2021-2022", "start": "2021-01-01", "end": "2022-12-31"},
                    {"name": "2023-2024", "start": "2023-01-01", "end": "2024-12-31"},
                ], "horizon": 5},
            }

    # Set random seed
    set_seeds(config.get("seed", 123))

    # Get experiment directory
    if args.exp_dir:
        exp_dir = args.exp_dir
    else:
        exp_dir = get_experiment_dir(config)

    resume = not args.no_resume

    print(f"{'='*60}")
    print("DEEP-LSTR ROBUSTNESS CHECKS")
    print(f"{'='*60}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Resume enabled: {resume}")
    print(f"Check(s) to run: {args.check}")

    all_results = {}

    # Run selected checks
    if args.check in ["latent_dims", "all"]:
        all_results["latent_dims"] = run_latent_dims_check(
            config, exp_dir, args.n_jobs, resume, args.verbose
        )

    if args.check in ["transition_fns", "all"]:
        baseline_dir = args.baseline_dir if args.baseline_dir else exp_dir
        all_results["transition_fns"] = run_transition_fns_check(
            config=config,
            exp_dir=exp_dir,
            baseline_exp_dir=baseline_dir,
            n_jobs=args.n_jobs,
            resume=resume,
            verbose=args.verbose,
            share_ssm=not args.no_share_ssm,
        )

    if args.check in ["volatility_estimators", "all"]:
        all_results["volatility_estimators"] = run_volatility_estimators_check(
            config, exp_dir, args.n_jobs, resume, args.verbose
        )

    if args.check in ["subsample", "all"]:
        all_results["subsample"] = run_subsample_stability_check(
            config, exp_dir, resume, args.verbose
        )

    print(f"\n{'='*60}")
    print("ROBUSTNESS CHECKS COMPLETE")
    print(f"{'='*60}")
    print(f"Results saved to: {exp_dir}/robustness/")

    return all_results


if __name__ == "__main__":
    main()
