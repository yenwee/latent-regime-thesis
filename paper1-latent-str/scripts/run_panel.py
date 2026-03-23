#!/usr/bin/env python3
"""
Run Deep-LSTR volatility forecasting across full asset panel.

Two-phase execution to handle PyTorch memory on Apple Silicon:
  Phase 1: Train VRNNs sequentially (imports torch, ~1.5GB per process)
  Phase 2: Run forecasting in parallel (no torch, ~300MB per worker)

Example usage:
    python scripts/run_panel.py
    python scripts/run_panel.py --horizons 1 5 22 --n-jobs 10
"""

import argparse
import os
import sys
import datetime
import time
from itertools import product

# Prevent thread oversubscription when using joblib parallelism.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

from src.utils import (
    set_seeds, load_config, ensure_writable_dir, get_experiment_dir,
    make_retrain_schedule, check_segment_checkpoint, save_segment_checkpoint,
    get_completed_segments,
)
from src.data import download_asset_data, prepare_features


def pretrain_vrnn_for_asset(ticker, H, config, exp_dir, verbose=False):
    """
    Phase 1: Train VRNN segments for a single (asset, horizon).

    Imports torch, trains all missing segments, saves checkpoints, then returns.
    This should run sequentially (1 process) to avoid memory issues.
    """
    from src.vrnn import train_deep_ssm
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    seed = config["seed"]
    set_seeds(seed)
    device = torch.device("cpu")

    exp_config = config.get("experiment", {})
    resume_enabled = exp_config.get("resume", True)

    # Download & prepare data
    df = download_asset_data(
        ticker,
        start=config["data"]["start"],
        end=config["data"]["end"],
        interval=config["data"]["interval"],
    )
    volatility_estimator = config.get("data", {}).get("volatility_estimator", "garman_klass")
    df = prepare_features(df, H, q_obs_smooth_span=config["smoothing"]["q_obs_span"],
                          volatility_estimator=volatility_estimator)

    roll_window = config["rolling"]["window"]
    min_start = config["rolling"]["min_start"]
    retrain_freq = config["rolling"].get("retrain_freq", "A")
    start_base = max(min_start, roll_window)
    oos_start_date = df.index[start_base]

    retrain_dates = make_retrain_schedule(df.index, oos_start_date, retrain_freq)
    segment_bounds = []
    for j, d0 in enumerate(retrain_dates):
        d1 = retrain_dates[j + 1] if j + 1 < len(retrain_dates) else df.index[-1]
        segment_bounds.append((d0, d1))

    X_cols = ["x1_logv", "x2_absr", "x3_logvol"]
    ssm_config = config["ssm"]

    completed_segments = get_completed_segments(exp_dir, ticker, H) if resume_enabled else set()
    trained = 0
    skipped = 0

    for seg_id, (seg_start, seg_end) in enumerate(segment_bounds, start=1):
        seg_start_pos = df.index.get_loc(seg_start)
        seg_end_pos = df.index.get_loc(seg_end)
        if seg_end_pos <= seg_start_pos:
            continue

        split = seg_start_pos
        if split < 200:
            continue

        # Skip if already checkpointed
        if resume_enabled and seg_id in completed_segments:
            skipped += 1
            continue

        # Train VRNN
        muX = df.iloc[:split][X_cols].mean()
        sdX = df.iloc[:split][X_cols].std().replace(0, 1.0)
        X_train_np = ((df.iloc[:split][X_cols] - muX) / sdX).values
        X_infer_np = ((df.iloc[:seg_end_pos][X_cols] - muX) / sdX).values

        X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
        X_infer = torch.tensor(X_infer_np, dtype=torch.float32, device=device)

        _, Z_infer, elboT = train_deep_ssm(
            X_train, X_infer,
            latent_dim=ssm_config["latent_dim"],
            gru_hidden=ssm_config["gru_hidden"],
            dec_hidden=ssm_config["decoder_hidden"],
            lr=ssm_config["lr"],
            weight_decay=ssm_config["weight_decay"],
            epochs=ssm_config["epochs"],
            patience=ssm_config["patience"],
            device=device,
            verbose=False,
        )

        # Project latent states to scalar q_ssm
        Z_train = Z_infer[:split]
        Z_mean = Z_train.mean(axis=0, keepdims=True)
        Z_std = Z_train.std(axis=0, keepdims=True)
        Z_std[Z_std < 1e-9] = 1.0
        Zs = (Z_infer - Z_mean) / Z_std

        y_train_for_w = df["x1_logv"].values[:split]
        latent_dim = ssm_config["latent_dim"]
        Xw = np.column_stack([np.ones(split)] + [Zs[:split, d] for d in range(latent_dim)])
        beta_w, *_ = np.linalg.lstsq(Xw, y_train_for_w, rcond=None)
        w = beta_w[1:]
        q = Zs @ w
        if np.corrcoef(q[:split], y_train_for_w)[0, 1] < 0:
            q = -q

        q_ssm_smooth_span = config["smoothing"].get("q_ssm_span", 0)
        if q_ssm_smooth_span > 0:
            q_series = pd.Series(q, index=df.index[:seg_end_pos]).ewm(
                span=q_ssm_smooth_span, adjust=False).mean()
        else:
            q_series = pd.Series(q, index=df.index[:seg_end_pos])

        save_segment_checkpoint(
            exp_dir, ticker, H, seg_id,
            q_series, Z_infer, elboT,
            seg_start, seg_end,
            verbose=verbose,
        )
        trained += 1

        if verbose:
            print(f"    {ticker} H={H} seg {seg_id}/{len(segment_bounds)}: ELBO/T={elboT:.4f}")

    return {"ticker": ticker, "H": H, "trained": trained, "skipped": skipped}


def run_panel(
    horizons: list = None,
    n_jobs: int = 10,
    config: dict = None,
    exp_dir: str = None,
    verbose: bool = False,
    config_path: str = None,
):
    """
    Run Deep-LSTR analysis for all assets across multiple horizons.

    Pipelined execution:
      - VRNN training runs sequentially in the main process (needs torch, ~500MB)
      - As soon as all horizons for an asset are trained, forecasting tasks
        are dispatched to a parallel worker pool (no torch, ~200MB each)
      - This overlaps Phase 1 and Phase 2, saving significant wall-clock time
    """
    if config is None:
        config = load_config()

    if horizons is None:
        horizons = config.get("horizons", [1, 5, 22])

    if exp_dir is None:
        exp_dir = get_experiment_dir(config)

    assets = config["asset_basket"]
    n_assets = len(assets)
    total_tasks = n_assets * len(horizons)

    print(f"{'='*60}")
    print("DEEP-LSTR PANEL ANALYSIS (PIPELINED)")
    print(f"{'='*60}")
    print(f"Experiment directory: {exp_dir}")
    print(f"Horizons: {horizons}")
    print(f"Assets: {n_assets}")
    print(f"Total tasks: {total_tasks}")
    print(f"Forecast workers: {n_jobs}")

    # Create output directories
    for H in horizons:
        os.makedirs(os.path.join(exp_dir, f"H{H}"), exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor
    import subprocess

    # Resolve config path for subprocesses
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "default.yaml")

    def forecast_subprocess(ticker, H):
        """Run forecasting as isolated subprocess (no torch VSIZE in worker pool)."""
        cmd = [
            sys.executable, "-u", "-c",
            f"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'
import sys
sys.path.insert(0, '{PROJECT_ROOT}')
from scripts.run_single_asset import run_single_asset
from src.utils import load_config
config = load_config('{config_path}')
try:
    result = run_single_asset('{ticker}', H={H}, config=config, exp_dir='{exp_dir}', verbose=False)
    qlike = result.get('qlike', {{}})
    print(f'OK {ticker} H={H} qlike_ssm={{qlike.get("str_ssm", "N/A")}}')
except Exception as e:
    print(f'ERROR {ticker} H={H} {{e}}')
"""
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr[-300:] if result.stderr else "unknown"
            return {"ticker": ticker, "H": H, "error": err}
        return {"ticker": ticker, "H": H, "output": output}

    # ============================================================
    # PIPELINED EXECUTION
    # ============================================================
    # Main thread: train VRNNs sequentially per asset
    # Worker pool: forecast tasks dispatched as assets become ready
    print(f"\n{'='*60}")
    print(f"PIPELINED: VRNN sequential + {n_jobs} forecast workers")
    print(f"{'='*60}")

    t0_total = time.perf_counter()
    total_trained = 0
    total_skipped = 0
    forecast_futures = []
    completed_forecasts = []
    errors = []

    executor = ThreadPoolExecutor(max_workers=n_jobs)

    for asset_idx, ticker in enumerate(assets, 1):
        # Phase 1: Train all horizons for this asset
        for H in horizons:
            task_num = (asset_idx - 1) * len(horizons) + horizons.index(H) + 1
            try:
                result = pretrain_vrnn_for_asset(ticker, H, config, exp_dir, verbose=verbose)
            except Exception as e:
                print(f"  VRNN ERROR {ticker} H={H}: {e}")
                result = {"ticker": ticker, "H": H, "trained": 0, "skipped": 0}
            total_trained += result["trained"]
            total_skipped += result["skipped"]
            if result["trained"] > 0:
                print(f"  [VRNN {task_num}/{total_tasks}] {ticker} H={H}: trained {result['trained']} segments")

        # Phase 2: Dispatch all horizons for this asset to worker pool
        for H in horizons:
            future = executor.submit(forecast_subprocess, ticker, H)
            forecast_futures.append((ticker, H, future))
        print(f"  [DISPATCH {asset_idx}/{n_assets}] {ticker}: queued {len(horizons)} forecast tasks")

        # Collect any completed forecasts (non-blocking)
        newly_done = []
        for t, h, f in forecast_futures:
            if f.done() and (t, h) not in [(c["ticker"], c["H"]) for c in completed_forecasts]:
                try:
                    r = f.result(timeout=0)
                    completed_forecasts.append(r)
                    if "error" in r:
                        errors.append(r)
                        print(f"  [FORECAST] {t} H={h}: ERROR")
                    else:
                        print(f"  [FORECAST] {t} H={h}: {r.get('output', 'done')}")
                except Exception as e:
                    errors.append({"ticker": t, "H": h, "error": str(e)})

    dt_vrnn = time.perf_counter() - t0_total
    print(f"\nAll VRNN training done ({dt_vrnn:.0f}s). Waiting for remaining forecasts...")

    # Wait for remaining forecast tasks
    for ticker, H, future in forecast_futures:
        if (ticker, H) in [(c["ticker"], c["H"]) for c in completed_forecasts]:
            continue
        try:
            r = future.result(timeout=1200)
            completed_forecasts.append(r)
            if "error" in r:
                errors.append(r)
                print(f"  [FORECAST] {ticker} H={H}: ERROR")
            else:
                print(f"  [FORECAST] {ticker} H={H}: {r.get('output', 'done')}")
        except Exception as e:
            errors.append({"ticker": ticker, "H": H, "error": str(e)})
            print(f"  [FORECAST] {ticker} H={H}: TIMEOUT/ERROR")

    executor.shutdown(wait=False)
    dt_total = time.perf_counter() - t0_total

    # Summary
    if errors:
        print(f"\n{'='*60}")
        print(f"ERRORS: {len(errors)} tasks failed")
        print(f"{'='*60}")
        for e in errors:
            print(f"  {e['ticker']} H={e['H']}: {e.get('error', 'unknown')}")

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"VRNN training:  {total_trained} trained, {total_skipped} cached")
    print(f"Forecasts:      {len(completed_forecasts)} completed, {len(errors)} errors")
    print(f"Total time:     {dt_total:.0f}s ({dt_total/60:.1f} min)")
    print(f"Results saved to: {exp_dir}")

    return completed_forecasts


def main():
    parser = argparse.ArgumentParser(
        description="Run Deep-LSTR volatility forecasting across full asset panel"
    )
    parser.add_argument("--horizons", type=int, nargs="+", default=None,
                        help="Forecast horizons (default: 1 5 22)")
    parser.add_argument("--n-jobs", type=int, default=10,
                        help="Number of parallel jobs for Phase 2 (default: 10)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    parser.add_argument("--exp-dir", type=str, default=None,
                        help="Experiment directory")
    parser.add_argument("--exp-id", type=str, default=None,
                        help="Experiment ID (overrides config)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable checkpoint resumability")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed progress")

    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    if args.exp_id:
        config.setdefault("experiment", {})["id"] = args.exp_id
    if args.no_resume:
        config.setdefault("experiment", {})["resume"] = False

    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = get_experiment_dir(config)

    run_panel(
        horizons=args.horizons,
        n_jobs=args.n_jobs,
        config=config,
        exp_dir=exp_dir,
        verbose=args.verbose,
        config_path=args.config or os.path.join(PROJECT_ROOT, "configs", "default.yaml"),
    )


if __name__ == "__main__":
    main()
