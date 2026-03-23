#!/usr/bin/env python3
"""
Run Deep-LSTR volatility forecasting for a single asset.

This script demonstrates the full pipeline:
1. Download OHLC data from Yahoo Finance
2. Prepare HAR features and targets
3. Train Deep SSM for latent state inference (with annual retraining)
4. Fit STR-HAR models with observable and latent transitions
5. Evaluate forecasts with QLIKE, Diebold-Mariano tests, and MCS

Example usage:
    python scripts/run_single_asset.py --ticker ^GSPC --horizon 5
    python scripts/run_single_asset.py --ticker CL=F --horizon 1 --verbose
"""

import argparse
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

# Lazy-import torch and VRNN: only loaded when VRNN training is actually needed.
# This avoids PyTorch's ~400GB VSIZE mmap (MPS/Metal) in parallel workers
# that only need numpy/scipy for the forecasting loop.
torch = None
train_deep_ssm = None

def _ensure_torch():
    """Import torch and VRNN on first use, pin threads for parallel safety."""
    global torch, train_deep_ssm
    if torch is not None:
        return
    import torch as _torch
    torch = _torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from src.vrnn import train_deep_ssm as _train
    train_deep_ssm = _train

from src.utils import (
    set_seeds,
    load_config,
    ensure_writable_dir,
    make_retrain_schedule,
    get_experiment_dir,
    is_run_complete,
    check_segment_checkpoint,
    save_segment_checkpoint,
    load_forecast_checkpoint,
    save_forecast_checkpoint,
    cleanup_checkpoints,
    get_completed_segments,
)
from src.data import download_asset_data, prepare_features, standardize_features
from src.str_har import fit_har_ols, har_predict, fit_str2_window_robust, str2_forecast_one, str2_in_sample_yhat
from src.metrics import qlike, mse_logv
from src.dm_test import dm_test
from src.mcs import bootstrap_mcs
from src.garch import Garch11_t, Egarch11_t, MSGarch2_t
from src.risk import (
    fit_nu_mle_var1,
    risk_table_fz_es_dynamic,
    fz_loss_series_dynamic,
)
from src.smoothers import get_transition_smoother, TRANSITION_SMOOTHERS
from src.har import fit_lhar_ols, lhar_predict, fit_har_j_ols, har_j_predict

# Numerical stability constant
EPS = 1e-12


def is_ablation_mode(cfg: dict) -> bool:
    """Check if config enables ablation experiments."""
    return "ablation" in cfg and cfg["ablation"] is not None


def get_ablation_smoothers(cfg: dict) -> list:
    """Get list of transition smoothers for ablation."""
    if not is_ablation_mode(cfg):
        return []
    return cfg["ablation"].get("smoothers", [])


def get_ablation_har_variants(cfg: dict) -> list:
    """Get list of HAR variants for ablation."""
    if not is_ablation_mode(cfg):
        return []
    return cfg["ablation"].get("har_variants", [])


def run_single_asset(
    ticker: str,
    H: int = 5,
    config: dict = None,
    exp_dir: str = None,
    verbose: bool = False,
):
    """
    Run complete Deep-LSTR analysis for a single asset.

    Uses segment-based SSM retraining with annual schedule (configurable via retrain_freq).

    Supports partial resumability:
    - If run is already complete, returns early
    - Reloads completed segment SSM checkpoints
    - Resumes forecasting from last checkpoint

    Args:
        ticker: Asset ticker symbol
        H: Forecast horizon in days
        config: Configuration dictionary (loads default if None)
        exp_dir: Experiment directory (auto-generated if None)
        verbose: Print progress information

    Returns:
        Dictionary with results including MCS analysis
    """
    if config is None:
        config = load_config()

    # Set seeds for reproducibility
    seed = config["seed"]
    set_seeds(seed)

    # Get experiment directory (uses config experiment.id or auto-generates)
    if exp_dir is None:
        exp_dir = get_experiment_dir(config)

    # Create horizon-specific output directory
    save_dir = os.path.join(exp_dir, f"H{H}")
    os.makedirs(save_dir, exist_ok=True)

    # Check resumability settings
    exp_config = config.get("experiment", {})
    resume_enabled = exp_config.get("resume", True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running Deep-LSTR for {ticker} (H={H})")
        print(f"{'='*60}")
        print(f"Experiment directory: {exp_dir}")
        print(f"Resume enabled: {resume_enabled}")

    # Check if run is already complete
    if resume_enabled and is_run_complete(exp_dir, ticker, H):
        if verbose:
            print(f"\n[SKIP] Run already complete for {ticker} H={H}")
        # Load and return existing results
        safe_name = ticker.replace("=", "").replace("^", "")
        results_path = os.path.join(save_dir, f"{safe_name}_H{H}_results.csv")
        res_df = pd.read_csv(results_path, index_col=0, parse_dates=True)

        # Load MCS results
        mcs_path = os.path.join(save_dir, f"{safe_name}_H{H}_mcs.csv")
        mcs_df = pd.read_csv(mcs_path)
        mcs_models = mcs_df[mcs_df["in_mcs"]]["model"].tolist()
        mcs_pvals = dict(zip(mcs_df["model"], mcs_df["mcs_pval"]))

        return {
            "ticker": ticker,
            "H": H,
            "results": res_df,
            "mcs": {"included": mcs_models, "pvals": mcs_pvals},
            "resumed": True,
        }

    # Download data
    if verbose:
        print(f"\n[1/6] Downloading data from Yahoo Finance...")
    df = download_asset_data(
        ticker,
        start=config["data"]["start"],
        end=config["data"]["end"],
        interval=config["data"]["interval"],
    )

    if len(df) < config["rolling"]["window"] + config["rolling"]["min_start"] + H + 50:
        raise ValueError(f"Insufficient data for {ticker}: {len(df)} rows")

    if verbose:
        print(f"      Downloaded {len(df)} days of data")

    # Prepare features
    if verbose:
        print(f"\n[2/6] Preparing features...")
    volatility_estimator = config.get("data", {}).get("volatility_estimator", "garman_klass")
    df = prepare_features(
        df, H,
        q_obs_smooth_span=config["smoothing"]["q_obs_span"],
        volatility_estimator=volatility_estimator,
    )
    if verbose:
        print(f"      Feature matrix: {df.shape}")

    # Define rolling window parameters
    roll_window = config["rolling"]["window"]
    min_start = config["rolling"]["min_start"]
    retrain_freq = config["rolling"].get("retrain_freq", "A")

    # Find OOS start date
    start_base = max(min_start, roll_window)
    oos_start_idx = start_base
    oos_start_date = df.index[oos_start_idx]

    if verbose:
        print(f"      OOS start: {oos_start_date}")
        print(f"      Retrain frequency: {retrain_freq}")

    # Create retrain schedule for segment-based SSM training
    retrain_dates = make_retrain_schedule(df.index, oos_start_date, retrain_freq)
    segment_bounds = []
    for j, d0 in enumerate(retrain_dates):
        d1 = retrain_dates[j + 1] if j + 1 < len(retrain_dates) else df.index[-1]
        segment_bounds.append((d0, d1))

    if verbose:
        print(f"      Number of segments: {len(segment_bounds)}")

    # Prepare SSM input features
    X_cols = ["x1_logv", "x2_absr", "x3_logvol"]
    ssm_config = config["ssm"]
    str_config = config["str_har"]

    # Holds q_ssm up to each segment end
    q_ssm_full = pd.Series(index=df.index, dtype=float)

    # OOS containers - try to load from checkpoint
    results, last_forecast_idx = [], -1
    if resume_enabled:
        results, last_forecast_idx = load_forecast_checkpoint(exp_dir, ticker, H, verbose=verbose)
        if results:
            if verbose:
                print(f"      Resumed {len(results)} forecasts from checkpoint")

    elbo_step_dates = []
    elbo_step_values = []

    # Check which segments are already completed
    completed_segments = get_completed_segments(exp_dir, ticker, H) if resume_enabled else set()

    if verbose:
        print(f"\n[3/6] Running segment-based SSM training and forecasting...")
        if completed_segments:
            print(f"      Resuming: {len(completed_segments)} segments already complete")

    # Segment-based SSM retraining loop
    for seg_id, (seg_start, seg_end) in enumerate(segment_bounds, start=1):
        seg_start_pos = df.index.get_loc(seg_start)
        seg_end_pos = df.index.get_loc(seg_end)
        if seg_end_pos <= seg_start_pos:
            continue

        # Expanding train size for SSM
        split = seg_start_pos
        if split < 200:
            continue

        # Check for segment checkpoint (SSM already trained)
        seg_checkpoint = None
        if resume_enabled and seg_id in completed_segments:
            seg_checkpoint = check_segment_checkpoint(exp_dir, ticker, H, seg_id)

        if seg_checkpoint is not None:
            # Load SSM results from checkpoint
            if verbose:
                print(f"\n  [Segment {seg_id}/{len(segment_bounds)}] "
                      f"LOADED from checkpoint (ELBO/T ~ {seg_checkpoint['elbo']:.4f})")

            q_series = seg_checkpoint["q_ssm_segment"]
            q_ssm_full.loc[q_series.index] = q_series.values
            elbo_step_dates.append(seg_start)
            elbo_step_values.append(seg_checkpoint["elbo"])
        else:
            # Train SSM for this segment (lazy-load torch only when needed)
            _ensure_torch()
            device = torch.device("cpu")

            # Standardize using training data only
            muX = df.iloc[:split][X_cols].mean()
            sdX = df.iloc[:split][X_cols].std().replace(0, 1.0)

            X_train_np = ((df.iloc[:split][X_cols] - muX) / sdX).values
            X_infer_np = ((df.iloc[:seg_end_pos][X_cols] - muX) / sdX).values

            X_train = torch.tensor(X_train_np, dtype=torch.float32, device=device)
            X_infer = torch.tensor(X_infer_np, dtype=torch.float32, device=device)

            if verbose:
                print(f"\n  [Segment {seg_id}/{len(segment_bounds)}] "
                      f"train<= {df.index[split-1].date()} | infer<= {df.index[seg_end_pos-1].date()}")

            # Train SSM
            _, Z_infer, elboT = train_deep_ssm(
                X_train,
                X_infer,
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

            if verbose:
                print(f"    SSM done. (last ELBO/T ~ {elboT:.4f})")

            elbo_step_dates.append(seg_start)
            elbo_step_values.append(elboT)

            # Build scalar q_ssm via train-only OLS weights
            Z_train = Z_infer[:split]
            Z_mean = Z_train.mean(axis=0, keepdims=True)
            Z_std = Z_train.std(axis=0, keepdims=True)
            Z_std[Z_std < 1e-9] = 1.0
            Zs = (Z_infer - Z_mean) / Z_std

            y_train_for_w = df["x1_logv"].values[:split]

            # Use latent dimensions for OLS projection
            latent_dim = ssm_config["latent_dim"]
            Xw = np.column_stack([np.ones(split)] + [Zs[:split, d] for d in range(latent_dim)])
            beta_w, *_ = np.linalg.lstsq(Xw, y_train_for_w, rcond=None)
            w = beta_w[1:]

            q = Zs @ w
            if np.corrcoef(q[:split], y_train_for_w)[0, 1] < 0:
                q = -q

            # Apply smoothing if configured
            q_ssm_smooth_span = config["smoothing"].get("q_ssm_span", 0)
            if q_ssm_smooth_span > 0:
                q_series = pd.Series(q, index=df.index[:seg_end_pos]).ewm(
                    span=q_ssm_smooth_span, adjust=False
                ).mean()
            else:
                q_series = pd.Series(q, index=df.index[:seg_end_pos])

            q_ssm_full.loc[q_series.index] = q_series.values

            # Save segment checkpoint
            if resume_enabled:
                save_segment_checkpoint(
                    exp_dir, ticker, H, seg_id,
                    q_series, Z_infer, elboT,
                    seg_start, seg_end,
                    verbose=verbose,
                )

        # Run rolling forecasts inside segment
        seg_forecast_start = max(seg_start_pos, start_base)
        seg_forecast_end = seg_end_pos

        new_forecasts_in_segment = 0
        for i in range(seg_forecast_start, seg_forecast_end - 1):
            if i - roll_window < 0:
                continue

            # Skip already-computed forecasts (resume support)
            if resume_enabled and i <= last_forecast_idx:
                continue

            train_i = df.iloc[i - roll_window:i].copy()
            test_i = df.iloc[i:i + 1].copy()

            train_i["q_ssm"] = q_ssm_full.loc[train_i.index].values
            test_i["q_ssm"] = q_ssm_full.loc[test_i.index].values

            if train_i["q_ssm"].isna().any() or test_i["q_ssm"].isna().any():
                continue

            # Fit HAR
            b = fit_har_ols(train_i)
            yh_har = float(har_predict(b, test_i)[0])

            # Get transition function setting
            transition_fn = str_config.get("transition_fn", "logistic")

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
            yh_obs, g_obs = str2_forecast_one(p_obs, mu_obs, sd_obs, test_i, q_col="q_obs", transition_fn=transition_fn)

            # Fit STR-HAR with latent transition
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
            yh_ssm, g_ssm = str2_forecast_one(p_ssm, mu_ssm, sd_ssm, test_i, q_col="q_ssm", transition_fn=transition_fn)

            # ============ Ablation: Alternative transition smoothers ============
            ablation_results = {}
            ablation_smoothers = get_ablation_smoothers(config)
            if ablation_smoothers:
                # Base input for smoothers: log volatility from training window
                logv_train = train_i["x1_logv"].values
                logv_test_val = test_i["x1_logv"].values[0]

                for smoother_name in ablation_smoothers:
                    try:
                        smoother_fn = get_transition_smoother(smoother_name)
                        # Apply smoother to training data
                        q_smooth_train = smoother_fn(logv_train)
                        # For test point, extend series and get last value
                        logv_extended = np.append(logv_train, logv_test_val)
                        q_smooth_extended = smoother_fn(logv_extended)
                        q_smooth_test = q_smooth_extended[-1]

                        # Add smoothed transition to train/test DataFrames
                        train_i[f"q_{smoother_name}"] = q_smooth_train
                        test_i[f"q_{smoother_name}"] = q_smooth_test

                        # Fit STR-HAR with this smoother's transition
                        p_abl, mu_abl, sd_abl = fit_str2_window_robust(
                            train_i,
                            q_col=f"q_{smoother_name}",
                            use_basinhopping=str_config["use_basinhopping"],
                            bh_niter=str_config["basinhopping_niter"],
                            n_starts=str_config["n_starts"],
                            gamma_max=str_config["gamma_max"],
                            gamma_lam=str_config["gamma_lambda"],
                            transition_fn=transition_fn,
                        )
                        yh_abl, g_abl = str2_forecast_one(
                            p_abl, mu_abl, sd_abl, test_i,
                            q_col=f"q_{smoother_name}",
                            transition_fn=transition_fn
                        )
                        ablation_results[f"str_{smoother_name}"] = float(yh_abl)
                        ablation_results[f"G_{smoother_name}"] = float(g_abl)
                    except Exception as e:
                        # Log warning but continue with other smoothers
                        if verbose and new_forecasts_in_segment == 0:
                            print(f"      Warning: smoother {smoother_name} failed: {e}")
                        ablation_results[f"str_{smoother_name}"] = np.nan
                        ablation_results[f"G_{smoother_name}"] = np.nan

            # ============ Ablation: HAR variants (LHAR, HAR-J) ============
            har_variant_results = {}
            har_variants = get_ablation_har_variants(config)
            if har_variants:
                for variant in har_variants:
                    try:
                        if variant == "har":
                            # Already computed above
                            har_variant_results["har_baseline"] = yh_har
                        elif variant == "lhar":
                            # LHAR requires 'r' column for leverage term
                            if "r" in train_i.columns:
                                b_lhar = fit_lhar_ols(train_i)
                                yh_lhar = float(lhar_predict(b_lhar, test_i)[0])
                                har_variant_results["lhar"] = yh_lhar
                            else:
                                har_variant_results["lhar"] = np.nan
                        elif variant == "har_j":
                            # HAR-J requires 'j_d' (jump component) column
                            if "j_d" in train_i.columns:
                                b_harj = fit_har_j_ols(train_i)
                                yh_harj = float(har_j_predict(b_harj, test_i)[0])
                                har_variant_results["har_j"] = yh_harj
                            else:
                                har_variant_results["har_j"] = np.nan
                    except Exception as e:
                        if verbose and new_forecasts_in_segment == 0:
                            print(f"      Warning: HAR variant {variant} failed: {e}")
                        har_variant_results[variant] = np.nan

            # GARCH fits (on daily returns)
            r_hist = train_i["r"].dropna().values
            garch_config = config.get("garch", {})
            garch_bh = garch_config.get("use_basinhopping", True)

            # GARCH(1,1)-t
            g11 = Garch11_t(use_basinhopping=garch_bh).fit(r_hist)
            yh_garch = g11.predict(r_hist, H)

            # EGARCH(1,1)-t with seed for reproducibility
            eg11 = Egarch11_t(use_basinhopping=garch_bh).fit(r_hist)
            yh_egarch = eg11.predict(r_hist, H, seed=(seed + i))

            # MS-GARCH(2)-t
            msg2 = MSGarch2_t(use_basinhopping=garch_bh).fit(r_hist)
            yh_msgarch = msg2.predict(r_hist, H)

            # ============ Risk evaluation data collection ============
            # Get risk config
            risk_config = config.get("risk", {})
            use_rolling_k = risk_config.get("use_rolling_k", True)
            rolling_nu = risk_config.get("rolling_nu", True)
            mean_adjust = risk_config.get("mean_adjust", True)

            # H-day forward return for VaR backtesting
            rH_col = f"r_fwd_sum_{H}"
            rH = float(test_i[rH_col].values[0]) if rH_col in test_i.columns else np.nan

            # Mean adjustment muH from training window
            if mean_adjust and rH_col in df.columns:
                r_train_H_all = df.loc[train_i.index, rH_col].values
                muH = float(np.nanmean(r_train_H_all))
            else:
                muH = 0.0

            # Rolling k_t and nu_t from training window
            r_train_H = df.loc[train_i.index, rH_col].values if rH_col in df.columns else np.array([])
            mask = np.isfinite(r_train_H)
            r_train_H_clean = r_train_H[mask]

            # In-sample predictions for k_t computation
            yhat_train_har = har_predict(b, train_i)
            yhat_train_obs = str2_in_sample_yhat(p_obs, mu_obs, sd_obs, train_i, q_col="q_obs", transition_fn=transition_fn)
            yhat_train_ssm = str2_in_sample_yhat(p_ssm, mu_ssm, sd_ssm, train_i, q_col="q_ssm", transition_fn=transition_fn)

            sigH_train_har = np.sqrt(H * np.exp(yhat_train_har))[mask]
            sigH_train_obs = np.sqrt(H * np.exp(yhat_train_obs))[mask]
            sigH_train_ssm = np.sqrt(H * np.exp(yhat_train_ssm))[mask]

            # Compute rolling k_t (volatility scaling)
            if use_rolling_k and len(r_train_H_clean) >= 20:
                k_har = np.sqrt(np.mean(r_train_H_clean**2) / (np.mean(sigH_train_har**2) + EPS))
                k_obs = np.sqrt(np.mean(r_train_H_clean**2) / (np.mean(sigH_train_obs**2) + EPS))
                k_ssm = np.sqrt(np.mean(r_train_H_clean**2) / (np.mean(sigH_train_ssm**2) + EPS))
            else:
                k_har = k_obs = k_ssm = 1.0

            # Compute rolling nu_t (Student-t degrees of freedom)
            if rolling_nu and len(r_train_H_clean) >= 80:
                eps_har = (r_train_H_clean - muH) / (k_har * sigH_train_har + EPS)
                eps_obs = (r_train_H_clean - muH) / (k_obs * sigH_train_obs + EPS)
                eps_ssm = (r_train_H_clean - muH) / (k_ssm * sigH_train_ssm + EPS)
                nu_har = fit_nu_mle_var1(eps_har, nu0=8.0)
                nu_obs = fit_nu_mle_var1(eps_obs, nu0=8.0)
                nu_ssm = fit_nu_mle_var1(eps_ssm, nu0=8.0)
            else:
                nu_har = nu_obs = nu_ssm = 8.0

            # GARCH models: k=1.0, nu from fitted params
            k_garch = k_egarch = k_msgarch = 1.0
            nu_garch = 2.0 + np.exp(g11.params[3]) if len(g11.params) > 3 else 8.0
            nu_egarch = 2.0 + np.exp(eg11.params[4]) if len(eg11.params) > 4 else 8.0
            nu_msgarch = 2.0 + np.exp(msg2.params[8]) if len(msg2.params) > 8 else 8.0

            # Store results with risk data
            result_row = {
                "date": test_i.index[0],
                "y": float(test_i["y"].values[0]),
                "rH": rH,
                "har": yh_har,
                "str_obs": yh_obs,
                "str_ssm": yh_ssm,
                "garch": yh_garch,
                "egarch": yh_egarch,
                "msgarch": yh_msgarch,
                "G_obs": g_obs,
                "G_ssm": g_ssm,
                "muH": muH,
                "k_har": float(k_har),
                "k_obs": float(k_obs),
                "k_ssm": float(k_ssm),
                "k_garch": float(k_garch),
                "k_egarch": float(k_egarch),
                "k_msgarch": float(k_msgarch),
                "nu_har": float(nu_har),
                "nu_obs": float(nu_obs),
                "nu_ssm": float(nu_ssm),
                "nu_garch": float(nu_garch),
                "nu_egarch": float(nu_egarch),
                "nu_msgarch": float(nu_msgarch),
            }
            # Add ablation smoother results
            result_row.update(ablation_results)
            # Add HAR variant results
            result_row.update(har_variant_results)
            results.append(result_row)
            new_forecasts_in_segment += 1

            # Periodic checkpoint save (every 50 forecasts)
            if resume_enabled and new_forecasts_in_segment % 50 == 0:
                save_forecast_checkpoint(exp_dir, ticker, H, results, i, verbose=False)

        # Save checkpoint at end of segment
        if resume_enabled and new_forecasts_in_segment > 0:
            save_forecast_checkpoint(exp_dir, ticker, H, results, seg_forecast_end - 2, verbose=verbose)

        if verbose:
            seg_count = len([r for r in results if r["date"] >= seg_start])
            print(f"    Forecasts in segment: {seg_count} (new: {new_forecasts_in_segment})")

    if len(results) == 0:
        raise ValueError(f"No forecasts produced for {ticker}")

    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    res_df.set_index("date", inplace=True)

    # Compute metrics
    if verbose:
        print(f"\n[4/6] Computing evaluation metrics...")

    # Model names for evaluation (base models)
    models = ["har", "str_obs", "str_ssm", "garch", "egarch", "msgarch"]

    # Add ablation models to evaluation if present
    ablation_model_cols = [c for c in res_df.columns if c.startswith("str_") and c not in ["str_obs", "str_ssm"]]
    har_variant_cols = [c for c in res_df.columns if c in ["lhar", "har_j", "har_baseline"]]
    all_models = models + ablation_model_cols + har_variant_cols

    # QLIKE losses per model
    qlike_dict = {}
    mse_dict = {}
    for m in all_models:
        if m in res_df.columns and res_df[m].notna().any():
            qlike_dict[m] = float(qlike(res_df["y"], res_df[m]).mean())
            mse_dict[m] = float(mse_logv(res_df["y"], res_df[m]))
        else:
            qlike_dict[m] = np.nan
            mse_dict[m] = np.nan

    # DM tests
    dm_lag = max(20, 2 * H)
    loss_har = qlike(res_df["y"], res_df["har"])
    loss_obs = qlike(res_df["y"], res_df["str_obs"])
    loss_ssm = qlike(res_df["y"], res_df["str_ssm"])

    dm_har_obs = dm_test(loss_har, loss_obs, L=dm_lag)
    dm_har_ssm = dm_test(loss_har, loss_ssm, L=dm_lag)
    dm_obs_ssm = dm_test(loss_obs, loss_ssm, L=dm_lag)

    # MCS (Model Confidence Set)
    if verbose:
        print(f"\n[5/6] Running Model Confidence Set (MCS)...")

    mcs_config = config.get("mcs", {})
    mcs_alpha = mcs_config.get("alpha", 0.10)
    mcs_block_size = mcs_config.get("block_size", 10)
    mcs_n_boot = mcs_config.get("n_bootstrap", 1000)

    # Build QLIKE loss DataFrame for MCS (include ablation models)
    losses_df = pd.DataFrame(index=res_df.index)
    for m in all_models:
        if m in res_df.columns and res_df[m].notna().any():
            losses_df[m] = qlike(res_df["y"], res_df[m])

    mcs_models, mcs_pvals = bootstrap_mcs(
        losses_df, alpha=mcs_alpha, n_boot=mcs_n_boot, block_size=mcs_block_size
    )

    # ============ Risk evaluation (VaR/ES/FZ0) ============
    if verbose:
        print(f"\n[6/7] Running risk evaluation (VaR/ES/FZ0)...")

    risk_config = config.get("risk", {})
    alphas_risk = risk_config.get("alphas", [0.01, 0.05])
    alpha_fz_dm = risk_config.get("alpha_fz_dm", 0.01)
    nonoverlapping = risk_config.get("nonoverlapping", True)
    rolling_nu = risk_config.get("rolling_nu", True)
    mean_adjust = risk_config.get("mean_adjust", True)

    # Create non-overlapping risk evaluation sample if configured
    if nonoverlapping and len(res_df) > 0:
        mask = (np.arange(len(res_df)) % H) == 0
        res_risk = res_df.iloc[mask].copy()
    else:
        res_risk = res_df.copy()

    if verbose:
        print(f"      Risk eval uses {'NON-overlapping' if nonoverlapping else 'overlapping'} blocks. Rows={len(res_risk)}")

    # Generate FZ tables for each model
    # Model names in risk tables: har, obs, ssm, garch, egarch, msgarch
    risk_model_names = ["har", "obs", "ssm", "garch", "egarch", "msgarch"]
    fz_tables = []
    for model_name in risk_model_names:
        tbl = risk_table_fz_es_dynamic(
            res_risk, model_name, alphas_risk, H,
            mean_adjust=mean_adjust, rolling_nu=rolling_nu
        )
        if not tbl.empty:
            fz_tables.append(tbl)

    if fz_tables:
        fz_table = pd.concat(fz_tables, ignore_index=True)
        fz_table["asset"] = ticker
        fz_table["nonoverlap"] = bool(nonoverlapping)
        fz_table["rolling_nu"] = bool(rolling_nu)
        fz_table["mean_adjust"] = bool(mean_adjust)
    else:
        fz_table = pd.DataFrame()

    # DM tests on FZ0 loss
    dm_fz = {}
    try:
        L_har = fz_loss_series_dynamic(res_risk, "har", alpha_fz_dm, H, mean_adjust, rolling_nu)
        L_obs = fz_loss_series_dynamic(res_risk, "obs", alpha_fz_dm, H, mean_adjust, rolling_nu)
        L_ssm = fz_loss_series_dynamic(res_risk, "ssm", alpha_fz_dm, H, mean_adjust, rolling_nu)

        dm_fz["har_vs_obs"] = dm_test(L_har, L_obs, L=dm_lag)
        dm_fz["har_vs_ssm"] = dm_test(L_har, L_ssm, L=dm_lag)
        dm_fz["obs_vs_ssm"] = dm_test(L_obs, L_ssm, L=dm_lag)
    except Exception as e:
        if verbose:
            print(f"      Warning: DM test on FZ0 failed: {e}")
        dm_fz = {"har_vs_obs": (np.nan, np.nan), "har_vs_ssm": (np.nan, np.nan), "obs_vs_ssm": (np.nan, np.nan)}

    # Print summary
    if verbose:
        print(f"\n[7/7] Results Summary")
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY: {ticker} (H={H})")
        print(f"{'='*60}")
        print(f"\nForecast Accuracy (QLIKE):")
        for m in all_models:
            if m in qlike_dict and not np.isnan(qlike_dict[m]):
                print(f"  {m:20s}: {qlike_dict[m]:.4f}")
        print(f"\nMSE (log variance):")
        for m in all_models:
            if m in mse_dict and not np.isnan(mse_dict[m]):
                print(f"  {m:20s}: {mse_dict[m]:.4f}")
        print(f"\nDiebold-Mariano Tests (QLIKE, HAC lag={dm_lag}):")
        print(f"  HAR vs OBS: stat={dm_har_obs[0]:.3f}, p={dm_har_obs[1]:.4f}")
        print(f"  HAR vs SSM: stat={dm_har_ssm[0]:.3f}, p={dm_har_ssm[1]:.4f}")
        print(f"  OBS vs SSM: stat={dm_obs_ssm[0]:.3f}, p={dm_obs_ssm[1]:.4f}")
        print(f"\nModel Confidence Set (MCS, alpha={mcs_alpha}):")
        print(f"  Included models: {mcs_models}")
        print(f"  P-values: {mcs_pvals}")

        # Ablation summary if in ablation mode
        if is_ablation_mode(config):
            print(f"\n{'='*60}")
            print("ABLATION RESULTS (Transition Smoothers)")
            print(f"{'='*60}")
            ablation_qlike = [(m, qlike_dict[m]) for m in ablation_model_cols if m in qlike_dict and not np.isnan(qlike_dict[m])]
            ablation_qlike_sorted = sorted(ablation_qlike, key=lambda x: x[1])
            for m, q in ablation_qlike_sorted:
                print(f"  {m:20s}: QLIKE={q:.4f}")
            if har_variant_cols:
                print(f"\nHAR Variants:")
                for m in har_variant_cols:
                    if m in qlike_dict and not np.isnan(qlike_dict[m]):
                        print(f"  {m:20s}: QLIKE={qlike_dict[m]:.4f}")

        # Risk evaluation summary
        if not fz_table.empty:
            print(f"\nRisk Evaluation (VaR/ES, alpha={alpha_fz_dm}):")
            show = fz_table[fz_table["alpha"] == alpha_fz_dm][["model", "viol_rate", "mean_FZ0", "kupiec_p"]]
            print(show.to_string(index=False))

            print(f"\nDM tests on FZ0 (HAC lag={dm_lag}):")
            for k, (st, pv) in dm_fz.items():
                print(f"  {k}: stat={st:.3f}, p={pv:.4f}")

    # Prepare output
    output = {
        "ticker": ticker,
        "H": H,
        "results": res_df,
        "res_risk": res_risk,
        "qlike": qlike_dict,
        "mse": mse_dict,
        "dm_tests": {
            "har_vs_obs": dm_har_obs,
            "har_vs_ssm": dm_har_ssm,
            "obs_vs_ssm": dm_obs_ssm,
        },
        "dm_fz": dm_fz,
        "mcs": {
            "included": mcs_models,
            "pvals": mcs_pvals,
        },
        "fz_table": fz_table,
        "elbo_diagnostics": {
            "dates": elbo_step_dates,
            "values": elbo_step_values,
        },
        "ablation": {
            "smoothers": ablation_model_cols,
            "har_variants": har_variant_cols,
            "enabled": is_ablation_mode(config),
        },
    }

    # Save results
    safe_name = ticker.replace("=", "").replace("^", "")

    # Save overlapping OOS results
    res_df.to_csv(os.path.join(save_dir, f"{safe_name}_oos_overlapping.csv"))

    # Save non-overlapping risk evaluation sample
    res_risk.to_csv(os.path.join(save_dir, f"{safe_name}_oos_risk_nonoverlap.csv"))

    # Save FZ table (VaR/ES metrics)
    if not fz_table.empty:
        fz_table.to_csv(os.path.join(save_dir, f"{safe_name}_fz_table.csv"), index=False)

    # Save MCS results
    mcs_df = pd.DataFrame({
        "model": list(mcs_pvals.keys()),
        "mcs_pval": list(mcs_pvals.values()),
        "in_mcs": [m in mcs_models for m in mcs_pvals.keys()],
    })
    mcs_df.to_csv(os.path.join(save_dir, f"{safe_name}_H{H}_mcs.csv"), index=False)

    # Save accuracy summary (for easy aggregation) - include all evaluated models
    acc_models = [m for m in all_models if m in qlike_dict]
    acc_df = pd.DataFrame({
        "model": acc_models,
        "QLIKE": [qlike_dict.get(m, np.nan) for m in acc_models],
        "MSE_logV": [mse_dict.get(m, np.nan) for m in acc_models],
    })
    acc_df.to_csv(os.path.join(save_dir, f"{safe_name}_accuracy.csv"), index=False)

    # Save ablation-specific results if in ablation mode
    if is_ablation_mode(config) and (ablation_model_cols or har_variant_cols):
        ablation_df = pd.DataFrame({
            "model": ablation_model_cols + har_variant_cols,
            "QLIKE": [qlike_dict.get(m, np.nan) for m in ablation_model_cols + har_variant_cols],
            "MSE_logV": [mse_dict.get(m, np.nan) for m in ablation_model_cols + har_variant_cols],
        })
        ablation_df.to_csv(os.path.join(save_dir, f"{safe_name}_ablation.csv"), index=False)

    # Save results marker file (for resumability check - matches is_run_complete)
    res_df.to_csv(os.path.join(save_dir, f"{safe_name}_H{H}_results.csv"))

    # Keep checkpoints for inspection/debugging (no cleanup)

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Run Deep-LSTR volatility forecasting for a single asset"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Asset ticker symbol (e.g., ^GSPC, CL=F)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Forecast horizon in days (default: 5)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Experiment directory (uses config experiment.id if not specified)",
    )
    parser.add_argument(
        "--exp-id",
        type=str,
        default=None,
        help="Experiment ID (overrides config)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable checkpoint resumability",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()

    config = load_config(args.config) if args.config else load_config()

    # Override experiment settings from CLI
    if args.exp_id:
        config.setdefault("experiment", {})["id"] = args.exp_id
    if args.no_resume:
        config.setdefault("experiment", {})["resume"] = False

    # Determine experiment directory
    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = get_experiment_dir(config)

    run_single_asset(
        ticker=args.ticker,
        H=args.horizon,
        config=config,
        exp_dir=exp_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
