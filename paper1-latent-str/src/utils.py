"""
Utility functions for Deep-LSTR.
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import yaml
from pathlib import Path


def set_seeds(seed: int = 123):
    """Set random seeds for reproducibility across numpy and torch."""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(config_path: str = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        # Default to configs/default.yaml relative to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "configs", "default.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ensure_writable_dir(preferred_dir: str) -> str:
    """
    Try preferred_dir; if not writable, fall back to /tmp/results.
    """
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


def set_thread_limits():
    """
    Limit intra-process threading to maximize inter-process parallelism.
    Should be called before importing numpy/scipy in parallel workers.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Force PyTorch to use single thread per process
    torch.set_num_threads(1)


def make_retrain_schedule(index: pd.DatetimeIndex, oos_start_date: pd.Timestamp, freq: str):
    """
    Create retrain schedule for expanding-window SSM training.

    Generates boundaries at period ends and maps to first available date on/after boundary.
    Ensures OOS start is included as first boundary.

    Args:
        index: DatetimeIndex of the data
        oos_start_date: First date of out-of-sample period
        freq: Pandas frequency string (e.g., "A" for annual, "Q" for quarterly)

    Returns:
        List of dates when SSM should be retrained
    """
    idx = pd.DatetimeIndex(index).sort_values()
    start = oos_start_date.normalize()
    boundaries = pd.date_range(start=start, end=idx[-1], freq=freq)

    retrain_dates = []
    for b in boundaries:
        pos = idx.searchsorted(b)
        if pos < len(idx):
            d = idx[pos]
            if d >= oos_start_date:
                retrain_dates.append(d)

    retrain_dates = sorted(set(retrain_dates))
    if len(retrain_dates) == 0 or retrain_dates[0] != oos_start_date:
        retrain_dates = sorted(set([oos_start_date] + retrain_dates))
    return retrain_dates


# =============================================================================
# Experiment and Checkpoint Management
# =============================================================================

def get_experiment_dir(config: dict, base_dir: str = None) -> str:
    """
    Get or create experiment directory based on config.

    If experiment.id is set in config, uses that ID.
    Otherwise, generates timestamp-based ID.

    Args:
        config: Configuration dictionary
        base_dir: Base output directory (uses config if None)

    Returns:
        Path to experiment directory
    """
    if base_dir is None:
        base_dir = config.get("output", {}).get("base_dir", "outputs")

    exp_config = config.get("experiment", {})
    exp_id = exp_config.get("id")

    if exp_id is None:
        # Auto-generate timestamp-based ID
        exp_id = datetime.datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    exp_dir = os.path.join(base_dir, exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    return exp_dir


def get_checkpoint_path(exp_dir: str, ticker: str, H: int, checkpoint_type: str = "results") -> str:
    """
    Generate checkpoint file path for a specific asset/horizon/type.

    Args:
        exp_dir: Experiment directory
        ticker: Asset ticker
        H: Forecast horizon
        checkpoint_type: Type of checkpoint (results, ssm, forecasts, segment)

    Returns:
        Path to checkpoint file
    """
    safe_ticker = ticker.replace("=", "").replace("^", "")
    checkpoint_dir = os.path.join(exp_dir, f"H{H}", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    return os.path.join(checkpoint_dir, f"{safe_ticker}_{checkpoint_type}.pkl")


def save_checkpoint(checkpoint_path: str, data: dict, verbose: bool = False):
    """
    Save checkpoint data to disk.

    Args:
        checkpoint_path: Path to save checkpoint
        data: Dictionary of data to checkpoint
        verbose: Print save message
    """
    with open(checkpoint_path, "wb") as f:
        pickle.dump(data, f)

    if verbose:
        print(f"  [CHECKPOINT] Saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, verbose: bool = False) -> dict:
    """
    Load checkpoint data from disk.

    Args:
        checkpoint_path: Path to checkpoint file
        verbose: Print load message

    Returns:
        Dictionary of checkpoint data, or None if not found
    """
    if not os.path.exists(checkpoint_path):
        return None

    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)

    if verbose:
        print(f"  [CHECKPOINT] Loaded: {checkpoint_path}")

    return data


def check_segment_checkpoint(exp_dir: str, ticker: str, H: int, seg_id: int) -> dict:
    """
    Check if a specific segment checkpoint exists.

    Args:
        exp_dir: Experiment directory
        ticker: Asset ticker
        H: Forecast horizon
        seg_id: Segment ID

    Returns:
        Segment checkpoint data if exists, None otherwise
    """
    checkpoint_path = get_checkpoint_path(exp_dir, ticker, H, f"segment_{seg_id}")
    return load_checkpoint(checkpoint_path)


def save_segment_checkpoint(
    exp_dir: str,
    ticker: str,
    H: int,
    seg_id: int,
    q_ssm_segment: pd.Series,
    Z_infer: np.ndarray,
    elbo: float,
    seg_start: pd.Timestamp,
    seg_end: pd.Timestamp,
    verbose: bool = False,
):
    """
    Save SSM training results for a segment.

    Args:
        exp_dir: Experiment directory
        ticker: Asset ticker
        H: Forecast horizon
        seg_id: Segment ID
        q_ssm_segment: Projected q_ssm values for this segment
        Z_infer: Raw latent states
        elbo: Final ELBO value
        seg_start: Segment start date
        seg_end: Segment end date
        verbose: Print save message
    """
    checkpoint_path = get_checkpoint_path(exp_dir, ticker, H, f"segment_{seg_id}")
    data = {
        "seg_id": seg_id,
        "seg_start": seg_start,
        "seg_end": seg_end,
        "q_ssm_segment": q_ssm_segment,
        "Z_infer": Z_infer,
        "elbo": elbo,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    save_checkpoint(checkpoint_path, data, verbose=verbose)


def save_forecast_checkpoint(
    exp_dir: str,
    ticker: str,
    H: int,
    results: list,
    last_idx: int,
    verbose: bool = False,
):
    """
    Save incremental forecast results.

    Args:
        exp_dir: Experiment directory
        ticker: Asset ticker
        H: Forecast horizon
        results: List of forecast result dictionaries
        last_idx: Last processed index
        verbose: Print save message
    """
    checkpoint_path = get_checkpoint_path(exp_dir, ticker, H, "forecasts")
    data = {
        "results": results,
        "last_idx": last_idx,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    save_checkpoint(checkpoint_path, data, verbose=verbose)


def load_forecast_checkpoint(exp_dir: str, ticker: str, H: int, verbose: bool = False) -> tuple:
    """
    Load forecast checkpoint.

    Args:
        exp_dir: Experiment directory
        ticker: Asset ticker
        H: Forecast horizon
        verbose: Print load message

    Returns:
        Tuple of (results_list, last_idx) or ([], -1) if no checkpoint
    """
    checkpoint_path = get_checkpoint_path(exp_dir, ticker, H, "forecasts")
    data = load_checkpoint(checkpoint_path, verbose=verbose)

    if data is None:
        return [], -1

    return data.get("results", []), data.get("last_idx", -1)


def is_run_complete(exp_dir: str, ticker: str, H: int) -> bool:
    """
    Check if a run is already complete.

    Args:
        exp_dir: Experiment directory
        ticker: Asset ticker
        H: Forecast horizon

    Returns:
        True if final results exist
    """
    safe_ticker = ticker.replace("=", "").replace("^", "")
    h_dir = os.path.join(exp_dir, f"H{H}")
    results_path = os.path.join(h_dir, f"{safe_ticker}_H{H}_results.csv")
    return os.path.exists(results_path)


def get_completed_segments(exp_dir: str, ticker: str, H: int) -> set:
    """
    Get set of completed segment IDs.

    Args:
        exp_dir: Experiment directory
        ticker: Asset ticker
        H: Forecast horizon

    Returns:
        Set of completed segment IDs
    """
    safe_ticker = ticker.replace("=", "").replace("^", "")
    checkpoint_dir = os.path.join(exp_dir, f"H{H}", "checkpoints")

    if not os.path.exists(checkpoint_dir):
        return set()

    completed = set()
    for f in os.listdir(checkpoint_dir):
        if f.startswith(safe_ticker) and "_segment_" in f and f.endswith(".pkl"):
            try:
                seg_id = int(f.split("_segment_")[1].replace(".pkl", ""))
                completed.add(seg_id)
            except (ValueError, IndexError):
                pass

    return completed


def cleanup_checkpoints(exp_dir: str, ticker: str, H: int, verbose: bool = False):
    """
    Remove checkpoint files after successful completion.

    Args:
        exp_dir: Experiment directory
        ticker: Asset ticker
        H: Forecast horizon
        verbose: Print cleanup message
    """
    safe_ticker = ticker.replace("=", "").replace("^", "")
    checkpoint_dir = os.path.join(exp_dir, f"H{H}", "checkpoints")

    if not os.path.exists(checkpoint_dir):
        return

    removed = 0
    for f in os.listdir(checkpoint_dir):
        if f.startswith(safe_ticker):
            os.remove(os.path.join(checkpoint_dir, f))
            removed += 1

    if verbose and removed > 0:
        print(f"  [CLEANUP] Removed {removed} checkpoint files for {ticker}")
