"""
Crisis event study analysis for latent volatility regimes.

Examines regime behavior around known stress episodes to assess whether
latent regimes activate before, during, and after crises in a manner
consistent with economic stress interpretation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union


# Known crisis episodes within the Paper 1 OOS window (2017-06 to 2025-12)
_EPISODES = [
    {
        "name": "Volmageddon",
        "start": "2018-02-01",
        "peak": "2018-02-05",
        "end": "2018-02-28",
        "description": "XIV collapse, VIX spike",
    },
    {
        "name": "COVID Crash",
        "start": "2020-02-20",
        "peak": "2020-03-16",
        "end": "2020-04-30",
        "description": "Pandemic market crash",
    },
    {
        "name": "Rate Shock 2022",
        "start": "2022-09-01",
        "peak": "2022-10-15",
        "end": "2022-11-30",
        "description": "Fed tightening, bond selloff",
    },
    {
        "name": "SVB Crisis",
        "start": "2023-03-08",
        "peak": "2023-03-13",
        "end": "2023-03-31",
        "description": "Regional banking stress",
    },
]


def define_episodes() -> List[Dict[str, str]]:
    """
    Return list of crisis episode definitions.

    Each episode is a dict with keys: name, start, peak, end, description.
    Dates are ISO-format strings. The peak date is the single day of maximum
    stress used as the event-time anchor (t=0).

    Returns:
        List of episode definition dicts
    """
    return [ep.copy() for ep in _EPISODES]


def _nearest_trading_day(
    date: pd.Timestamp, trading_dates: pd.DatetimeIndex, direction: str = "forward"
) -> pd.Timestamp:
    """
    Find nearest trading day in the index if the target date is not a trading day.

    Args:
        date: Target date
        trading_dates: DatetimeIndex of available trading days
        direction: 'forward' snaps to next trading day, 'backward' to previous

    Returns:
        Nearest trading day as pd.Timestamp
    """
    if date in trading_dates:
        return date

    if direction == "forward":
        future = trading_dates[trading_dates >= date]
        if len(future) == 0:
            return trading_dates[-1]
        return future[0]
    else:
        past = trading_dates[trading_dates <= date]
        if len(past) == 0:
            return trading_dates[0]
        return past[-1]


def extract_episode_window(
    df: pd.DataFrame,
    episode: Dict[str, str],
    before: int = 20,
    after: int = 40,
) -> pd.DataFrame:
    """
    Extract a window of trading days around an episode peak.

    The returned DataFrame has an additional column `t` indicating trading days
    relative to the peak (t=0 is peak day). Regime columns (G_obs, G_ssm) are
    normalized by subtracting their pre-window mean (the `before` days preceding
    the episode start date) so that deviations are interpretable.

    Args:
        df: DataFrame with DatetimeIndex (or 'date' column) containing at
            minimum G_obs and G_ssm columns
        episode: Episode dict from define_episodes()
        before: Trading days before episode start to include
        after: Trading days after peak to include

    Returns:
        DataFrame with columns from df plus `t` (relative trading day).
        Regime columns are mean-normalized to the pre-window baseline.
    """
    if "date" in df.columns:
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    trading_dates = df.index.sort_values()

    peak_date = _nearest_trading_day(
        pd.Timestamp(episode["peak"]), trading_dates, direction="backward"
    )
    start_date = _nearest_trading_day(
        pd.Timestamp(episode["start"]), trading_dates, direction="backward"
    )

    # Locate positions in the sorted index
    peak_idx = trading_dates.get_loc(peak_date)
    start_idx = trading_dates.get_loc(start_date)

    # Window boundaries (in index positions)
    window_start = max(0, start_idx - before)
    window_end = min(len(trading_dates) - 1, peak_idx + after)

    window = df.iloc[window_start : window_end + 1].copy()

    # Compute relative trading-day index (t=0 at peak)
    peak_pos_in_window = window.index.get_loc(peak_date)
    window["t"] = np.arange(len(window)) - peak_pos_in_window

    # Pre-window baseline: the `before` days preceding episode start
    baseline_start = max(0, start_idx - before)
    baseline_end = start_idx
    baseline_slice = df.iloc[baseline_start:baseline_end]

    # Normalize regime columns to pre-window mean
    regime_cols = [c for c in ["G_obs", "G_ssm"] if c in window.columns]
    for col in regime_cols:
        baseline_mean = baseline_slice[col].mean()
        if np.isfinite(baseline_mean):
            window[col] = window[col] - baseline_mean

    return window.reset_index().rename(columns={"index": "date"})


def regime_timing(
    G_series: pd.Series,
    episode: Dict[str, str],
    activation_quantile: float = 0.75,
) -> Dict[str, Union[float, int, None]]:
    """
    Compute regime activation timing relative to a crisis episode.

    Uses quantile-based activation thresholds rather than fixed 0.5,
    because G_ssm is concentrated near 0.5 (std ~ 0.107) while G_obs
    has wider spread (std ~ 0.267).

    Args:
        G_series: Series with DatetimeIndex containing regime transition
            function values (G_obs or G_ssm), NOT normalized
        episode: Episode dict from define_episodes()
        activation_quantile: Quantile of the G distribution to use as
            activation threshold (default 0.75 = 75th percentile)

    Returns:
        Dict with:
            first_activation: Date when G first exceeds its activation threshold
                within the episode window
            days_before_peak: Trading days between first_activation and peak
                (positive = activation leads peak)
            peak_G: Maximum G value during the episode [start, end]
            days_elevated: Number of trading days after peak date where G
                remains above the activation threshold
    """
    if not isinstance(G_series.index, pd.DatetimeIndex):
        G_series = G_series.copy()
        G_series.index = pd.to_datetime(G_series.index)

    trading_dates = G_series.index.sort_values()

    start = _nearest_trading_day(
        pd.Timestamp(episode["start"]), trading_dates, direction="forward"
    )
    peak = _nearest_trading_day(
        pd.Timestamp(episode["peak"]), trading_dates, direction="backward"
    )
    end = _nearest_trading_day(
        pd.Timestamp(episode["end"]), trading_dates, direction="backward"
    )

    # Activation threshold: quantile of the full G series
    threshold = G_series.quantile(activation_quantile)

    # Episode window
    episode_mask = (G_series.index >= start) & (G_series.index <= end)
    episode_G = G_series[episode_mask]

    if len(episode_G) == 0:
        return {
            "first_activation": None,
            "days_before_peak": None,
            "peak_G": None,
            "days_elevated": None,
        }

    # First activation: first day G exceeds threshold within episode window
    activated = episode_G[episode_G >= threshold]
    if len(activated) > 0:
        first_activation = activated.index[0]
        # Count trading days between first_activation and peak
        days_between = trading_dates[
            (trading_dates >= first_activation) & (trading_dates <= peak)
        ]
        days_before_peak = len(days_between) - 1  # exclude the peak day itself
        if first_activation > peak:
            days_before_peak = -days_before_peak  # activation AFTER peak
    else:
        first_activation = None
        days_before_peak = None

    # Peak G value during episode
    peak_G = float(episode_G.max())

    # Days elevated after peak: how many trading days after peak
    # does G remain above threshold
    post_peak_mask = (G_series.index > peak) & (G_series.index <= end)
    post_peak_G = G_series[post_peak_mask]
    if len(post_peak_G) > 0:
        # Count consecutive days above threshold from peak onwards
        above = post_peak_G >= threshold
        # Count total days above threshold (not necessarily consecutive)
        days_elevated = int(above.sum())
    else:
        days_elevated = 0

    return {
        "first_activation": first_activation,
        "days_before_peak": days_before_peak,
        "peak_G": peak_G,
        "days_elevated": days_elevated,
    }


def episode_summary(
    regime_dict: Dict[str, pd.DataFrame],
    episodes: List[Dict[str, str]],
    regime_type: str = "G_ssm",
    primary_asset: str = "GSPC",
) -> pd.DataFrame:
    """
    Summarize regime timing across episodes for primary asset and panel mean.

    Args:
        regime_dict: Dict mapping ticker -> DataFrame with columns including
            'date', 'G_obs', 'G_ssm'
        episodes: List of episode dicts from define_episodes()
        regime_type: Which regime column to analyze ('G_ssm' or 'G_obs')
        primary_asset: Ticker to use as primary illustration (default 'GSPC')

    Returns:
        DataFrame with columns: episode, asset, first_activation,
        days_before_peak, peak_G, days_elevated.
        Includes rows for primary_asset and 'Panel Mean'.
    """
    rows = []

    for ep in episodes:
        # Primary asset
        if primary_asset in regime_dict:
            df = regime_dict[primary_asset]
            G_series = _extract_G_series(df, regime_type)
            timing = regime_timing(G_series, ep)
            rows.append({
                "episode": ep["name"],
                "asset": primary_asset,
                "first_activation": timing["first_activation"],
                "days_before_peak": timing["days_before_peak"],
                "peak_G": timing["peak_G"],
                "days_elevated": timing["days_elevated"],
            })

        # Panel mean across all assets
        panel_days_before = []
        panel_peak_G = []
        panel_days_elevated = []

        for ticker, df in regime_dict.items():
            G_series = _extract_G_series(df, regime_type)
            timing = regime_timing(G_series, ep)
            if timing["days_before_peak"] is not None:
                panel_days_before.append(timing["days_before_peak"])
            if timing["peak_G"] is not None:
                panel_peak_G.append(timing["peak_G"])
            if timing["days_elevated"] is not None:
                panel_days_elevated.append(timing["days_elevated"])

        rows.append({
            "episode": ep["name"],
            "asset": "Panel Mean",
            "first_activation": None,
            "days_before_peak": (
                np.mean(panel_days_before) if panel_days_before else None
            ),
            "peak_G": np.mean(panel_peak_G) if panel_peak_G else None,
            "days_elevated": (
                np.mean(panel_days_elevated) if panel_days_elevated else None
            ),
        })

    result = pd.DataFrame(rows)
    return result


def compare_obs_vs_latent(
    regime_dict: Dict[str, pd.DataFrame],
    episodes: List[Dict[str, str]],
    primary_asset: str = "GSPC",
) -> pd.DataFrame:
    """
    Side-by-side comparison of G_obs and G_ssm timing for each episode.

    Args:
        regime_dict: Dict mapping ticker -> DataFrame
        episodes: List of episode dicts from define_episodes()
        primary_asset: Ticker to use (default 'GSPC')

    Returns:
        DataFrame with columns: episode, metric, G_obs, G_ssm, difference.
        Metrics include days_before_peak, peak_G, days_elevated for both
        the primary asset and the panel mean.
    """
    if primary_asset not in regime_dict:
        available = list(regime_dict.keys())
        raise ValueError(
            f"Primary asset '{primary_asset}' not found. "
            f"Available: {available[:5]}..."
        )

    rows = []

    for ep in episodes:
        df = regime_dict[primary_asset]

        # G_obs timing
        G_obs_series = _extract_G_series(df, "G_obs")
        timing_obs = regime_timing(G_obs_series, ep, activation_quantile=0.75)

        # G_ssm timing
        G_ssm_series = _extract_G_series(df, "G_ssm")
        timing_ssm = regime_timing(G_ssm_series, ep, activation_quantile=0.75)

        # Primary asset rows
        for metric_key, metric_label in [
            ("days_before_peak", "Days before peak"),
            ("peak_G", "Peak G"),
            ("days_elevated", "Days elevated"),
        ]:
            obs_val = timing_obs[metric_key]
            ssm_val = timing_ssm[metric_key]
            diff = None
            if obs_val is not None and ssm_val is not None:
                diff = ssm_val - obs_val

            rows.append({
                "episode": ep["name"],
                "scope": primary_asset,
                "metric": metric_label,
                "G_obs": obs_val,
                "G_ssm": ssm_val,
                "difference": diff,
            })

        # Panel mean
        panel_obs = {"days_before_peak": [], "peak_G": [], "days_elevated": []}
        panel_ssm = {"days_before_peak": [], "peak_G": [], "days_elevated": []}

        for ticker, ticker_df in regime_dict.items():
            G_obs_s = _extract_G_series(ticker_df, "G_obs")
            G_ssm_s = _extract_G_series(ticker_df, "G_ssm")
            t_obs = regime_timing(G_obs_s, ep, activation_quantile=0.75)
            t_ssm = regime_timing(G_ssm_s, ep, activation_quantile=0.75)

            for key in panel_obs:
                if t_obs[key] is not None:
                    panel_obs[key].append(t_obs[key])
                if t_ssm[key] is not None:
                    panel_ssm[key].append(t_ssm[key])

        for metric_key, metric_label in [
            ("days_before_peak", "Days before peak"),
            ("peak_G", "Peak G"),
            ("days_elevated", "Days elevated"),
        ]:
            obs_mean = (
                np.mean(panel_obs[metric_key])
                if panel_obs[metric_key]
                else None
            )
            ssm_mean = (
                np.mean(panel_ssm[metric_key])
                if panel_ssm[metric_key]
                else None
            )
            diff = None
            if obs_mean is not None and ssm_mean is not None:
                diff = ssm_mean - obs_mean

            rows.append({
                "episode": ep["name"],
                "scope": "Panel Mean",
                "metric": metric_label,
                "G_obs": obs_mean,
                "G_ssm": ssm_mean,
                "difference": diff,
            })

    return pd.DataFrame(rows)


def _extract_G_series(
    df: pd.DataFrame, regime_type: str
) -> pd.Series:
    """
    Extract a regime transition function column as a pd.Series with
    DatetimeIndex, handling both indexed and column-based date formats.

    Args:
        df: DataFrame from regime loader (has 'date' column or DatetimeIndex)
        regime_type: Column name ('G_obs' or 'G_ssm')

    Returns:
        pd.Series with DatetimeIndex
    """
    if regime_type not in df.columns:
        raise ValueError(
            f"Column '{regime_type}' not found. "
            f"Available: {list(df.columns)}"
        )

    if "date" in df.columns:
        series = df.set_index("date")[regime_type]
    else:
        series = df[regime_type]

    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index)

    return series
