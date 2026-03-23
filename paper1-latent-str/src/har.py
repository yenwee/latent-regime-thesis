# src/har.py
"""
HAR-family models for realized volatility forecasting.

Implements:
- HAR: Standard Heterogeneous Autoregressive (Corsi 2009)
- LHAR: Leverage HAR (Corsi & Reno 2012)
- HAR-J: HAR with Jump component (Andersen, Bollerslev, Diebold 2007)

References:
    Corsi, F. (2009). A Simple Approximate Long-Memory Model of
    Realized Volatility. Journal of Financial Econometrics, 7(2), 174-196.

    Corsi, F. & Reno, R. (2012). Discrete-Time Volatility Forecasting
    with Persistent Leverage Effect and the Link with Continuous-Time
    Volatility Modeling. Journal of Business & Economic Statistics, 30(3).

    Andersen, T.G., Bollerslev, T. & Diebold, F.X. (2007). Roughing It Up:
    Including Jump Components in the Measurement, Modeling, and Forecasting
    of Return Volatility. Review of Economics and Statistics, 89(4), 701-720.
"""

import numpy as np
import pandas as pd


def fit_lhar_ols(train_df: pd.DataFrame) -> np.ndarray:
    """
    Fit Leverage HAR model by OLS.

    Model: y = b0 + b_d * x_d + b_w * x_w + b_m * x_m + b_lev * r_neg

    The leverage term r_neg = min(r, 0) captures the asymmetric response
    of volatility to negative returns (leverage effect).

    Args:
        train_df: DataFrame with columns 'y', 'x_d', 'x_w', 'x_m', 'r'
                  where 'r' is daily log return

    Returns:
        Array of coefficients [b0, b_d, b_w, b_m, b_lev]

    References:
        Corsi, F. & Reno, R. (2012). Discrete-Time Volatility Forecasting
        with Persistent Leverage Effect. JBES 30(3), 368-380.
    """
    Y = train_df["y"].values

    # Leverage term: negative returns only
    r_neg = np.minimum(train_df["r"].values, 0)

    X = np.column_stack([
        np.ones(len(train_df)),
        train_df["x_d"].values,
        train_df["x_w"].values,
        train_df["x_m"].values,
        r_neg,
    ])

    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta


def lhar_predict(beta: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """
    Generate LHAR predictions.

    Args:
        beta: LHAR coefficients [b0, b_d, b_w, b_m, b_lev]
        df: DataFrame with 'x_d', 'x_w', 'x_m', 'r' columns

    Returns:
        Array of predictions
    """
    r_neg = np.minimum(df["r"].values, 0)

    X = np.column_stack([
        np.ones(len(df)),
        df["x_d"].values,
        df["x_w"].values,
        df["x_m"].values,
        r_neg,
    ])

    return X @ beta


def fit_har_j_ols(train_df: pd.DataFrame) -> np.ndarray:
    """
    Fit HAR-J model (HAR with jump component) by OLS.

    Model: y = b0 + b_d * x_d + b_w * x_w + b_m * x_m + b_j * j_d

    where j_d is the daily jump component = max(RV - BPV, 0).

    Separating jumps from continuous variation improves forecasts
    because jumps are less persistent than continuous volatility.

    Args:
        train_df: DataFrame with 'y', 'x_d', 'x_w', 'x_m', 'j_d' columns
                  where 'j_d' is the jump component (log scale)

    Returns:
        Array of coefficients [b0, b_d, b_w, b_m, b_j]

    References:
        Andersen, T.G., Bollerslev, T. & Diebold, F.X. (2007). Roughing
        It Up: Including Jump Components in the Measurement, Modeling,
        and Forecasting of Return Volatility. REStat, 89(4), 701-720.
    """
    Y = train_df["y"].values

    X = np.column_stack([
        np.ones(len(train_df)),
        train_df["x_d"].values,
        train_df["x_w"].values,
        train_df["x_m"].values,
        train_df["j_d"].values,
    ])

    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta


def har_j_predict(beta: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """
    Generate HAR-J predictions.

    Args:
        beta: HAR-J coefficients [b0, b_d, b_w, b_m, b_j]
        df: DataFrame with 'x_d', 'x_w', 'x_m', 'j_d' columns

    Returns:
        Array of predictions
    """
    X = np.column_stack([
        np.ones(len(df)),
        df["x_d"].values,
        df["x_w"].values,
        df["x_m"].values,
        df["j_d"].values,
    ])

    return X @ beta
