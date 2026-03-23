"""
Forecast evaluation metrics for Deep-LSTR.
"""

import numpy as np

# Numerical stability constant
EPS = 1e-12


def qlike(y_true_logv, y_pred_logv):
    """
    QLIKE loss function for volatility forecasting.
    Quasi-likelihood loss that penalizes both over- and under-prediction.

    Args:
        y_true_logv: True log variance
        y_pred_logv: Predicted log variance

    Returns:
        Array of QLIKE losses (element-wise)
    """
    v_true = np.exp(y_true_logv)
    v_hat = np.exp(y_pred_logv)
    return np.log(v_hat + EPS) + (v_true / (v_hat + EPS))


def mse_logv(y_true_logv, y_pred_logv):
    """
    Mean squared error on log variance.

    Args:
        y_true_logv: True log variance
        y_pred_logv: Predicted log variance

    Returns:
        Scalar MSE value
    """
    return np.mean((y_true_logv - y_pred_logv) ** 2)


def fz0_loss(y, v, e, alpha):
    """
    FZ0 loss (Patton-Ziegel-Chen style) for joint VaR+ES evaluation.
    Requires e < v < 0 (left tail); enforce numerically.

    This is the proper elicitable scoring rule for Expected Shortfall
    as established by Fissler & Ziegel (2016).

    Args:
        y: Realized returns (H-day aggregated)
        v: VaR forecasts (negative for left tail)
        e: ES forecasts (negative, more extreme than VaR)
        alpha: Confidence level (e.g., 0.01, 0.025, 0.05)

    Returns:
        Array of FZ0 losses (element-wise)
    """
    y = np.asarray(y)
    v = np.asarray(v)
    e = np.asarray(e)

    # Enforce v < 0 and e < v for numerical stability
    v_safe = np.minimum(v, -1e-12)
    e_safe = np.minimum(e, v_safe - 1e-12)

    I = (y <= v_safe).astype(float)
    L = -(1.0 / (alpha * e_safe)) * I * (v_safe - y) + (v_safe / e_safe) + np.log(-e_safe) - 1.0
    return L
