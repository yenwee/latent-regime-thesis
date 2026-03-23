"""
Risk evaluation functions for VaR and Expected Shortfall.

Key parameters:
- USE_ROLLING_K = True
- ROLLING_NU = True
- MEAN_ADJUST = True
- RISK_NONOVERLAPPING = True
- ALPHAS_RISK = [0.01, 0.05]
"""

import numpy as np
import pandas as pd
from scipy.stats import t as student_t, chi2, norm
from scipy.optimize import minimize

# Constants matching 
EPS = 1e-12
DEFAULT_ALPHAS = [0.01, 0.05]


def _newey_west_var(x, L):
    """
    Newey-West HAC variance estimator.
    Inlined from dm_test.py to avoid import issues.
    """
    x = np.asarray(x)
    Tn = len(x)
    x0 = x - x.mean()
    gamma0 = np.dot(x0, x0) / Tn
    S = gamma0
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        gam = np.dot(x0[l:], x0[:-l]) / Tn
        S += 2.0 * w * gam
    return S / Tn


def _dm_test(loss_a, loss_b, L=5):
    """
    Diebold-Mariano test for equal predictive accuracy.
    Inlined from dm_test.py to avoid import issues.
    """
    a = np.asarray(loss_a)
    b = np.asarray(loss_b)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]

    if len(a) < (L + 5):
        return np.nan, np.nan

    d = a - b
    var_mean = _newey_west_var(d, L)
    if var_mean <= 0 or not np.isfinite(var_mean):
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var_mean + EPS)
    p = 2.0 * (1.0 - norm.cdf(np.abs(dm)))
    return float(dm), float(p)


def t_var_es_var1(alpha, nu):
    """
    Compute VaR and ES multipliers for Student-t innovations standardized to Var=1.

    For a Student-t distribution with nu degrees of freedom, we scale to unit
    variance (Var = 1) rather than the natural Var = nu/(nu-2).

    Args:
        alpha: Tail probability (e.g., 0.01, 0.025, 0.05)
        nu: Degrees of freedom

    Returns:
        Tuple of (q_alpha, es_alpha) - both negative for left tail
    """
    nu = float(nu)
    if nu <= 2.0:
        nu = 2.01
    s = np.sqrt(nu / (nu - 2.0))
    qZ = student_t.ppf(alpha, df=nu)  # negative
    fZ = student_t.pdf(qZ, df=nu)
    esZ = -((nu + qZ ** 2) / (nu - 1.0)) * (fZ / alpha)
    return float(qZ / s), float(esZ / s)


def fit_nu_mle_var1(eps_hat, nu0=8.0):
    """
    Fit Student-t degrees of freedom by MLE for Var=1 standardized residuals.

    Assumes eps_hat ~ t_nu scaled to Var=1.
    If Z ~ t_nu, Var(Z) = nu/(nu-2). Let eps = Z/s, s = sqrt(nu/(nu-2)) => Var(eps) = 1.

    Args:
        eps_hat: Array of standardized residuals
        nu0: Initial guess for nu

    Returns:
        Estimated nu (constrained to [2.05, 200])
    """
    eps_hat = np.asarray(eps_hat)
    eps_hat = eps_hat[np.isfinite(eps_hat)]
    if len(eps_hat) < 80:
        return float(nu0)

    def nll(theta):
        nu = 2.0 + np.exp(theta[0])  # enforce nu > 2
        s = np.sqrt(nu / (nu - 2.0))
        z = eps_hat * s
        ll = student_t.logpdf(z, df=nu) + np.log(s)
        return -float(np.sum(ll))

    theta0 = np.array([np.log(max(nu0 - 2.0, 1e-3))])
    res = minimize(
        nll, theta0, method="L-BFGS-B", bounds=[(np.log(1e-4), np.log(1e4))]
    )
    if not res.success:
        return float(nu0)
    nu_hat = 2.0 + np.exp(res.x[0])
    return float(np.clip(nu_hat, 2.05, 200.0))


def risk_series_var_es_dynamic(df_res, model_name, alpha, H, mean_adjust=True, rolling_nu=True):
    """
    Compute dynamic VaR and ES series using rolling nu and mean adjustment.

    VaR_t = mu_H + q_mult(nu_t) * k_t * sigma_H
    ES_t  = mu_H + es_mult(nu_t) * k_t * sigma_H

    where sigma_H = sqrt(H * exp(yhat))

    Args:
        df_res: Results DataFrame with columns:
            - 'rH': H-day aggregated returns
            - model forecast column (e.g., 'har', 'str_obs', 'str_ssm')
            - optional: 'k_{model}', 'nu_{model}', 'muH'
        model_name: Model identifier ('har', 'obs', 'ssm', 'garch', etc.)
        alpha: VaR/ES confidence level
        H: Forecast horizon
        mean_adjust: Use rolling mean adjustment
        rolling_nu: Use rolling Student-t degrees of freedom

    Returns:
        Tuple of (rH, VaR, ES) arrays
    """
    rH = df_res["rH"].values

    # Map model name to forecast column
    nm_map = {
        "har": "har",
        "obs": "str_obs",
        "ssm": "str_ssm",
        "garch": "garch",
        "egarch": "egarch",
        "msgarch": "msgarch",
    }
    yhat_col = nm_map.get(model_name, model_name)

    # Volatility scaling factor k_t
    k_col = f"k_{model_name}"
    if k_col in df_res.columns:
        k_t = df_res[k_col].values
    else:
        k_t = np.ones_like(rH)

    sig_H = np.sqrt(H * np.exp(df_res[yhat_col].values))
    base = k_t * sig_H

    # Mean adjustment
    if mean_adjust and ("muH" in df_res.columns):
        muH = df_res["muH"].values
    else:
        muH = np.zeros_like(base)

    # Rolling degrees of freedom
    if rolling_nu and (f"nu_{model_name}" in df_res.columns):
        nu_t = df_res[f"nu_{model_name}"].values
    else:
        nu_t = np.full_like(base, 8.0, dtype=float)

    VaR = np.empty_like(base)
    ES = np.empty_like(base)
    for i in range(len(base)):
        q_mult, es_mult = t_var_es_var1(alpha, nu_t[i])
        VaR[i] = muH[i] + q_mult * base[i]
        ES[i] = muH[i] + es_mult * base[i]

    return rH, VaR, ES


def kupiec_test(hits, T, alpha):
    """
    Kupiec POF (Proportion of Failures) test for VaR backtesting.

    Tests H0: p = alpha (correct coverage) against two-sided alternative.

    LR = -2 * ln( (1-alpha)^(T-N) * alpha^N / (1-p_hat)^(T-N) * p_hat^N )

    where p_hat = N/T is the observed violation rate.

    Args:
        hits: Array of VaR violations (1 if violation, 0 otherwise)
        T: Total number of observations
        alpha: Nominal VaR level

    Returns:
        Tuple of (LR statistic, p-value, observed violation rate)
    """
    N = np.sum(hits)
    T = float(T)
    p_hat = N / T

    # Handle edge cases N=0 or N=T to avoid log(0)
    if N == 0:
        ll1 = 0.0
    elif N == T:
        ll1 = 0.0
    else:
        ll1 = (T - N) * np.log(1 - p_hat) + N * np.log(p_hat)

    # Under H0 (alpha)
    ll0 = (T - N) * np.log(1 - alpha) + N * np.log(alpha)

    lr_stat = -2 * (ll0 - ll1)
    lr_stat = max(0.0, lr_stat)  # Clip small negative due to numerics

    p_val = 1 - chi2.cdf(lr_stat, 1)

    return float(lr_stat), float(p_val), float(p_hat)


def fz0_loss(y, v, e, alpha):
    """
    FZ0 loss (Patton-Ziegel-Chen style) for joint VaR+ES scoring.

    This is a proper scoring rule for jointly evaluating VaR and ES forecasts.
    Requires e < v < 0 (left tail); enforced numerically.

    Args:
        y: Realized returns (array)
        v: VaR forecasts (negative for left tail)
        e: ES forecasts (negative, more extreme than VaR)
        alpha: VaR/ES confidence level

    Returns:
        Array of FZ0 loss values (lower is better)
    """
    y = np.asarray(y)
    v = np.asarray(v)
    e = np.asarray(e)

    # Enforce v < 0 and e < v
    v_safe = np.minimum(v, -1e-12)
    e_safe = np.minimum(e, v_safe - 1e-12)

    I = (y <= v_safe).astype(float)
    L = -(1.0 / (alpha * e_safe)) * I * (v_safe - y) + (v_safe / e_safe) + np.log(-e_safe) - 1.0
    return L


def compute_rolling_k(r_train_H, sigH_train):
    """
    Compute rolling volatility scaling factor k_t.

    k_t calibrates the model's predicted volatility to realized returns.
    k_t = sqrt( mean(r_H^2) / mean(sigma_H^2) )

    Args:
        r_train_H: Training window H-day returns
        sigH_train: Model-predicted H-day volatilities (sigma_H = sqrt(H * exp(yhat)))

    Returns:
        Scaling factor k (float)
    """
    r_train_H = np.asarray(r_train_H)
    sigH_train = np.asarray(sigH_train)

    # Filter to valid observations
    mask = np.isfinite(r_train_H) & np.isfinite(sigH_train)
    r_train_H = r_train_H[mask]
    sigH_train = sigH_train[mask]

    if len(r_train_H) < 20:
        return 1.0

    k = np.sqrt(np.mean(r_train_H ** 2) / (np.mean(sigH_train ** 2) + EPS))
    return float(np.clip(k, 0.1, 10.0))


def compute_rolling_nu(r_train_H, sigH_train, k, muH=0.0, nu0=8.0):
    """
    Compute rolling Student-t degrees of freedom from standardized residuals.

    Args:
        r_train_H: Training window H-day returns
        sigH_train: Model-predicted H-day volatilities
        k: Volatility scaling factor
        muH: Mean adjustment
        nu0: Initial nu estimate

    Returns:
        Estimated nu (degrees of freedom)
    """
    r_train_H = np.asarray(r_train_H)
    sigH_train = np.asarray(sigH_train)

    # Filter to valid observations
    mask = np.isfinite(r_train_H) & np.isfinite(sigH_train)
    r_train_H = r_train_H[mask]
    sigH_train = sigH_train[mask]

    if len(r_train_H) < 80:
        return float(nu0)

    # Standardized residuals
    eps = (r_train_H - muH) / (k * sigH_train + EPS)
    return fit_nu_mle_var1(eps, nu0=nu0)


def risk_table_fz_es_dynamic(df_res, model_name, alphas, H, mean_adjust=True, rolling_nu=True):
    """
    Generate FZ0 loss table for a model across multiple alpha levels.

    Args:
        df_res: Results DataFrame with columns:
            - 'rH': H-day aggregated returns
            - model forecast column
            - optional: 'k_{model}', 'nu_{model}', 'muH'
        model_name: Model identifier
        alphas: List of VaR/ES confidence levels
        H: Forecast horizon
        mean_adjust: Use rolling mean adjustment
        rolling_nu: Use rolling Student-t degrees of freedom

    Returns:
        DataFrame with risk metrics for each alpha level
    """
    # Map model name to yhat column
    nm_map = {
        "har": "har",
        "obs": "str_obs",
        "ssm": "str_ssm",
        "garch": "garch",
        "egarch": "egarch",
        "msgarch": "msgarch",
    }
    yhat_col = nm_map.get(model_name, model_name)

    # Build target columns for filtering
    target_cols = ["rH"]
    target_cols.append(yhat_col)

    k_col = f"k_{model_name}"
    if k_col in df_res.columns:
        target_cols.append(k_col)

    nu_col = f"nu_{model_name}"
    if rolling_nu and nu_col in df_res.columns:
        target_cols.append(nu_col)

    if mean_adjust and "muH" in df_res.columns:
        target_cols.append("muH")

    # Drop rows where any required column is NaN
    df_clean = df_res[target_cols].dropna()

    if len(df_clean) == 0:
        return pd.DataFrame()

    rows = []
    for a in alphas:
        # Compute VaR/ES series
        y, v, e = risk_series_var_es_dynamic(df_clean, model_name, a, H, mean_adjust, rolling_nu)

        # FZ0 loss
        L = fz0_loss(y, v, e, a)

        # Violations (returns below VaR)
        exceed = (y < v)

        # Kupiec test
        kup_lr, kup_p, kup_rate = kupiec_test(exceed, len(exceed), a)

        # Realized average shortfall (mean return when VaR exceeded)
        ras = float(np.mean(y[exceed])) if exceed.any() else np.nan

        rows.append({
            "model": model_name,
            "H": H,
            "alpha": a,
            "violations": int(exceed.sum()),
            "T": int(len(exceed)),
            "viol_rate": float(kup_rate),
            "kupiec_p": float(kup_p),
            "mean_ES_forecast": float(np.mean(e)),
            "realized_avg_shortfall": ras,
            "mean_FZ0": float(np.mean(L)),
        })

    return pd.DataFrame(rows)


def fz_loss_series_dynamic(df_res, model_name, alpha, H, mean_adjust=True, rolling_nu=True):
    """
    Compute FZ0 loss series for DM testing between models.

    Args:
        df_res: Results DataFrame
        model_name: Model identifier
        alpha: VaR/ES confidence level
        H: Forecast horizon
        mean_adjust: Use rolling mean adjustment
        rolling_nu: Use rolling Student-t degrees of freedom

    Returns:
        Array of FZ0 losses (preserves NaNs for alignment)
    """
    y, v, e = risk_series_var_es_dynamic(df_res, model_name, alpha, H, mean_adjust, rolling_nu)
    return fz0_loss(y, v, e, alpha)


def prepare_risk_data(
    df_res,
    models,
    H,
    use_rolling_k=True,
    rolling_nu=True,
    mean_adjust=True,
    nonoverlapping=True,
):
    """
    Prepare DataFrame for risk evaluation with k_t, nu_t, and muH columns.

    This function expects df_res to already have the forecast columns and
    optionally the risk parameters. If nonoverlapping=True, it subsamples
    to non-overlapping H-day blocks for proper risk evaluation.

    Args:
        df_res: Results DataFrame with forecasts
        models: List of model names
        H: Forecast horizon
        use_rolling_k: Whether k_t columns are expected
        rolling_nu: Whether nu_t columns are expected
        mean_adjust: Whether muH column is expected
        nonoverlapping: Subsample to non-overlapping blocks

    Returns:
        Prepared DataFrame for risk evaluation
    """
    df = df_res.copy()

    # Subsample to non-overlapping H-day blocks if requested
    if nonoverlapping and H > 1:
        # Start from first valid index, take every H-th observation
        df = df.iloc[::H].copy()

    return df


def run_risk_evaluation(
    df_res,
    models,
    H,
    alphas=None,
    use_rolling_k=True,
    rolling_nu=True,
    mean_adjust=True,
    nonoverlapping=True,
    dm_lag=20,
):
    """
    Run complete risk evaluation pipeline.

    This is the main entry point for risk evaluation, combining:
    1. FZ0 loss tables for each model
    2. Kupiec tests for VaR coverage
    3. DM tests comparing FZ0 losses between models

    Args:
        df_res: Results DataFrame with forecasts and risk parameters
        models: List of model names to evaluate
        H: Forecast horizon
        alphas: List of VaR/ES confidence levels (default: [0.01, 0.05])
        use_rolling_k: Use rolling volatility scaling
        rolling_nu: Use rolling Student-t degrees of freedom
        mean_adjust: Use rolling mean adjustment
        nonoverlapping: Use non-overlapping H-day blocks
        dm_lag: Lag for DM test HAC standard errors

    Returns:
        Dictionary with:
            - 'risk_tables': DataFrame with FZ0 metrics by model and alpha
            - 'dm_tests': DataFrame with DM test results between models
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    # Prepare data (subsample if nonoverlapping)
    df = prepare_risk_data(
        df_res, models, H, use_rolling_k, rolling_nu, mean_adjust, nonoverlapping
    )

    # Generate risk tables
    risk_dfs = []
    for model in models:
        tbl = risk_table_fz_es_dynamic(df, model, alphas, H, mean_adjust, rolling_nu)
        if not tbl.empty:
            risk_dfs.append(tbl)

    if risk_dfs:
        risk_tables = pd.concat(risk_dfs, ignore_index=True)
    else:
        risk_tables = pd.DataFrame()

    # DM tests on FZ0 loss
    dm_results = []
    alpha_dm = alphas[0] if alphas else 0.01

    for i, model_a in enumerate(models):
        for model_b in models[i + 1:]:
            try:
                loss_a = fz_loss_series_dynamic(df, model_a, alpha_dm, H, mean_adjust, rolling_nu)
                loss_b = fz_loss_series_dynamic(df, model_b, alpha_dm, H, mean_adjust, rolling_nu)

                stat, pval = _dm_test(loss_a, loss_b, L=dm_lag)

                dm_results.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "alpha": alpha_dm,
                    "dm_stat": stat,
                    "dm_pval": pval,
                })
            except Exception:
                dm_results.append({
                    "model_a": model_a,
                    "model_b": model_b,
                    "alpha": alpha_dm,
                    "dm_stat": np.nan,
                    "dm_pval": np.nan,
                })

    dm_tests = pd.DataFrame(dm_results) if dm_results else pd.DataFrame()

    return {
        "risk_tables": risk_tables,
        "dm_tests": dm_tests,
    }
