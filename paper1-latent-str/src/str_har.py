"""
Smooth Transition HAR (STR-HAR) model implementation.
"""

import numpy as np
from scipy.optimize import minimize, basinhopping


def logistic(x):
    """
    Logistic sigmoid function with numerical stability clipping.
    """
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def exponential_transition(x):
    """
    Exponential transition function.

    G(x) = 1 - exp(-x^2) for x >= 0
    G(x) = 0 for x < 0

    Symmetric around zero, faster transition than logistic.
    """
    x = np.clip(x, -50, 50)
    return 1.0 - np.exp(-(x ** 2))


def double_logistic(x, c2=0.5):
    """
    Double-logistic transition function with two thresholds.

    G(x) = logistic(x) * logistic(x - c2)

    Allows for asymmetric regime transitions.
    Extra parameter c2 controls second threshold offset.
    """
    x = np.clip(x, -50, 50)
    g1 = 1.0 / (1.0 + np.exp(-x))
    g2 = 1.0 / (1.0 + np.exp(-(x - c2)))
    return g1 * g2


# Transition function registry for robustness checks
TRANSITION_FUNCTIONS = {
    "logistic": logistic,
    "exponential": exponential_transition,
    "double_logistic": double_logistic,
}


def get_transition_function(name: str):
    """
    Get transition function by name.

    Args:
        name: Function name ('logistic', 'exponential', 'double_logistic')

    Returns:
        Transition function
    """
    if name not in TRANSITION_FUNCTIONS:
        raise ValueError(f"Unknown transition: {name}. Available: {list(TRANSITION_FUNCTIONS.keys())}")
    return TRANSITION_FUNCTIONS[name]


def fit_har_ols(train_df):
    """
    Fit standard HAR model by OLS.

    Model: y = b0 + b_d * x_d + b_w * x_w + b_m * x_m

    Args:
        train_df: DataFrame with columns 'y', 'x_d', 'x_w', 'x_m'

    Returns:
        Array of coefficients [b0, b_d, b_w, b_m]
    """
    Y = train_df["y"].values
    X = np.column_stack(
        [
            np.ones(len(train_df)),
            train_df["x_d"].values,
            train_df["x_w"].values,
            train_df["x_m"].values,
        ]
    )
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta


def har_predict(beta, df_):
    """
    Generate HAR predictions.

    Args:
        beta: HAR coefficients [b0, b_d, b_w, b_m]
        df_: DataFrame with 'x_d', 'x_w', 'x_m' columns

    Returns:
        Array of predictions
    """
    X = np.column_stack(
        [
            np.ones(len(df_)),
            df_["x_d"].values,
            df_["x_w"].values,
            df_["x_m"].values,
        ]
    )
    return X @ beta


def str2_predict_c0(params, X_std, transition_fn="logistic"):
    """
    STR-HAR two-regime prediction with c=0 (centered transition).

    Model:
        yhat = (1-G)*yL + G*yH
        G = transition_fn(gamma * q)
        yL = b0L + bdL*x_d + bwL*x_w + bmL*x_m  (low regime)
        yH = b0H + bdH*x_d + bwH*x_w + bmH*x_m  (high regime)

    Args:
        params: [b0L, bdL, bwL, bmL, b0H, bdH, bwH, bmH, gamma] for logistic/exponential
                [b0L, bdL, bwL, bmL, b0H, bdH, bwH, bmH, gamma, c2] for double_logistic
        X_std: DataFrame with standardized features ['x_d', 'x_w', 'x_m', 'q']
        transition_fn: Name of transition function or callable

    Returns:
        Tuple of (predictions, transition weights G)
    """
    # Get transition function
    if callable(transition_fn):
        trans_func = transition_fn
        trans_name = getattr(transition_fn, "__name__", "custom")
    else:
        trans_func = get_transition_function(transition_fn)
        trans_name = transition_fn

    # Extract parameters based on transition function
    if trans_name == "double_logistic" and len(params) >= 10:
        b0L, bdL, bwL, bmL, b0H, bdH, bwH, bmH, gamma, c2 = params[:10]
    else:
        b0L, bdL, bwL, bmL, b0H, bdH, bwH, bmH, gamma = params[:9]
        c2 = 0.5  # default for double_logistic

    q = X_std["q"].values

    # Compute transition weights
    if trans_name == "double_logistic":
        G = trans_func(gamma * q, c2=c2)
    else:
        G = trans_func(gamma * q)

    yL = b0L + bdL * X_std["x_d"].values + bwL * X_std["x_w"].values + bmL * X_std["x_m"].values
    yH = b0H + bdH * X_std["x_d"].values + bwH * X_std["x_w"].values + bmH * X_std["x_m"].values
    yhat = (1.0 - G) * yL + G * yH
    return yhat, G


def sse_obj_str2(params, y_std, X_std, lam=1e-3, transition_fn="logistic"):
    """
    SSE objective with L2 regularization on gamma.

    Args:
        params: STR-HAR parameters
        y_std: Standardized target
        X_std: Standardized features
        lam: Regularization strength for gamma
        transition_fn: Transition function name

    Returns:
        Regularized SSE
    """
    yhat, _ = str2_predict_c0(params, X_std, transition_fn=transition_fn)
    resid = y_std - yhat
    # Gamma is always second-to-last or last param (index -1 for 9 params, -2 for 10 params)
    gamma_idx = 8  # gamma is always at index 8
    return float(np.sum(resid * resid) + lam * (params[gamma_idx] ** 2))


def fit_str2_window_robust(
    train_df,
    q_col="q",
    use_basinhopping=True,
    bh_niter=5,
    n_starts=3,
    gamma_max=12.0,
    gamma_lam=1e-3,
    transition_fn="logistic",
):
    """
    Fit STR-HAR model with robust multi-start optimization.

    Uses combination of multi-start L-BFGS-B and optional basin-hopping
    global optimization for robustness.

    Args:
        train_df: Training DataFrame with 'y', 'x_d', 'x_w', 'x_m', and q_col
        q_col: Name of transition variable column
        use_basinhopping: Use basin-hopping as fallback
        bh_niter: Basin-hopping iterations
        n_starts: Number of random starts for local optimization
        gamma_max: Upper bound on gamma (steepness)
        gamma_lam: Regularization strength for gamma
        transition_fn: Transition function name ('logistic', 'exponential', 'double_logistic')

    Returns:
        Tuple of (best_params, mean_vector, std_vector)
    """
    cols = ["y", "x_d", "x_w", "x_m", q_col]
    tmp = train_df[cols].copy()
    mu = tmp.mean()
    sd = tmp.std().replace(0, 1.0)
    tmp_std = (tmp - mu) / sd

    y_std = tmp_std["y"].values
    X_std = tmp_std.rename(columns={q_col: "q"})[["x_d", "x_w", "x_m", "q"]]

    # Parameter initialization depends on transition function
    if transition_fn == "double_logistic":
        # Extra parameter c2 for second threshold
        init0 = np.array([0.0, 0.3, 0.3, 0.3, 0.0, 0.3, 0.3, 0.3, 2.0, 0.5])
        bounds = [
            (-10, 10),
            (-5, 5),
            (-5, 5),
            (-5, 5),
            (-10, 10),
            (-5, 5),
            (-5, 5),
            (-5, 5),
            (1e-3, gamma_max),
            (-2.0, 2.0),  # c2 bounds
        ]
    else:
        init0 = np.array([0.0, 0.3, 0.3, 0.3, 0.0, 0.3, 0.3, 0.3, 2.0])
        bounds = [
            (-10, 10),
            (-5, 5),
            (-5, 5),
            (-5, 5),
            (-10, 10),
            (-5, 5),
            (-5, 5),
            (-5, 5),
            (1e-3, gamma_max),
        ]

    def obj(p):
        return sse_obj_str2(p, y_std, X_std, lam=gamma_lam, transition_fn=transition_fn)

    best_x, best_f = None, np.inf

    # Multi-start local optimization
    for k in range(n_starts):
        if k == 0:
            x0 = init0.copy()
        else:
            x0 = init0 + np.random.normal(scale=0.25, size=init0.shape)
            x0[8] = np.clip(x0[8], 0.5, 8.0)  # gamma at index 8
            if transition_fn == "double_logistic":
                x0[9] = np.clip(x0[9], -1.5, 1.5)  # c2

        res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 1500})
        if res.success and np.isfinite(res.fun) and (res.fun < best_f):
            best_x, best_f = res.x, res.fun

    # Basin-hopping fallback
    if (best_x is None) and use_basinhopping:
        minimizer_kwargs = dict(
            method="L-BFGS-B", bounds=bounds, options={"maxiter": 1500}
        )

        class Stepper:
            def __init__(self, stepsize=0.6):
                self.stepsize = stepsize

            def __call__(self, x):
                x_new = x + np.random.normal(scale=self.stepsize, size=x.shape)
                for j, (lo, hi) in enumerate(bounds):
                    x_new[j] = np.clip(x_new[j], lo, hi)
                return x_new

        bh = basinhopping(
            obj,
            init0,
            niter=bh_niter,
            minimizer_kwargs=minimizer_kwargs,
            take_step=Stepper(),
            disp=False,
        )
        if np.isfinite(bh.fun):
            best_x = bh.x

    if best_x is None:
        best_x = init0.copy()

    return best_x, mu, sd


def str2_forecast_one(params, mu, sd, test_df, q_col="q", transition_fn="logistic"):
    """
    Generate single-step STR-HAR forecast.

    Args:
        params: STR-HAR parameters
        mu: Training mean vector
        sd: Training std vector
        test_df: Test DataFrame (single row)
        q_col: Transition variable column name
        transition_fn: Transition function name

    Returns:
        Tuple of (forecast, transition weight G)
    """
    tmp = test_df[["y", "x_d", "x_w", "x_m", q_col]].copy()
    tmp_std = (tmp - mu) / sd
    X_std = tmp_std.rename(columns={q_col: "q"})[["x_d", "x_w", "x_m", "q"]]
    yhat_std, G = str2_predict_c0(params, X_std, transition_fn=transition_fn)
    yhat = yhat_std * sd["y"] + mu["y"]
    return float(yhat[0]), float(G[0])


def str2_in_sample_yhat(params, mu, sd, train_df, q_col, transition_fn="logistic"):
    """
    Generate in-sample STR-HAR fitted values.

    Args:
        params: STR-HAR parameters
        mu: Training mean vector
        sd: Training std vector
        train_df: Training DataFrame
        q_col: Transition variable column name
        transition_fn: Transition function name

    Returns:
        Array of in-sample predictions
    """
    tmp = train_df[["y", "x_d", "x_w", "x_m", q_col]].copy()
    tmp_std = (tmp - mu) / sd
    X_std = tmp_std.rename(columns={q_col: "q"})[["x_d", "x_w", "x_m", "q"]]
    yhat_std, _ = str2_predict_c0(params, X_std, transition_fn=transition_fn)
    yhat = yhat_std * sd["y"] + mu["y"]
    return yhat
