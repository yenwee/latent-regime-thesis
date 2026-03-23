"""
GARCH family baseline models for volatility forecasting.

All inner variance/filter loops are JIT-compiled with Numba for performance.
"""

import numpy as np
from numba import njit
from scipy.optimize import minimize, basinhopping


# ============================================================
# Numba-compiled helper functions
# ============================================================

@njit(cache=True)
def _logistic(x):
    """Logistic sigmoid with numerical stability."""
    if x < -50.0:
        return 0.0
    if x > 50.0:
        return 1.0
    return 1.0 / (1.0 + np.exp(-x))


@njit(cache=True)
def _gammaln(x):
    """Log-gamma via Lanczos approximation (g=7, n=9)."""
    cof = np.array([
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ])
    if x < 0.5:
        return np.log(np.pi / np.sin(np.pi * x)) - _gammaln(1.0 - x)
    x = x - 1.0
    a = cof[0]
    t = x + 7.5
    for j in range(1, 9):
        a += cof[j] / (x + j)
    return 0.5 * np.log(2.0 * np.pi) + (x + 0.5) * np.log(t) - t + np.log(a)


@njit(cache=True)
def _student_t_pdf(x, nu, scale):
    """Student-t PDF with location=0 and given scale."""
    z = x / scale
    log_norm = _gammaln(0.5 * (nu + 1.0)) - _gammaln(0.5 * nu) - 0.5 * np.log(nu * np.pi)
    log_pdf = log_norm - 0.5 * (nu + 1.0) * np.log(1.0 + z * z / nu) - np.log(scale)
    return np.exp(log_pdf)


@njit(cache=True)
def _student_t_logpdf_array(x, nu, scale):
    """Vectorized Student-t log-PDF over arrays x and scale."""
    T = len(x)
    out = np.empty(T)
    log_norm = _gammaln(0.5 * (nu + 1.0)) - _gammaln(0.5 * nu) - 0.5 * np.log(nu * np.pi)
    half_nup1 = 0.5 * (nu + 1.0)
    for i in range(T):
        z = x[i] / scale[i]
        out[i] = log_norm - half_nup1 * np.log(1.0 + z * z / nu) - np.log(scale[i])
    return out


# --- GARCH(1,1) kernels ---

@njit(cache=True)
def _garch_variance(r, omega, alpha, beta, h0):
    """Compute full conditional variance array."""
    T = len(r)
    h = np.empty(T)
    h[0] = h0
    for t in range(1, T):
        h[t] = omega + alpha * r[t - 1] * r[t - 1] + beta * h[t - 1]
    return h


@njit(cache=True)
def _garch_filter_last(r, omega, alpha, beta, h0):
    """Filter to get final variance only (for predict)."""
    h = h0
    for t in range(1, len(r)):
        h = omega + alpha * r[t - 1] * r[t - 1] + beta * h
    return h


@njit(cache=True)
def _garch_nll(r, omega, alpha, beta, nu, h0):
    """Full GARCH(1,1)-t NLL in a single compiled function."""
    T = len(r)
    h = np.empty(T)
    h[0] = h0
    for t in range(1, T):
        h[t] = omega + alpha * r[t - 1] * r[t - 1] + beta * h[t - 1]

    scale = np.empty(T)
    factor = (nu - 2.0) / nu
    for i in range(T):
        scale[i] = np.sqrt(h[i] * factor)

    logpdf = _student_t_logpdf_array(r, nu, scale)
    nll = 0.0
    for i in range(T):
        nll -= logpdf[i]
    return nll


# --- EGARCH(1,1) kernels ---

@njit(cache=True)
def _egarch_logvar(r, omega, alpha, gamma, beta, log_h0):
    """Compute full log-variance array for EGARCH."""
    T = len(r)
    log_h = np.empty(T)
    log_h[0] = log_h0
    Ez = np.sqrt(2.0 / np.pi)
    for t in range(1, T):
        lh_prev = log_h[t - 1]
        if lh_prev < -50.0:
            lh_prev = -50.0
        elif lh_prev > 50.0:
            lh_prev = 50.0
        sigma_prev = np.sqrt(np.exp(lh_prev))
        z_prev = r[t - 1] / (sigma_prev + 1e-9)
        lh_next = omega + beta * log_h[t - 1] + alpha * (np.abs(z_prev) - Ez) + gamma * z_prev
        if lh_next < -50.0:
            lh_next = -50.0
        elif lh_next > 50.0:
            lh_next = 50.0
        log_h[t] = lh_next
    return log_h


@njit(cache=True)
def _egarch_filter_last(r, omega, alpha, gamma, beta, log_h0):
    """Filter to get final log-variance only (for predict)."""
    log_h = log_h0
    Ez = np.sqrt(2.0 / np.pi)
    for t in range(1, len(r)):
        lh_clipped = log_h
        if lh_clipped < -50.0:
            lh_clipped = -50.0
        elif lh_clipped > 50.0:
            lh_clipped = 50.0
        sigma_prev = np.sqrt(np.exp(lh_clipped))
        z_prev = r[t - 1] / (sigma_prev + 1e-9)
        lh_next = omega + beta * log_h + alpha * (np.abs(z_prev) - Ez) + gamma * z_prev
        if lh_next < -50.0:
            lh_next = -50.0
        elif lh_next > 50.0:
            lh_next = 50.0
        log_h = lh_next
    return log_h


@njit(cache=True)
def _egarch_nll(r, omega, alpha, gamma, beta, nu, log_h0):
    """Full EGARCH(1,1)-t NLL in a single compiled function."""
    T = len(r)
    log_h = _egarch_logvar(r, omega, alpha, gamma, beta, log_h0)

    scale = np.empty(T)
    factor = (nu - 2.0) / nu
    for i in range(T):
        lh = log_h[i]
        if lh < -50.0:
            lh = -50.0
        elif lh > 50.0:
            lh = 50.0
        scale[i] = np.sqrt(np.exp(lh) * factor)

    logpdf = _student_t_logpdf_array(r, nu, scale)
    nll = 0.0
    for i in range(T):
        nll -= logpdf[i]
    return nll


@njit(cache=True)
def _egarch_simulate(log_h_next, omega, alpha, gamma, beta, nu, H, seed):
    """Monte Carlo EGARCH H-step forecast via compiled simulation."""
    np.random.seed(seed)
    Ez = np.sqrt(2.0 / np.pi)
    n_paths = 100
    path_means = np.empty(n_paths)

    for i in range(n_paths):
        lh = log_h_next
        path_sum = 0.0
        for k in range(H):
            z_norm = np.random.standard_normal()
            chi2_val = np.random.chisquare(nu)
            raw_t = z_norm / np.sqrt(chi2_val / nu)
            z = raw_t / np.sqrt(nu / (nu - 2.0))

            lh_c = lh
            if lh_c < -50.0:
                lh_c = -50.0
            elif lh_c > 50.0:
                lh_c = 50.0
            path_sum += np.exp(lh_c)

            lh_new = omega + beta * lh + alpha * (np.abs(z) - Ez) + gamma * z
            if lh_new < -50.0:
                lh = -50.0
            elif lh_new > 50.0:
                lh = 50.0
            else:
                lh = lh_new

        path_means[i] = path_sum / H

    return np.log(np.mean(path_means) + 1e-9)


# --- MS-GARCH kernels ---

@njit(cache=True)
def _msgarch_nll(r, theta, var_r):
    """Full MS-GARCH(1,1)-t NLL with Hamilton filter."""
    T = len(r)
    p00 = _logistic(theta[0])
    p11 = _logistic(theta[1])

    w1 = np.exp(theta[2])
    a1 = _logistic(theta[3])
    b1 = _logistic(theta[4])
    w2 = np.exp(theta[5])
    a2 = _logistic(theta[6])
    b2 = _logistic(theta[7])
    nu = 2.05 + np.exp(theta[8])

    if (a1 + b1 > 0.999) or (a2 + b2 > 0.999):
        return 1e10

    h1 = var_r
    h2 = var_r
    lik = 0.0

    denom = (2.0 - p00 - p11) + 1e-9
    pi0 = (1.0 - p11) / denom
    xi_0 = pi0
    xi_1 = 1.0 - pi0

    P00 = p00
    P01 = 1.0 - p11
    P10 = 1.0 - p00
    P11_val = p11

    for t in range(1, T):
        src = r[t - 1] * r[t - 1]
        h1 = w1 + a1 * src + b1 * h1
        h2 = w2 + a2 * src + b2 * h2

        s1 = np.sqrt(h1 * (nu - 2.0) / nu)
        s2 = np.sqrt(h2 * (nu - 2.0) / nu)
        d1 = _student_t_pdf(r[t], nu, s1)
        d2 = _student_t_pdf(r[t], nu, s2)

        xp_0 = xi_0 * P00 + xi_1 * P10
        xp_1 = xi_0 * P01 + xi_1 * P11_val

        f = xp_0 * d1 + xp_1 * d2
        lik += np.log(f + 1e-20)

        xi_0 = (xp_0 * d1) / (f + 1e-20)
        xi_1 = (xp_1 * d2) / (f + 1e-20)

    return -lik


@njit(cache=True)
def _msgarch_filter(r, p00, p11, w1, a1, b1, w2, a2, b2, nu, var_r):
    """Hamilton filter through history, return (h1, h2, xi_0, xi_1)."""
    T = len(r)
    h1 = var_r
    h2 = var_r

    denom = (2.0 - p00 - p11) + 1e-9
    pi0 = (1.0 - p11) / denom
    xi_0 = pi0
    xi_1 = 1.0 - pi0

    P00 = p00
    P01 = 1.0 - p11
    P10 = 1.0 - p00
    P11_val = p11

    for t in range(1, T):
        src = r[t - 1] * r[t - 1]
        h1 = w1 + a1 * src + b1 * h1
        h2 = w2 + a2 * src + b2 * h2

        s1 = np.sqrt(h1 * (nu - 2.0) / nu)
        s2 = np.sqrt(h2 * (nu - 2.0) / nu)
        d1 = _student_t_pdf(r[t], nu, s1)
        d2 = _student_t_pdf(r[t], nu, s2)

        xp_0 = xi_0 * P00 + xi_1 * P10
        xp_1 = xi_0 * P01 + xi_1 * P11_val

        f = xp_0 * d1 + xp_1 * d2
        xi_0 = (xp_0 * d1) / (f + 1e-20)
        xi_1 = (xp_1 * d2) / (f + 1e-20)

    return h1, h2, xi_0, xi_1


# ============================================================
# Public API (unchanged interfaces)
# ============================================================

def logistic(x):
    """Logistic sigmoid with numerical stability."""
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


class Garch11_t:
    """
    GARCH(1,1) model with Student-t innovations.

    Variance equation:
        h_t = omega + alpha * r_{t-1}^2 + beta * h_{t-1}
    """

    def __init__(self, use_basinhopping=True):
        self.params = None  # [omega, alpha, beta, nu_log]
        self.use_basinhopping = use_basinhopping

    def fit(self, r):
        """
        Fit GARCH(1,1)-t by maximum likelihood.

        Args:
            r: Array of returns (already demeaned)

        Returns:
            self (fitted model)
        """
        r = np.asarray(r, dtype=np.float64)
        var_r = np.var(r)

        def nll(theta):
            omega, alpha, beta, nu_log = theta
            nu_log_clipped = np.clip(nu_log, -10, 10)
            nu = 2.05 + np.exp(nu_log_clipped)
            if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta >= 0.999) or nu > 1000:
                return 1e10
            return _garch_nll(r, omega, alpha, beta, nu, var_r)

        init = [var_r * 0.05, 0.05, 0.90, np.log(6.0)]
        bounds = [(1e-6, None), (0, 1), (0, 1), (-5, 8)]

        if self.use_basinhopping:
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "options": {"maxiter": 100}}
            try:
                res = basinhopping(nll, init, minimizer_kwargs=minimizer_kwargs, niter=3, seed=42)
                self.params = res.x
            except Exception:
                res = minimize(nll, init, method="L-BFGS-B", bounds=bounds)
                self.params = res.x if res.success else init
        else:
            res = minimize(nll, init, method="L-BFGS-B", bounds=bounds)
            self.params = res.x if res.success else init
        return self

    def predict(self, r_history, H):
        """
        Generate H-step ahead forecast.

        Args:
            r_history: Historical returns for conditioning
            H: Forecast horizon

        Returns:
            log(mean variance over next H days)
        """
        r_history = np.asarray(r_history, dtype=np.float64)
        omega, alpha, beta, nu_log = self.params
        nu = 2.05 + np.exp(nu_log)

        h_last = _garch_filter_last(r_history, omega, alpha, beta, np.var(r_history))

        h_next = omega + alpha * r_history[-1] ** 2 + beta * h_last

        pers = alpha + beta
        preds = np.empty(H)
        curr = h_next
        for k in range(H):
            preds[k] = curr
            curr = omega + pers * curr

        return np.log(np.mean(preds) + 1e-9)


class Egarch11_t:
    """
    EGARCH(1,1) model with Student-t innovations (Nelson 1991).

    Log-variance equation:
        log(h_t) = omega + beta * log(h_{t-1}) + alpha * (|z_{t-1}| - E|z|) + gamma * z_{t-1}

    The leverage effect is captured by gamma (typically negative).
    """

    def __init__(self, use_basinhopping=True):
        self.params = None
        self.use_basinhopping = use_basinhopping

    def fit(self, r):
        """
        Fit EGARCH(1,1)-t by maximum likelihood.

        Args:
            r: Array of returns

        Returns:
            self (fitted model)
        """
        r = np.asarray(r, dtype=np.float64)
        log_h0 = np.log(np.var(r) + 1e-9)

        def nll(theta):
            omega, alpha, gamma, beta, nu_log = theta
            nu_log_clipped = np.clip(nu_log, -10, 10)
            nu = 2.05 + np.exp(nu_log_clipped)
            if abs(beta) >= 0.999 or nu > 1000:
                return 1e10
            return _egarch_nll(r, omega, alpha, gamma, beta, nu, log_h0)

        init = [-0.1, 0.1, -0.05, 0.95, np.log(6.0)]
        bounds = [(None, None), (None, None), (None, None), (-0.999, 0.999), (-5, 8)]

        if self.use_basinhopping:
            minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "options": {"maxiter": 100}}
            try:
                res = basinhopping(nll, init, minimizer_kwargs=minimizer_kwargs, niter=3, seed=42)
                self.params = res.x
            except Exception:
                res = minimize(nll, init, method="L-BFGS-B", bounds=bounds)
                self.params = res.x if res.success else init
        else:
            res = minimize(nll, init, method="L-BFGS-B", bounds=bounds)
            self.params = res.x if res.success else init
        return self

    def predict(self, r_history, H, seed=None):
        """
        Generate H-step ahead forecast via compiled Monte Carlo simulation.

        Args:
            r_history: Historical returns for conditioning
            H: Forecast horizon
            seed: Random seed for reproducibility

        Returns:
            log(mean variance over next H days)
        """
        r_history = np.asarray(r_history, dtype=np.float64)
        omega, alpha, gamma, beta, nu_log = self.params
        nu = 2.05 + np.exp(nu_log)
        Ez = np.sqrt(2.0 / np.pi)

        log_h = _egarch_filter_last(r_history, omega, alpha, gamma, beta,
                                     np.log(np.var(r_history) + 1e-9))

        sigma_last = np.sqrt(np.exp(np.clip(log_h, -50, 50)))
        z_last = r_history[-1] / (sigma_last + 1e-9)
        log_h_next = (
            omega + beta * log_h + alpha * (np.abs(z_last) - Ez) + gamma * z_last
        )

        sim_seed = seed if seed is not None else 123
        return _egarch_simulate(log_h_next, omega, alpha, gamma, beta, nu, H, sim_seed)


class MSGarch2_t:
    """
    2-State Markov-Switching GARCH(1,1) with Student-t innovations.

    Based on Haas, Mittnik, and Paolella (2004) diagonal specification.
    Allows path-independent filtering by conditioning on regime.
    """

    def __init__(self, use_basinhopping=True):
        # [p00, p11, w1, a1, b1, w2, a2, b2, nu_log]
        self.params = None
        self.last_h = None
        self.last_prob = None
        self.use_basinhopping = use_basinhopping

    def fit(self, r):
        """
        Fit MS-GARCH(1,1)-t by maximum likelihood with Hamilton filter.

        Args:
            r: Array of returns

        Returns:
            self (fitted model)
        """
        r = np.asarray(r, dtype=np.float64)
        var_r = np.var(r)

        def nll(theta):
            theta_arr = np.asarray(theta, dtype=np.float64)
            return _msgarch_nll(r, theta_arr, var_r)

        log_w1 = np.log(var_r * 0.02 + 1e-12)
        log_w2 = np.log(var_r * 0.10 + 1e-12)

        init = [
            2.0,        # p00 -> logistic(2) ~ 0.88 (persistent)
            2.0,        # p11 -> logistic(2) ~ 0.88 (persistent)
            log_w1,     # w1 in log scale (data-driven)
            -2.5,       # a1 -> logistic(-2.5) ~ 0.08
            2.5,        # b1 -> logistic(2.5) ~ 0.92
            log_w2,     # w2 in log scale (data-driven, higher)
            -1.5,       # a2 -> logistic(-1.5) ~ 0.18
            1.5,        # b2 -> logistic(1.5) ~ 0.82
            np.log(6.0) # nu_log -> nu ~ 8
        ]

        if self.use_basinhopping:
            minimizer_kwargs = {"method": "L-BFGS-B", "options": {"maxiter": 100}}
            try:
                res = basinhopping(nll, init, minimizer_kwargs=minimizer_kwargs, niter=5, seed=42)
                self.params = res.x
            except Exception:
                res = minimize(nll, init, method="L-BFGS-B", options={"maxiter": 300})
                self.params = res.x if res.success else init
        else:
            res = minimize(nll, init, method="L-BFGS-B", options={"maxiter": 300})
            self.params = res.x if res.success else init
        return self

    def predict(self, r_history, H):
        """
        Generate H-step ahead forecast with regime averaging.

        Args:
            r_history: Historical returns for conditioning
            H: Forecast horizon

        Returns:
            log(mean variance over next H days)
        """
        r = np.asarray(r_history, dtype=np.float64)
        theta = self.params

        p00 = logistic(theta[0])
        p11 = logistic(theta[1])
        P = np.array([[p00, 1 - p11], [1 - p00, p11]])
        w1, a1, b1 = np.exp(theta[2]), logistic(theta[3]), logistic(theta[4])
        w2, a2, b2 = np.exp(theta[5]), logistic(theta[6]), logistic(theta[7])
        nu = 2.05 + np.exp(theta[8])

        var_r = np.var(r)
        h1, h2, xi_0, xi_1 = _msgarch_filter(r, p00, p11, w1, a1, b1, w2, a2, b2, nu, var_r)

        # Forecast H steps
        curr_xi = np.array([xi_0, xi_1]) @ P
        src = r[-1] ** 2
        h1_next = w1 + a1 * src + b1 * h1
        h2_next = w2 + a2 * src + b2 * h2

        preds = np.empty(H)
        c_h1 = h1_next
        c_h2 = h2_next
        c_xi = curr_xi

        for k in range(H):
            preds[k] = c_xi[0] * c_h1 + c_xi[1] * c_h2
            c_xi = c_xi @ P
            c_h1 = w1 + (a1 + b1) * c_h1
            c_h2 = w2 + (a2 + b2) * c_h2

        return np.log(np.mean(preds) + 1e-9)
