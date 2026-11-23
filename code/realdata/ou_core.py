"""
Shared functions for the empirical analyses (stability-inference-OU).

The estimator, the moving-block bootstrap and the standardization are the same as in estimate_realdata_MOU.py,
which cannot be imported because it parses its arguments at module level.
Added here: data loaders, inflation factors (kappa_W, kappa_cW, kappa_r), the theoretical sigma_r/r,
innovations, the soft-mode projection and an FFT-based Ljung-Box test (no extra packages needed).
"""

import os

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import logm, solve_discrete_lyapunov
from scipy.stats import chi2, kurtosis, skew, jarque_bera

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

REV_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(REV_DIR))   # stability-inference-OU/
REAL_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results", "realdata")
FIG_DIR = os.path.join(ROOT, "figures", "other", "realdata")

FIN_FILE = os.path.join(REAL_DIR, "FINANCIAL_data", "Combined_processed", "Combined_processed", "returns_wide_1s.csv")
EEG_DIR = os.path.join(REAL_DIR, "EEG_data")


######################################################################################################################
# ESTIMATOR (identical to estimate_realdata_MOU.py)
######################################################################################################################

def estimate_bayes_I(X, delta_t, lag=1):
    """X: (N, T). Returns A, Q, Sigma_dt, Sigma_inf, B."""

    X = X.astype(np.float64)
    N, T = X.shape

    T1 = (X[:, lag:] @ X[:, lag:].T) / (T - lag)
    T2 = (X[:, lag:] @ X[:, :-lag].T) / (T - lag)
    T3 = (X[:, :-lag] @ X[:, :-lag].T) / (T - lag)

    T3_inv = np.linalg.inv(T3)

    Q_est = np.real(T2 @ T3_inv)

    Sigma_dt_est = T1 - T2 @ T3_inv @ T2.T

    A_est = np.real(-logm(Q_est) / (lag * delta_t))

    Sigma_inf_est = solve_discrete_lyapunov(Q_est, Sigma_dt_est)

    B_est = 0.5 * (A_est @ Sigma_inf_est + Sigma_inf_est @ A_est.T)

    return A_est, Q_est, Sigma_dt_est, Sigma_inf_est, B_est


def r_hat(X, delta_t, lag=1):
    """Distance to instability min Re lambda(A_hat) (in 1/time units of delta_t) and Q_hat.
    Same A_hat as estimate_bayes_I, without the Lyapunov solve (not needed for r)."""

    X = X.astype(np.float64)
    T = X.shape[1]
    T2 = (X[:, lag:] @ X[:, :-lag].T) / (T - lag)
    T3 = (X[:, :-lag] @ X[:, :-lag].T) / (T - lag)
    Q = np.real(T2 @ np.linalg.inv(T3))
    A = np.real(-logm(Q, disp=False)[0] / (lag * delta_t))

    return np.min(np.real(np.linalg.eigvals(A))), Q


######################################################################################################################
# MOVING-BLOCK BOOTSTRAP (identical to estimate_realdata_MOU.py)
######################################################################################################################

def estimate_tau(Q_est):

    eigvals_Q = np.abs(np.linalg.eigvals(Q_est))
    eigvals_Q = eigvals_Q[np.isfinite(eigvals_Q)]
    eigvals_Q = eigvals_Q[(eigvals_Q > 0) & (eigvals_Q < 1)]

    if len(eigvals_Q) == 0:
        return np.nan

    x = -np.log(np.max(eigvals_Q))

    return (1 + np.exp(-2 * x)) / (1 - np.exp(-2 * x))


def moving_block_bootstrap(X, delta_t, lag, n_boot, c0, Q_est, seed):

    rng = np.random.default_rng(seed)

    N, T = X.shape

    if not np.all(np.isfinite(Q_est)):
        return np.full(n_boot, np.nan)

    tau = estimate_tau(Q_est)

    L = lag + 1 if not np.isfinite(tau) else max(int(c0 * tau), lag + 1)
    L = max(min(L, T // 2), lag + 1)

    n_blocks = T - L + 1
    n_needed = int(np.ceil(T / L))

    r_boot = np.full(n_boot, np.nan)

    for b in range(n_boot):

        starts = rng.integers(0, n_blocks, size=n_needed)

        X_boot = np.concatenate([X[:, s:s + L] for s in starts], axis=1)[:, :T]

        try:
            r_boot[b] = r_hat(X_boot, delta_t, lag)[0]
        except Exception:
            pass

    return r_boot


def residual_bootstrap(X, delta_t, Q, n_boot, seed):
    """Residual bootstrap of r_hat for short records: fitted innovations eps_t = x_{t+1} - Q x_t (centred) are
    resampled one step at a time with replacement, and a new trajectory x_{t+1} = Q x_t + eps* is generated from the
    observed x_0 with the same length. Keeps the empirical (non-Gaussian) innovations; assumes they are independent
    in time."""

    rng = np.random.default_rng(seed)
    N, T = X.shape
    eps = X[:, 1:] - Q @ X[:, :-1]
    eps = eps - eps.mean(axis=1, keepdims=True)

    r_boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, T - 1, size=T - 1)
        Xb = np.empty((N, T))
        Xb[:, 0] = X[:, 0]
        for t in range(T - 1):
            Xb[:, t + 1] = Q @ Xb[:, t] + eps[:, idx[t]]
        try:
            r_boot[b] = r_hat(Xb, delta_t)[0]
        except Exception:
            pass

    return r_boot


######################################################################################################################
# PREPROCESSING AND LOADERS
######################################################################################################################

def standardize(X, center="per-variable"):
    """X: (T, N). Global or per-variable centering, then division by one global std."""

    if center == "global":
        X = X - np.mean(X)
    else:
        X = X - np.mean(X, axis=0, keepdims=True)

    std = np.std(X)

    return X / (std if std > 0 else 1.0)


def load_fin_returns(n_assets=47, cache=True):
    """1-second log-returns (T x N DataFrame, index = timestamps). Cached as .npy after the first read of the CSV."""

    cache_dir = os.path.join(RESULTS_DIR, "cache")
    f_val = os.path.join(cache_dir, "fin_returns_1s.npy")
    f_idx = os.path.join(cache_dir, "fin_index_1s.npy")
    f_col = os.path.join(cache_dir, "fin_columns_1s.npy")

    if cache and os.path.exists(f_val):
        values = np.load(f_val)
        index = pd.to_datetime(np.load(f_idx, allow_pickle=True))
        columns = np.load(f_col, allow_pickle=True)
    else:
        df = pd.read_csv(FIN_FILE, index_col=0, parse_dates=True)
        values, index, columns = df.values.astype(np.float64), df.index, np.asarray(df.columns)
        if cache:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(f_val, values)
            np.save(f_idx, np.asarray(index.astype(str)), allow_pickle=True)
            np.save(f_col, columns, allow_pickle=True)

    return pd.DataFrame(values[:, :n_assets], index=index, columns=columns[:n_assets])


def fin_days(df):
    """List of (date, T_day x N array) for each trading day."""
    return [(day, g.values) for day, g in df.groupby(df.index.date)]


def load_plankton(n_species=47, min_presence=0.90):
    """Log-abundances (T x N) of the most abundant OTUs, as in estimate_realdata_MOU.py (dt = 1/3 day)."""

    path = os.path.join(REAL_DIR, "PLANKTON_data", "41467_2017_2571_MOESM6_ESM.xlsx")
    df = pd.read_excel(path, index_col=0).drop(columns=["ConsensusLineage"])
    df = df[(df > 0).sum(axis=1) >= int(df.shape[1] * min_presence)]
    data = df.loc[df.mean(axis=1).nlargest(n_species).index].values.T.astype(np.float64)

    for j in range(data.shape[1]):
        col = data[:, j]
        col[col == 0] = col[col > 0].min() / 2

    return np.log(data)


def load_eeg(name="Sz1.mat"):
    """Raw EEG (T x N) at 512 Hz."""
    return loadmat(os.path.join(EEG_DIR, name))["EEG"].astype(np.float64)


######################################################################################################################
# INFLATION FACTORS AND THEORETICAL sigma_r / r
######################################################################################################################

def tau_of_q(q):
    """Integrated autocorrelation time of the sample variance of an AR(1) with coefficient q (|q| < 1)."""
    q2 = np.clip(np.abs(q) ** 2, 0, 1 - 1e-15)
    return (1 + q2) / (1 - q2)


def kappa_W(M, N):
    return (M - 1) / (M - N - 2) if M > N + 2 else np.inf


def kappa_cW(M, N, q_eigs):
    """Paper Eq. (kcW): M -> M_eff = M / <tau>_w, <tau>_w = sum tau^2 / sum tau."""
    tau = tau_of_q(q_eigs)
    M_eff = M / (np.sum(tau ** 2) / np.sum(tau))
    return kappa_W(M_eff, N)


def kappa_r(n, q_eigs):
    """Soft-mode inflation factor, first order as in the paper (Methods, Eq. kappa_r):
    kappa_r = 1 + S / n = 1 + 1 / q_eff,  S = 2 tau_r + sum_{b != r} tau_rb,  tau_rb = (1 + q_r q_b) / (1 - q_r q_b),
    n = number of transitions (M - 1 for one window of M samples).
    For an estimated (non-symmetric) Q we use the moduli |q| = e^{-Re lambda dt}; the soft mode is the largest |q| < 1."""

    q = np.abs(np.asarray(q_eigs))
    q = q[np.isfinite(q)]
    q = np.clip(q, 0, 1 - 1e-12)

    i_r = np.argmax(q)
    q_r = q[i_r]
    q_b = np.delete(q, i_r)

    S = 2 * tau_of_q(q_r) + np.sum((1 + q_r * q_b) / (1 - q_r * q_b))

    return 1 + S / n


def eig_A_re(Q, delta_t, lag=1):
    """Real parts of the eigenvalues of A_hat = -log(Q_hat) / dt (principal log, same A as r_hat)."""
    A = np.real(-logm(Q, disp=False)[0] / (lag * delta_t))
    return np.real(np.linalg.eigvals(A))


def kappa_cW_si(n, N, re_lam, delta_t):
    """SI Proof (B), Eqs. (tau_a), (Meff_w), (kcW_global), (qeff_derived), on the estimated spectrum:
    |q_a| = exp(-Re lambda_a dt), tau_a = (1 + |q_a|^2) / (1 - |q_a|^2), tau_bar = arithmetic mean over ALL modes,
    n_eff = n / tau_bar, q_eff = n_eff / N, kappa_cW = n_eff / (n_eff - N - 1) if n_eff > N + 1 else NaN.
    If some Re lambda_a <= 0, tau_a diverges: everything is NaN and 'unstable' is True."""
    re_lam = np.asarray(re_lam, dtype=float)
    if np.any(~np.isfinite(re_lam)) or np.any(re_lam <= 0):
        return {"tau_bar": np.nan, "n_eff": np.nan, "q_eff": np.nan, "kappa_cW": np.nan, "unstable": True}
    q2 = np.exp(-2 * re_lam * delta_t)
    tau_bar = np.mean((1 + q2) / (1 - q2))
    n_eff = n / tau_bar
    kap = n_eff / (n_eff - N - 1) if n_eff > N + 1 else np.nan
    return {"tau_bar": tau_bar, "n_eff": n_eff, "q_eff": n_eff / N, "kappa_cW": kap, "unstable": False}


def rel_err_theory_nan(r, n, delta_t, kappa):
    """As rel_err_theory, but NaN (point not plotted) when kappa is undefined or r <= 0."""
    if not np.isfinite(kappa) or not np.isfinite(r) or r <= 0:
        return np.nan
    return np.sqrt(kappa * np.expm1(2 * r * delta_t) / (n * delta_t ** 2)) / r


MU_TW = -1.21


def c_from_r(r_mod, N, mu_TW=MU_TW):
    """c such that r_bar(c, N) = r_mod, with r_bar of code/synthetic/mou_core (GOE spectrum centred at 1, Tracy-Widom
    correction): r_bar = 1 - sqrt2 c - (c / sqrt2) mu_TW N^{-2/3}, linear in c."""
    return (1 - r_mod) / (np.sqrt(2) + mu_TW * N ** (-2 / 3) / np.sqrt(2))


def rel_err_theory(r, n, delta_t, kappa):
    """sigma_r / r from Var(r_hat) = kappa (e^{2 r dt} - 1) / (n dt^2), n = number of transitions (paper Eq. 10)."""
    if not np.isfinite(kappa) or r <= 0:
        return np.inf
    return np.sqrt(kappa * np.expm1(2 * r * delta_t) / (n * delta_t ** 2)) / r


######################################################################################################################
# INNOVATIONS, SOFT MODE, LJUNG-BOX
######################################################################################################################

def innovations(X, Q, lag=1):
    """X: (N, T). Fitted innovations eps_k = X_{k+lag} - Q X_k, shape (N, T - lag)."""
    return X[:, lag:] - Q @ X[:, :-lag]


def soft_left_vector(Q):
    """Left eigenvector of Q for the soft mode (largest real eigenvalue below 1 in modulus, else largest |q|)."""

    w, V = np.linalg.eig(Q.T)
    real = np.abs(np.imag(w)) < 1e-10
    cand = np.where(real & (np.abs(w) < 1))[0]
    i = cand[np.argmax(np.real(w[cand]))] if len(cand) else np.argmax(np.abs(w))
    ell = np.real(V[:, i])

    return ell / np.linalg.norm(ell), w[i]


def acf_fft(x, max_lag):
    """Sample autocorrelation of each row of x (n_series, T) up to max_lag, via FFT."""

    x = np.atleast_2d(x)
    x = x - x.mean(axis=1, keepdims=True)
    T = x.shape[1]
    n_fft = 1 << int(np.ceil(np.log2(2 * T - 1)))
    f = np.fft.rfft(x, n_fft, axis=1)
    acov = np.fft.irfft(f * np.conj(f), n_fft, axis=1)[:, :max_lag + 1]

    return acov[:, 1:] / acov[:, [0]]


def ljung_box(x, lags):
    """Ljung-Box Q statistic and p-value for each row of x (n_series, T), at each cumulative lag H in `lags`.
    Q_H = T (T + 2) sum_{h=1}^{H} rho_h^2 / (T - h). Returns p-values of shape (n_series, len(lags))."""

    x = np.atleast_2d(x)
    T = x.shape[1]
    lags = [int(h) for h in lags]
    rho = acf_fft(x, max(lags))
    terms = rho ** 2 / (T - np.arange(1, max(lags) + 1))
    cum = T * (T + 2) * np.cumsum(terms, axis=1)

    return np.stack([chi2.sf(cum[:, H - 1], H) for H in lags], axis=1)


def bh_reject(pvals, alpha=0.05):
    """Benjamini-Hochberg rejections over a 1D array of p-values."""

    p = np.asarray(pvals)
    n = len(p)
    order = np.argsort(p)
    thresh = alpha * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    k = np.max(np.where(passed)[0]) + 1 if np.any(passed) else 0
    rej = np.zeros(n, dtype=bool)
    rej[order[:k]] = True

    return rej


def gaussianity(x):
    """Skewness, excess kurtosis and Jarque-Bera p-value of a 1D series."""
    return {"skew": float(skew(x)), "ex_kurt": float(kurtosis(x)), "jb_p": float(jarque_bera(x).pvalue)}
