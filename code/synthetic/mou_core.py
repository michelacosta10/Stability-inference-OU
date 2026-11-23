"""
Shared functions for the synthetic multivariate OU data (Methods, Supplementary Sections 1-3).

    drift_matrix      A = I + W, W from the GOE or the real Ginibre ensemble, stable draws only
    true_parameters   Q = exp(-A dt), Sigma_inf (A S + S A^T = 2I), Sigma_dt = Sigma_inf - Q Sigma_inf Q^T
    simulate_var1     exact VAR(1) discretisation with stationary initial condition
    estimate          maximum-likelihood estimators Q_hat = T2 T3^{-1}, Sigma_dt_hat, A_hat, Sigma_inf_hat
    fisher_eigs       eigenvalues of the Fisher information per transition, I / n
    condition_number  max / min of the positive eigenvalues
    load_estimates    the files written by estimate.py for one run, stacked over c
    r_bar             Tracy-Widom corrected distance to instability r_bar(c, N) (Eq. rbar)
    semicircle_spectrum  soft mode at r_bar, the other N - 1 modes at the semicircle quantiles (phase diagrams)
    soft_mode_dof     sum_{b != r} tau_rb + 2 tau_r, so that q_eff = n / dof (Eq. kappa_r)
    bias_r            O(1/n) bias of r_hat (Nicholls-Pope + curvature of the log), plug-in on the estimates
    kappa_r_exact     Monte Carlo of kappa_r = n E[(S^{-1})_rr] from the eigenvalues (as kappa_r_mc.py)
    rel_err_r         sigma_r / r (Eq. rel_var_r)

Simulation and estimation:
  - exact propagator exp(-A dt);
  - stationary start X_0 ~ N(0, Sigma_inf);
  - Q_hat = T2 T3^{-1} for both ensembles; the symmetrisation (Q_hat + Q_hat^T)/2 only with symmetrize=True.
"""

import glob
import os

import numpy as np
from scipy.linalg import expm, logm, solve_continuous_lyapunov, solve_discrete_lyapunov


######################################################################################################################
# DRIFT MATRIX
######################################################################################################################

def drift_matrix(N, c, ensemble, rng):
    """A = I + W. GOE: off-diagonal variance c^2/(2N), diagonal c^2/N. Ginibre: i.i.d. entries of variance c^2/N.
    Unstable draws (min Re lambda <= 0) are discarded and redrawn; returns (A, n_rejected)."""

    n_rej = 0
    while True:
        G = rng.standard_normal((N, N))
        if ensemble == "GOE":
            G = (G + G.T) / 2.0
        A = np.eye(N) + G * (c / np.sqrt(N))
        lam = np.linalg.eigvalsh(A) if ensemble == "GOE" else np.linalg.eigvals(A)
        if np.min(np.real(lam)) > 0:
            return A, n_rej
        n_rej += 1


def true_parameters(A, delta_t):
    """Q = exp(-A dt), Sigma_inf from A S + S A^T = 2I (B = I), Sigma_dt = Sigma_inf - Q Sigma_inf Q^T."""

    N = A.shape[0]
    Q = expm(-delta_t * A)
    S_inf = solve_continuous_lyapunov(A, 2.0 * np.eye(N))
    S_inf = 0.5 * (S_inf + S_inf.T)
    S_dt = S_inf - Q @ S_inf @ Q.T
    S_dt = 0.5 * (S_dt + S_dt.T)

    return Q, S_inf, S_dt


######################################################################################################################
# SIMULATION
######################################################################################################################

def simulate_var1(Q, S_inf, S_dt, M, n_traj, rng):
    """X_{k+1} = Q X_k + eps_k, eps_k ~ N(0, Sigma_dt), X_0 ~ N(0, Sigma_inf). Returns (n_traj, N, M)."""

    N = Q.shape[0]
    L_inf = np.linalg.cholesky(S_inf)
    L_dt = np.linalg.cholesky(S_dt)

    X = np.empty((n_traj, N, M))
    X[:, :, 0] = rng.standard_normal((n_traj, N)) @ L_inf.T
    for k in range(1, M):
        X[:, :, k] = X[:, :, k - 1] @ Q.T + rng.standard_normal((n_traj, N)) @ L_dt.T

    return X


######################################################################################################################
# ESTIMATOR (Methods): sufficient statistics over the n = M - 1 transitions
######################################################################################################################

def estimate(X, delta_t, symmetrize=False):
    """X: (N, M). Q_hat = T2 T3^{-1}, Sigma_dt_hat = T1 - T2 T3^{-1} T2^T, A_hat = -logm(Q_hat)/dt (real part),
    Sigma_inf_hat from the discrete Lyapunov equation."""

    X = np.asarray(X, dtype=np.float64)
    n = X.shape[1] - 1
    X0, X1 = X[:, :-1], X[:, 1:]
    T1 = X1 @ X1.T / n
    T2 = X1 @ X0.T / n
    T3 = X0 @ X0.T / n

    Q = T2 @ np.linalg.inv(T3)
    if symmetrize:
        Q = 0.5 * (Q + Q.T)

    S_dt = T1 - T2 @ np.linalg.solve(T3, T2.T)
    S_dt = 0.5 * (S_dt + S_dt.T)

    A = np.real(-logm(Q) / delta_t)

    S_inf = solve_discrete_lyapunov(Q, S_dt)
    S_inf = 0.5 * (S_inf + S_inf.T)

    return A, Q, S_dt, S_inf


######################################################################################################################
# FISHER INFORMATION AND CONDITION NUMBERS
######################################################################################################################

def fisher_eigs(eig_Sinf, eig_Sdt, block="full"):
    """Eigenvalues of I / n (Methods, Supplementary Section 4.1): the QQ block Sigma_inf (x) J, J = Sigma_dt^{-1},
    and the JJ block (1/2) Sigma_dt (x) Sigma_dt. Eigenvalues of a Kronecker product are products of eigenvalues.
    block: "full" (both blocks), "QQ" or "JJ"."""

    QQ = np.outer(eig_Sinf, 1.0 / eig_Sdt).ravel()
    JJ = 0.5 * np.outer(eig_Sdt, eig_Sdt).ravel()

    return {"full": np.concatenate([JJ, QQ]), "QQ": QQ, "JJ": JJ}[block]


def condition_number(eigs):
    """Condition number: max / min over the finite positive real parts."""

    e = np.real(np.asarray(eigs)).ravel()
    e = e[np.isfinite(e) & (e > 0)]

    return e.max() / e.min() if len(e) else np.nan


######################################################################################################################
# GOE SPECTRUM AT FINITE N AND SOFT-MODE INFLATION FACTOR (Methods, Eqs. rbar, kappa_r, rel_var_r; SI Section 6.2)
######################################################################################################################

MU_TW = -1.21


def r_bar(c, N, mu_TW=MU_TW):
    """Eq. rbar: (1 - sqrt2 c) - (c / sqrt2) mu_TW N^{-2/3}."""

    return (1 - np.sqrt(2) * c) - c / np.sqrt(2) * mu_TW * N ** (-2 / 3)


def semicircle_spectrum(c, N):
    """N eigenvalues, ascending: the soft mode at r_bar(c, N), the other N - 1 at the midpoint quantiles of the
    semicircle (Eq. semicircle) of radius sqrt2 c, shifted rigidly so that its lower edge is at r_bar."""

    x = np.linspace(-1, 1, 20001)
    F = 0.5 + (x * np.sqrt(1 - x ** 2) + np.arcsin(x)) / np.pi
    xq = np.interp((np.arange(1, N) - 0.5) / (N - 1), F, x)
    r = r_bar(c, N)

    return np.concatenate([[r], r + np.sqrt(2) * c * (1 + xq)])


def soft_mode_dof(lam, delta_t):
    """sum_{b != r} tau_rb + 2 tau_r for eigenvalues lam with lam[0] = r, tau_rb = (1 + q_r q_b) / (1 - q_r q_b),
    q_a = exp(-lam_a dt): q_eff = n / dof and kappa_r^(1) = 1 + 1 / q_eff (Eq. kappa_r)."""

    qa = np.exp(-np.asarray(lam) * delta_t)
    tau_rb = (1 + qa[0] * qa[1:]) / (1 - qa[0] * qa[1:])
    tau_r = (1 + qa[0] ** 2) / (1 - qa[0] ** 2)

    return tau_rb.sum() + 2 * tau_r


def tau_bar(lam, delta_t):
    """Arithmetic mean over all modes of tau_a = (1 + q_a^2) / (1 - q_a^2), q_a = exp(-lam_a dt) (Methods, SI Proof B)."""

    qa2 = np.exp(-2 * np.asarray(lam, dtype=float) * delta_t)
    return np.mean((1 + qa2) / (1 - qa2))


def kappa_cW(lam, n, delta_t):
    """kappa_cW = (M_eff - 1) / (M_eff - N - 2) = n_eff / (n_eff - N - 1), M_eff = n / tau_bar + 1, and
    q_eff = n_eff / N = q / tau_bar. kappa_cW is NaN where n_eff <= N + 1 (undefined). n = number of transitions."""

    N = len(lam)
    n_eff = n / tau_bar(lam, delta_t)
    with np.errstate(divide="ignore", invalid="ignore"):
        kap = np.where(n_eff > N + 1, n_eff / (n_eff - N - 1), np.nan)
    return kap, n_eff / N


def bias_r(q_r, mu, n, delta_t):
    """Bias of r_hat = -log(q_hat_r) / dt to order 1/n (SI Section bias), evaluated on the estimates:
        b_q = -(1/n) [q_r + (1 - q_r^2) sum_b mu_b / (1 - q_r mu_b)]       (Nicholls-Pope, soft mode, known mean;
                                                                            sum over all eigenvalues mu_b of Q_hat)
        Var(q_hat_r) = kappa_cW (1 - q_r^2) / n,  kappa_cW from tau_b = (1 + |mu_b|^2) / (1 - |mu_b|^2)
        b_r = [-b_q / q_r + Var(q_hat_r) / (2 q_r^2)] / dt
    Corrected estimate: r_tilde = r_hat - b_r. NaN if kappa_cW is undefined or |mu_b| >= 1."""

    mu = np.asarray(mu)
    b_q = -(q_r + (1 - q_r ** 2) * np.real(np.sum(mu / (1 - q_r * mu)))) / n
    a2 = np.abs(mu) ** 2
    if np.any(a2 >= 1):
        return np.nan
    tau = (1 + a2) / (1 - a2)
    n_eff = n / tau.mean()
    if n_eff <= len(mu) + 1:
        return np.nan
    var_q = n_eff / (n_eff - len(mu) - 1) * (1 - q_r ** 2) / n
    return (-b_q / q_r + var_q / (2 * q_r ** 2)) / delta_t


def kappa_r_exact(lam, n, delta_t, n_sim, chunk, rng):
    """Monte Carlo of kappa_r = n E[(S^{-1})_rr], S = G^T G (SI Section 6.2, Step 2), as in kappa_r_mc.py: whitened
    modes z_{a,k+1} = q_a z_{a,k} + sqrt(1 - q_a^2) eps, z_{a,0} ~ N(0, 1), lam[0] = r. Returns mean and its s.e."""

    N = len(lam)
    qa = np.exp(-np.asarray(lam) * delta_t)
    sa = np.sqrt(1 - qa ** 2)

    samples = []
    for start in range(0, n_sim, chunk):
        m = min(chunk, n_sim - start)
        G = np.empty((m, n, N))
        G[:, 0] = rng.standard_normal((m, N))
        for k in range(1, n):
            G[:, k] = qa * G[:, k - 1] + sa * rng.standard_normal((m, N))
        S = np.matmul(G.transpose(0, 2, 1), G)
        e_r = np.zeros((m, N, 1))
        e_r[:, 0, 0] = 1.0
        samples.append(n * np.linalg.solve(S, e_r)[:, 0, 0])
    samples = np.concatenate(samples)

    return samples.mean(), samples.std(ddof=1) / np.sqrt(len(samples))


def rel_err_r(r, kappa, n, delta_t):
    """sigma_r / r = sqrt(kappa (e^{2 r dt} - 1) / (n r^2 dt^2)) (Eq. rel_var_r, no slow-mode approximation)."""

    return np.sqrt(kappa * np.expm1(2 * r * delta_t) / (n * r ** 2 * delta_t ** 2))


######################################################################################################################
# I/O
######################################################################################################################

def load_estimates(data_dir, ensemble, q, tag="main", symmetrize=False):
    """Files <data-dir>/estimates/EST_<ensemble>_*_q<q>_*_<tag>[_sym].npz, sorted by c, stacked on a leading c axis.
    Matrices: A_true (n_c, n_A, N, N); A_hat, Q_hat, Sigma_dt_hat, Sigma_inf_hat (n_c, n_A, n_sim, N, N)."""

    suffix = f"_{tag}_sym.npz" if symmetrize else f"_{tag}.npz"
    files = [f for f in glob.glob(os.path.join(data_dir, "estimates", f"EST_{ensemble}_*_q{q}_*{suffix}"))
             if symmetrize or not f.endswith("_sym.npz")]
    if not files:
        raise FileNotFoundError(f"No estimates for {ensemble}, q={q}, tag={tag}, symmetrize={symmetrize} in {data_dir}")

    data = sorted((np.load(f) for f in files), key=lambda z: float(z["c"]))
    d = {k: np.stack([z[k] for z in data]) for k in ["A_true", "A_hat", "Q_hat", "Sigma_dt_hat", "Sigma_inf_hat"]}
    d["c_values"] = np.array([float(z["c"]) for z in data])
    d.update({k: data[0][k].item() for k in ["ensemble", "N", "q", "M", "Delta_t", "n_A", "n_sim"]})
    d["symmetrize"] = symmetrize

    return d
