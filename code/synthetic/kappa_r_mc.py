"""
Monte Carlo of the soft-mode inflation factor kappa_r (Supplementary Section 6.2-6.3, Fig. SX).

For symmetric A the modes are independent AR(1) processes, so kappa_r is measured directly in the eigenbasis,
without simulating the N x N process: for each GOE realisation (mou_core.drift_matrix), with q_a = exp(-lambda_a dt),
the whitened modes z_{a,k+1} = q_a z_{a,k} + sqrt(1 - q_a^2) eps, z_{a,0} ~ N(0, 1), give the regressor matrix
G (n x N, n = M - 1) and
    kappa_r = n E[(S^{-1})_rr],  S = G^T G      (exact definition, SI Step 2; no bias or delta-method terms),
averaged over n_sim trajectories. Compared with the first order
    kappa_r^(1) = 1 + 1/q_eff,  q_eff = n / (sum_{b != r} tau_rb + 2 tau_r),  tau_rb = (1 + q_r q_b) / (1 - q_r q_b).

Grid: N x dt x c x q (q = M / N), n_A realisations of A per point. Output: data/kappa_r_mc/kappa_r_mc_<tag>.npz,
one entry per realisation of A. Plot with figS_kappa_r.py.
"""

import argparse
import itertools
import os
import time

import numpy as np
from joblib import Parallel, delayed

from mou_core import drift_matrix

REV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stability-inference-OU/

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, nargs="+", default=[50, 100, 200])
parser.add_argument("--delta-t", type=float, nargs="+", default=[0.5, 1.0, 2.0], dest="delta_t")
parser.add_argument("--c-values", type=float, nargs="+", default=[0.3, 0.5, 0.6, 0.7], dest="c_values")
parser.add_argument("--q", type=float, nargs="+", default=[2, 3, 5, 10, 20])
parser.add_argument("--n-A", type=int, default=10, dest="n_A")
parser.add_argument("--n-sim", type=int, default=200, dest="n_sim")
parser.add_argument("--chunk", type=int, default=25, help="trajectories per batch (memory)")
parser.add_argument("--seed", type=int, default=20261027)
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument("--tag", type=str, default="main")
parser.add_argument("--data-dir", type=str, default=os.path.join(REV_DIR, "results", "synthetic"))
args = parser.parse_args()


def first_order(lam, n, delta_t):
    """q_eff and kappa_r^(1) = 1 + 1/q_eff for eigenvalues lam (ascending, lam[0] = r)."""
    qa = np.exp(-lam * delta_t)
    q_r, q_b = qa[0], qa[1:]
    tau_rb = (1 + q_r * q_b) / (1 - q_r * q_b)
    tau_r = (1 + q_r ** 2) / (1 - q_r ** 2)
    q_eff = n / (tau_rb.sum() + 2 * tau_r)
    return q_eff, 1 + 1 / q_eff


def kappa_r_one_A(N, delta_t, c, q, n_sim, chunk, seed):
    """One GOE realisation: r, q_eff, kappa_r^(1), and Monte Carlo kappa_r with its standard error."""

    rng = np.random.default_rng(seed)
    A, _ = drift_matrix(N, c, "GOE", rng)
    lam = np.linalg.eigvalsh(A)
    M = int(round(q * N))
    n = M - 1
    qa = np.exp(-lam * delta_t)
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
    q_eff, kr1 = first_order(lam, n, delta_t)

    return dict(N=N, delta_t=delta_t, c=c, q=q, n=n, r=lam[0], q_eff=q_eff, kappa_r1=kr1,
                kappa_W=n / (n - N - 1), kappa_r_mc=samples.mean(),
                kappa_r_mc_sem=samples.std(ddof=1) / np.sqrt(len(samples)))


if __name__ == "__main__":

    out_dir = os.path.join(args.data_dir, "kappa_r_mc")
    os.makedirs(out_dir, exist_ok=True)

    grid = [(N, dt, c, q) for N, dt, c, q in itertools.product(args.N, args.delta_t, args.c_values, args.q)
            if q * N - 1 > N + 2]
    tasks = [(N, dt, c, q, i_A) for (N, dt, c, q) in grid for i_A in range(args.n_A)]
    seeds = np.random.SeedSequence(args.seed).generate_state(len(tasks))
    print(f"{len(grid)} grid points x {args.n_A} realisations = {len(tasks)} tasks, n_sim = {args.n_sim}")

    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(kappa_r_one_A)(N, dt, c, q, args.n_sim, args.chunk, int(s))
        for (N, dt, c, q, _), s in zip(tasks, seeds)
    )
    print(f"done in {time.time() - t0:.0f} s")

    path = os.path.join(out_dir, f"kappa_r_mc_{args.tag}.npz")
    np.savez(path, **{k: np.array([r[k] for r in res]) for k in res[0]},
             n_sim=args.n_sim, n_A=args.n_A, seed=args.seed)
    print(f"Saved: {path}")
