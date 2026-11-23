"""
Financial data: one estimate pooled over all trading days, at sampling intervals up to 1 minute.

The sufficient statistics T1, T2, T3 are accumulated over all days using only pairs (t, t+1) inside the same day,
so there is no artificial overnight transition: this is the exact (conditional) likelihood of independent daily
segments sharing the same drift matrix A. M = total number of within-day pairs, so q = M / N grows with the
number of days.

Centering variants (after the transform, then one global std):
  panel    : per-asset mean over all days (paper Methods)
  day      : per-asset mean of each day (removes day-to-day level shifts, which would look like a very slow mode)
  deseason : 'day' + subtraction of each asset's mean intraday profile (U-shape)

sigma_r: moving-block bootstrap with blocks drawn inside days (never across the night), pooled the same way;
theory: sqrt(kappa (e^{2 r dt} - 1) / (n dt^2)) / r with n = M (M already counts transitions), kappa_r and kappa_cW
on the eigenvalues of Q_hat.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import logm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ou_core as ou

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--factors", type=int, nargs="+", default=[1, 2, 5, 10, 30, 60])
parser.add_argument("--transforms", nargs="+", default=["logrv", "abs_sub"], choices=["logrv", "abs_sub"])
parser.add_argument("--centerings", nargs="+", default=["panel", "day", "deseason"],
                    choices=["panel", "day", "deseason"])
parser.add_argument("--n-boot", type=int, default=200)
parser.add_argument("--c0", type=float, default=5.0)
parser.add_argument("--seed", type=int, default=12345)
parser.add_argument("--n-jobs", type=int, default=-1)
args = parser.parse_args()


def build(days, k, transform, centering):
    """List of per-day arrays (T_day x N) at coarsening factor k."""

    if transform == "logrv":
        rv = []
        for _, R in days:
            n = R.shape[0] // k
            rv.append((R[:n * k] ** 2).reshape(n, k, -1).sum(axis=1))
        allrv = np.concatenate(rv)
        floor = np.array([allrv[:, j][allrv[:, j] > 0].min() / 2 for j in range(allrv.shape[1])])
        segs = [np.log(np.maximum(x, floor)) for x in rv]
    else:
        segs = [np.abs(R[::k]) for _, R in days]

    if centering == "panel":
        mu = np.concatenate(segs).mean(axis=0, keepdims=True)
        segs = [x - mu for x in segs]
    else:
        segs = [x - x.mean(axis=0, keepdims=True) for x in segs]
        if centering == "deseason":
            n = min(x.shape[0] for x in segs)
            prof = np.mean([x[:n] for x in segs], axis=0)
            segs = [x[:n] - prof for x in segs]

    sd = np.concatenate(segs).std()
    return [x / sd for x in segs]


def pooled_r(segs, delta_t):
    """r_hat and Q_hat from T2, T3 summed over segments (pairs never cross a segment boundary)."""

    T2 = sum(x[1:].T @ x[:-1] for x in segs)
    T3 = sum(x[:-1].T @ x[:-1] for x in segs)
    Q = np.real(T2 @ np.linalg.inv(T3))
    A = np.real(-logm(Q, disp=False)[0] / delta_t)
    M = sum(x.shape[0] - 1 for x in segs)

    return np.min(np.real(np.linalg.eigvals(A))), Q, M


def block_bootstrap(segs, delta_t, Q, seed):
    """Moving-block bootstrap with blocks drawn inside days; each resample has the same total length."""

    rng = np.random.default_rng(seed)
    tau = ou.estimate_tau(Q)
    shortest = min(x.shape[0] for x in segs)
    L = 2 if not np.isfinite(tau) else max(int(args.c0 * tau), 2)
    L = max(min(L, shortest // 2), 2)

    # all admissible (day, start) pairs
    starts = [(d, s) for d, x in enumerate(segs) for s in range(x.shape[0] - L + 1)]
    n_blocks = int(np.ceil(sum(x.shape[0] for x in segs) / L))

    r_boot = np.full(args.n_boot, np.nan)
    for b in range(args.n_boot):
        pick = rng.integers(0, len(starts), size=n_blocks)
        blocks = [segs[starts[i][0]][starts[i][1]:starts[i][1] + L] for i in pick]
        try:
            r_boot[b] = pooled_r(blocks, delta_t)[0]
        except Exception:
            pass

    return r_boot, L


def run(days, k, transform, centering, seed):

    delta_t = float(k)
    segs = build(days, k, transform, centering)
    N = segs[0].shape[1]
    r, Q, M = pooled_r(segs, delta_t)
    r_boot, L = block_bootstrap(segs, delta_t, Q, seed)
    q_eigs = np.linalg.eigvals(Q)
    kr, kcw = ou.kappa_r(M, q_eigs), ou.kappa_cW(M, N, q_eigs)
    sb = np.nanstd(r_boot)

    return {
        "transform": transform, "centering": centering, "k": k, "delta_t": delta_t, "N": N, "M": M, "q": M / N,
        "T_day": segs[0].shape[0], "r_hat": r, "r_dt": r * delta_t, "sigma_boot": sb, "block_L": L,
        "rel_boot": sb / r if r > 0 else np.inf,
        "kappa_r": kr, "kappa_cW": kcw,
        "rel_theory_r": ou.rel_err_theory(r, M, delta_t, kr),
        "rel_theory_cW": ou.rel_err_theory(r, M, delta_t, kcw),
        "dt_star": 1.5936 / (2 * r) if r > 0 else np.nan,
        "r_boot": r_boot,
    }


if __name__ == "__main__":

    t0 = time.time()
    days = ou.fin_days(ou.load_fin_returns())
    print(f"{len(days)} days, N = {days[0][1].shape[1]}, B = {args.n_boot}")

    jobs = [(tr, ce, k) for tr in args.transforms for ce in args.centerings for k in args.factors]
    seeds = np.random.SeedSequence(args.seed).spawn(len(jobs))
    out = Parallel(n_jobs=args.n_jobs)(delayed(run)(days, k, tr, ce, sd) for (tr, ce, k), sd in zip(jobs, seeds))

    for o in out:
        print(f"{o['transform']:7s} {o['centering']:8s} dt = {o['delta_t']:4.0f} s  T_day = {o['T_day']:5d}  "
              f"M = {o['M']:6d}  q = {o['q']:7.1f}  r_hat = {o['r_hat']:.5f} 1/s  r*dt = {o['r_dt']:.4f}  "
              f"dt* = {o['dt_star']:7.1f} s  sigma/r boot = {o['rel_boot']:.3f} (L = {o['block_L']})  "
              f"theory k_r = {o['rel_theory_r']:.3f}  k_cW = {o['rel_theory_cW']:.3f}", flush=True)

    path = os.path.join(ou.RESULTS_DIR, "pooled_days_fin.pkl")
    with open(path, "wb") as f:
        pickle.dump({"config": vars(args), "results": out}, f)

    import pandas as pd
    pd.DataFrame([{k: v for k, v in o.items() if k != "r_boot"} for o in out]).to_csv(
        os.path.join(ou.RESULTS_DIR, "summary_pooled_days.csv"), index=False)
    print(f"\nSaved: {path}, results/summary_pooled_days.csv  ({time.time() - t0:.1f} s)")
