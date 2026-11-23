"""
Maximum-likelihood estimates (Methods) for every trajectory produced by generate_data.py.
Same estimator for both ensembles, Q_hat = T2 T3^{-1};
--symmetrize uses the symmetrised estimator (Q_hat + Q_hat^T)/2.

Input:  <data-dir>/sim_trajectories/DATASIM_<...>_c<c>_<tag>.npz   (default <data-dir> = results/synthetic)
Output, one file per c: <data-dir>/estimates/EST_<...>_c<c>_<tag>[_sym].npz with A_true (n_A, N, N) and
A_hat, Q_hat, Sigma_dt_hat, Sigma_inf_hat (n_A, n_sim, N, N), float32, NaN where the estimate failed.
"""

import argparse
import glob
import os
import time

import numpy as np
from joblib import Parallel, delayed

from mou_core import estimate

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

REV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stability-inference-OU/


######################################################################################################################
# ARGUMENTS
######################################################################################################################

parser = argparse.ArgumentParser(description="Estimate A, Q, Sigma_dt, Sigma_inf from the simulated trajectories.")
parser.add_argument("--ensemble", type=str, default="GOE", choices=["GOE", "GINIBRE"])
parser.add_argument("--q", type=int, default=10)
parser.add_argument("--tag", type=str, default="main")
parser.add_argument("--symmetrize", action="store_true", help="Q_hat -> (Q_hat + Q_hat^T)/2 (symmetrised estimator)")
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument("--data-dir", type=str, default=os.path.join(REV_DIR, "results", "synthetic"))
args = parser.parse_args()

sim_dir = os.path.join(args.data_dir, "sim_trajectories")
est_dir = os.path.join(args.data_dir, "estimates")
os.makedirs(est_dir, exist_ok=True)


######################################################################################################################
# ONE REALISATION OF A
######################################################################################################################

def estimate_one_A(X_block, delta_t, symmetrize):
    """X_block: (n_sim, N, M). Returns the four estimated matrices, (n_sim, N, N) each, and the failure count."""

    n_sim, N, _ = X_block.shape
    out = {k: np.full((n_sim, N, N), np.nan, dtype=np.float32) for k in ["A_hat", "Q_hat", "Sigma_dt_hat", "Sigma_inf_hat"]}
    fail = 0

    for s in range(n_sim):
        try:
            mats = estimate(X_block[s], delta_t, symmetrize)
        except Exception:
            fail += 1
            continue
        if not all(np.isfinite(m).all() for m in mats):
            fail += 1
            continue
        for k, m in zip(["A_hat", "Q_hat", "Sigma_dt_hat", "Sigma_inf_hat"], mats):
            out[k][s] = m

    return out, fail


######################################################################################################################
# MAIN: one file per c
######################################################################################################################

if __name__ == "__main__":

    pattern = os.path.join(sim_dir, f"DATASIM_{args.ensemble}_*_q{args.q}_*_{args.tag}.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No simulation file matches {pattern}")

    print(f"{len(files)} files, symmetrize={args.symmetrize}")

    for path in files:

        t0 = time.time()
        data = np.load(path)
        X, Delta_t = data["X"], float(data["Delta_t"])

        res = Parallel(n_jobs=args.n_jobs)(
            delayed(estimate_one_A)(X[i_A], Delta_t, args.symmetrize) for i_A in range(X.shape[0])
        )

        name = os.path.basename(path).replace("DATASIM_", "EST_")
        if args.symmetrize:
            name = name.replace(".npz", "_sym.npz")
        out_path = os.path.join(est_dir, name)

        meta = {k: data[k] for k in ["ensemble", "c", "N", "q", "M", "Delta_t", "n_A", "n_sim", "tag"]}
        np.savez(
            out_path,
            A_true=data["A_true"],
            **{k: np.stack([r[0][k] for r in res]) for k in ["A_hat", "Q_hat", "Sigma_dt_hat", "Sigma_inf_hat"]},
            fail_per_A=np.array([r[1] for r in res]), symmetrize=args.symmetrize, source=os.path.basename(path), **meta,
        )
        print(f"c = {float(data['c']):g}: {time.time() - t0:.1f} s, failures = {sum(r[1] for r in res)}  ->  {out_path}")
