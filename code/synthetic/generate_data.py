"""
Synthetic data: n_A stable drift matrices A per value of c, and n_sim stationary trajectories per A
(double Monte Carlo, Methods): exact VAR(1) update and stationary start (see mou_core.py).

Output, one file per c: <data-dir>/sim_trajectories/ (default results/synthetic/sim_trajectories/)
DATASIM_<ENSEMBLE>_N<N>_q<q>_Dt<dt>_nA<n_A>_nsim<n_sim>_c<c>_<tag>.npz
with A_true (n_A, N, N) and X (n_A, n_sim, N, M), float32.
"""

import argparse
import os
import time

import numpy as np
from joblib import Parallel, delayed

from mou_core import drift_matrix, true_parameters, simulate_var1

REV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stability-inference-OU/


######################################################################################################################
# ARGUMENTS
######################################################################################################################

parser = argparse.ArgumentParser(description="Simulate MOU trajectories for GOE or Ginibre drift matrices.")
parser.add_argument("--ensemble", type=str, default="GOE", choices=["GOE", "GINIBRE"])
parser.add_argument("--N", type=int, default=100)
parser.add_argument("--q", type=int, default=10, help="M = q N")
parser.add_argument("--delta-t", type=float, default=1.0, dest="Delta_t")
parser.add_argument("--n-A", type=int, default=30, dest="n_A")
parser.add_argument("--n-sim", type=int, default=30, dest="n_sim")
parser.add_argument("--c-values", type=float, nargs="+", default=None, dest="c_values",
                    help="default: 0.1..0.7 (GOE), 0.1..0.9, 0.99 (Ginibre), as in the paper")
parser.add_argument("--seed", type=int, default=20261024)
parser.add_argument("--n-jobs", type=int, default=-1)
parser.add_argument("--tag", type=str, default="main")
parser.add_argument("--data-dir", type=str, default=os.path.join(REV_DIR, "results", "synthetic"))
args = parser.parse_args()

if args.c_values is None:
    args.c_values = ([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] if args.ensemble == "GOE"
                     else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

save_dir = os.path.join(args.data_dir, "sim_trajectories")
os.makedirs(save_dir, exist_ok=True)


def sim_filename(ensemble, N, q, delta_t, n_A, n_sim, c, tag):
    return f"DATASIM_{ensemble}_N{N}_q{q}_Dt{delta_t:g}_nA{n_A}_nsim{n_sim}_c{c:g}_{tag}.npz"


######################################################################################################################
# ONE REALISATION OF A
######################################################################################################################

def run_one_A(N, M, c, delta_t, n_sim, ensemble, seed):

    rng = np.random.default_rng(seed)
    A, n_rej = drift_matrix(N, c, ensemble, rng)
    Q, S_inf, S_dt = true_parameters(A, delta_t)
    X = simulate_var1(Q, S_inf, S_dt, M, n_sim, rng)

    return A.astype(np.float32), X.astype(np.float32), n_rej


######################################################################################################################
# MAIN: one file per c
######################################################################################################################

if __name__ == "__main__":

    ensemble, N, Delta_t, n_A, n_sim = args.ensemble, args.N, args.Delta_t, args.n_A, args.n_sim
    M = int(args.q * N)
    c_values = np.array(args.c_values)

    print(f"{ensemble}  N={N}  M={M}  dt={Delta_t}  n_A={n_A}  n_sim={n_sim}  c={c_values}")

    seeds = np.random.SeedSequence(args.seed).generate_state(len(c_values) * n_A).reshape(len(c_values), n_A)

    for i_c, c in enumerate(c_values):

        t0 = time.time()
        out = Parallel(n_jobs=args.n_jobs)(
            delayed(run_one_A)(N, M, c, Delta_t, n_sim, ensemble, int(seeds[i_c, i_A])) for i_A in range(n_A)
        )

        path = os.path.join(save_dir, sim_filename(ensemble, N, args.q, Delta_t, n_A, n_sim, c, args.tag))
        np.savez(
            path,
            A_true=np.stack([o[0] for o in out]), X=np.stack([o[1] for o in out]),
            n_rej=np.array([o[2] for o in out]), seeds_A=seeds[i_c],
            ensemble=ensemble, c=c, N=N, q=args.q, M=M, Delta_t=Delta_t, n_A=n_A, n_sim=n_sim, tag=args.tag,
        )
        print(f"c = {c:g}: {time.time() - t0:.1f} s, rejected A = {sum(o[2] for o in out)}  ->  {path}")
