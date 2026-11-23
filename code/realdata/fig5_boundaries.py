"""
Paper Fig. 5: GOE boundaries sigma_r / r = 1 (diagnosis) and 0.1 (resolution) for each empirical dataset.

Each dataset is mapped onto the GOE model as in estimate_realdata_optimal.py (spectrum centred at 1): N and
dt_mod = lam_bar * dt. On the spectrum of Fig. 3 (mou_core.semicircle_spectrum), on a (c, q) grid:
    kappa_cW    = n_eff / (n_eff - N - 1),  n_eff = n / tau_bar,  tau_bar = arithmetic mean of tau_a (mou_core.kappa_cW,
                  as Fig. 3); NaN where n_eff <= N + 1 (undefined: no boundary there). Analytic, no Monte Carlo.
    sigma_r / r = sqrt(kappa_cW (e^{2 r dt} - 1) / (n r^2 dt^2)),  n = round(q N)  (q = n / N, as in the text)
Boundary q*(c): smallest q with sigma_r / r <= threshold (sigma_r / r decreases with q), interpolated in log q.
Input: results/realdata_fig5.pkl. Output: results/fig5_boundaries.npz, read by plots_realdata_revised.py.
"""

import argparse
import os
import pickle
import sys

import numpy as np

REV_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REV_DIR, "synthetic"))
from mou_core import r_bar, semicircle_spectrum, kappa_cW, rel_err_r

RESULTS_DIR = os.path.join(os.path.dirname(REV_DIR), "results", "realdata")

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default=os.path.join(RESULTS_DIR, "realdata_fig5.pkl"))
parser.add_argument("--c-range", type=float, nargs=2, default=[0.05, 0.74], dest="c_range")
parser.add_argument("--n-c", type=int, default=80, dest="n_c")
parser.add_argument("--q-range", type=float, nargs=2, default=[1.1, 1e5], dest="q_range")
parser.add_argument("--n-q", type=int, default=200, dest="n_q")
parser.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 0.1])
parser.add_argument("--dt-key", type=str, default="dt_mod", choices=["dt_mod", "delta_t"],
                    help="model dt: dt_mod = lam_bar * dt (default) or the physical dt")
parser.add_argument("--out", type=str, default=os.path.join(RESULTS_DIR, "fig5_boundaries.npz"))
args = parser.parse_args()


def kcw_grid(N, dt, c_grid, q_grid):
    """n, q_eff and kappa_cW on the (c, q) grid (rows c, columns q); NaN where n <= N + 1, r_bar <= 0 or kappa_cW
    is undefined."""

    n = np.round(q_grid * N)
    q_eff = np.full((len(c_grid), len(q_grid)), np.nan)
    kap = np.full((len(c_grid), len(q_grid)), np.nan)
    for i, c in enumerate(c_grid):
        if r_bar(c, N) > 0:
            kap[i], q_eff[i] = kappa_cW(semicircle_spectrum(c, N), n, dt)
    q_eff[:, n <= N + 1] = np.nan
    kap[:, n <= N + 1] = np.nan

    return n, q_eff, kap


def boundary(q_grid, E, threshold):
    """Smallest q with E <= threshold for each row, interpolated linearly in (log q, log E)."""

    q_star = np.full(E.shape[0], np.nan)
    for i, e in enumerate(E):
        ok = np.isfinite(e)
        j = np.where(ok & (e <= threshold))[0]
        if len(j) == 0:
            continue
        j = j[0]
        if j == 0 or not ok[j - 1]:
            q_star[i] = q_grid[j]
            continue
        x0, x1, y0, y1 = np.log(q_grid[j - 1]), np.log(q_grid[j]), np.log(e[j - 1]), np.log(e[j])
        q_star[i] = np.exp(x0 + (np.log(threshold) - y0) * (x1 - x0) / (y1 - y0))

    return q_star


if __name__ == "__main__":

    with open(args.data, "rb") as f:
        data = pickle.load(f)

    c_grid = np.linspace(*args.c_range, args.n_c)
    q_grid = np.geomspace(*args.q_range, args.n_q)
    names = [n for n in ("eeg", "fin", "plankton") if n in data["summary"]]

    grids = {}
    for name in names:
        N = data["datasets"][name]["N"]
        dt = data["summary"][name]["dt_mod"] if args.dt_key == "dt_mod" else data["datasets"][name]["delta_t"]
        n, q_eff, kap = kcw_grid(N, dt, c_grid, q_grid)
        grids[name] = dict(N=N, dt=dt, n=n, q_eff=q_eff, kappa=kap)
        print(f"{name}: N = {N}, dt_mod = {dt:.4f}")

    out = dict(c_grid=c_grid, q_grid=q_grid, thresholds=np.array(args.thresholds), kappa="kappa_cW",
               names=np.array(names))
    for name, g in grids.items():
        R = r_bar(c_grid, g["N"])[:, None]
        with np.errstate(invalid="ignore"):
            E = np.where(R > 0, rel_err_r(R, g["kappa"], g["n"][None, :], g["dt"]), np.nan)
        out.update({f"{name}_dt_mod": g["dt"], f"{name}_q_eff": g["q_eff"], f"{name}_kappa": g["kappa"],
                    f"{name}_rel_err": E})
        for t, thr in enumerate(args.thresholds):
            out[f"{name}_q_star_{t}"] = boundary(q_grid, E, thr)

    np.savez(args.out, **out)
    print(f"Saved: {args.out}")
