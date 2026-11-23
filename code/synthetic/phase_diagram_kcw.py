"""
Fig. 3 (fig:phase_diagram): GOE phase diagrams of sigma_r / r at N = 100 with kappa = kappa_cW (Methods).

Spectrum (mou_core.semicircle_spectrum): soft mode at r_bar(c, N) with the Tracy-Widom correction, the other N - 1
modes at the quantiles of the semicircle shifted so that its lower edge is at r_bar. On this spectrum, per grid point,
    tau_bar     = arithmetic mean of tau_a = (1 + q_a^2) / (1 - q_a^2),  q_a = exp(-lambda_a dt)
    n_eff       = n / tau_bar,  q_eff = n_eff / N = q / tau_bar,  n = round(q N)  (q = n / N, as in the text)
    kappa_cW    = n_eff / (n_eff - N - 1)   (NaN where n_eff <= N + 1: undefined, drawn in grey)
    sigma_r / r = sqrt(kappa_cW (e^{2 r dt} - 1) / (n r^2 dt^2))
Analytic, no Monte Carlo.
Panels: 1 = (c, q) at dt = dt_fixed; 2 = (c, dt) at q = q_fixed; 3 = (q, dt) at c = c_fixed (arrays: y rows, x columns).
Lines: gold curve q_eff = 1 at dt_fixed, i.e. q* = tau_bar; dt* = 0.80 / r_bar.
Output: data/phase_diagram/phase_diagram_<tag>.npz (default tag kcw). Plot with fig3_phase_diagram.py --tag kcw.
"""

import argparse
import os

import numpy as np

from mou_core import r_bar, semicircle_spectrum, tau_bar, kappa_cW, rel_err_r

REV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stability-inference-OU/

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=100)
parser.add_argument("--dt-fixed", type=float, default=1.0, dest="dt_fixed")
parser.add_argument("--q-fixed", type=float, default=10.0, dest="q_fixed")
parser.add_argument("--c-fixed", type=float, default=0.6, dest="c_fixed")
parser.add_argument("--c-range", type=float, nargs=2, default=[0.05, 0.72], dest="c_range")
parser.add_argument("--n-c", type=int, default=200, dest="n_c")
parser.add_argument("--q-max", type=float, default=40.0, dest="q_max")
parser.add_argument("--dt-max", type=float, nargs=2, default=[15.0, 10.0], dest="dt_max",
                    help="upper dt in panels 2 and 3")
parser.add_argument("--n-dt", type=int, default=300, dest="n_dt")
parser.add_argument("--tag", type=str, default="kcw")
parser.add_argument("--data-dir", type=str, default=os.path.join(REV_DIR, "results", "synthetic"))
args = parser.parse_args()


def q_grid(q_max):
    """Dense close to q = 1, where kappa_cW becomes undefined (analytic, so the grid can be fine)."""
    return np.concatenate([np.linspace(1.02, 3.0, 200), np.arange(3.1, q_max + 0.05, 0.1)])


_SPEC = {}


def point(c, q, dt, N):
    """q_eff and kappa_cW at one grid point (NaN where n <= N + 1 or kappa_cW is undefined)."""
    n = int(round(q * N))
    if n <= N + 1:
        return np.nan, np.nan
    if c not in _SPEC:
        _SPEC[c] = semicircle_spectrum(c, N)
    kap, q_eff = kappa_cW(_SPEC[c], n, dt)
    return float(q_eff), float(kap)


if __name__ == "__main__":

    N = args.N
    c_grid = np.linspace(*args.c_range, args.n_c)
    qg = q_grid(args.q_max)
    dt2 = np.linspace(0.05, args.dt_max[0], args.n_dt)
    dt3 = np.linspace(0.05, args.dt_max[1], args.n_dt)
    panels = {
        1: dict(x=c_grid, y=qg, cqdt=lambda x, y: (x, y, args.dt_fixed)),
        2: dict(x=c_grid, y=dt2, cqdt=lambda x, y: (x, args.q_fixed, y)),
        3: dict(x=qg, y=dt3, cqdt=lambda x, y: (args.c_fixed, x, y)),
    }

    out = dict(N=N, dt_fixed=args.dt_fixed, q_fixed=args.q_fixed, c_fixed=args.c_fixed, mu_TW=-1.21,
               kappa="kappa_cW")
    for p, P in panels.items():
        shape = (len(P["y"]), len(P["x"]))
        q_eff, kap = np.full(shape, np.nan), np.full(shape, np.nan)
        for iy, y in enumerate(P["y"]):
            for ix, x in enumerate(P["x"]):
                q_eff[iy, ix], kap[iy, ix] = point(*P["cqdt"](x, y), N)
        C, Q, DT = np.broadcast_arrays(*P["cqdt"](P["x"][None, :], P["y"][:, None]))
        n = np.round(Q * N)
        out.update({f"p{p}_x": P["x"], f"p{p}_y": P["y"], f"p{p}_q_eff": q_eff, f"p{p}_kappa": kap,
                    f"p{p}_rel_err": rel_err_r(r_bar(C, N), kap, n, DT)})

    # gold curve: q_eff = 1 at dt_fixed
    c_fine = np.linspace(*args.c_range, 400)
    out["c_fine"] = c_fine
    out["q_star_fine"] = np.array([tau_bar(semicircle_spectrum(c, N), args.dt_fixed) for c in c_fine])
    out["p1_q_star"] = np.array([tau_bar(semicircle_spectrum(c, N), args.dt_fixed) for c in c_grid])

    # optimal sampling interval
    out["p2_dt_star_0"] = 0.80 / r_bar(c_fine, N)
    out["p3_dt_star_0"] = 0.80 / r_bar(args.c_fixed, N)

    out_dir = os.path.join(args.data_dir, "phase_diagram")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"phase_diagram_{args.tag}.npz")
    np.savez(path, **out)
    print(f"Saved: {path}")
