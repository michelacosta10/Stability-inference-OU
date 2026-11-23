"""
kappa_W and kappa_cW against the Monte Carlo inflation factor kappa (exact definition n E[(S^-1)_rr], kappa_r_mc.py),
from the existing kappa_r_mc.py output (no new trajectories). One point per realisation of A, grid N x dt x c x q.

The stored file has r but not the full spectrum: each GOE realisation is rebuilt from its seed (same order as in
kappa_r_mc.py) and the stored r is used as a check.
    kappa_W  = n / (n - N - 1),  n = M - 1
    tau_a    = (1 + q_a^2) / (1 - q_a^2),  q_a = exp(-lambda_a dt)
    tau_bar  = arithmetic mean of tau_a over all modes
    n_eff    = n / tau_bar,  q_eff = n_eff / N
    kappa_cW = n_eff / (n_eff - N - 1)   (undefined where n_eff <= N + 1, not drawn)

Figures: fig_kcW_vs_mc_q, _N, _dt, _c, _qeff (colour = q, N, dt, c, q_eff). CSV: all points.
Summary: ratio kappa_cW / kappa_MC for q_eff > 2 and q_eff <= 2, and the points where the error on sigma_r / r,
sqrt(kappa_cW / kappa_MC) - 1, exceeds 10% in absolute value.
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from mou_core import drift_matrix
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style"))
import paperstyle as ps   # common style of the paper figures

REV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stability-inference-OU/

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="main")
parser.add_argument("--data-dir", type=str, default=os.path.join(REV_DIR, "results", "synthetic"))
parser.add_argument("--output-dir", type=str, default=os.path.join(REV_DIR, "figures", "other", "fig_kcW_vs_mc"))
parser.add_argument("--format", type=str, nargs="+", default=["pdf", "png"])
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument("--tau-w", action="store_true", dest="tau_w",
                    help="q_eff figure only: third panel with kappa_cW from the weighted mean tau_w = sum tau^2 / sum tau "
                         "(same colours = q_eff of tau_bar; output fig_kcW_vs_mc_qeff_tauw)")
parser.add_argument("--kmax", type=float, default=6.0, help="upper limit of both axes (linear)")
args = parser.parse_args()

Q_COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00"]
QEFF_EDGES = [1, 1.5, 2, 3, 5, 10]
LABEL_CW = r"$\kappa_{\rm cW}$ ($\bar\tau$)"
LABEL_CWW = r"$\kappa_{\rm cW}$ ($\langle\tau\rangle_w$)"


def spectrum_quantities(N, delta_t, c, q, seed):
    """Rebuilds one GOE realisation as in kappa_r_mc.py; returns r, kappa_cW (NaN if undefined) and q_eff."""
    A, _ = drift_matrix(N, c, "GOE", np.random.default_rng(seed))
    lam = np.linalg.eigvalsh(A)
    n = int(round(q * N)) - 1
    qa = np.exp(-lam * delta_t)
    tau_bar = np.mean((1 + qa ** 2) / (1 - qa ** 2))
    n_eff = n / tau_bar
    kappa_cW = n_eff / (n_eff - N - 1) if n_eff > N + 1 else np.nan
    tau = (1 + qa ** 2) / (1 - qa ** 2)
    n_eff_w = n / (np.sum(tau ** 2) / np.sum(tau))      # weighted mean tau_w = sum tau^2 / sum tau
    kappa_cWw = n_eff_w / (n_eff_w - N - 1) if n_eff_w > N + 1 else np.nan
    return lam[0], kappa_cW, n_eff / N, kappa_cWw


def setup_axes(axes, lim):
    for ax, name in zip(axes, [r"$\kappa_W$", LABEL_CW, LABEL_CWW]):
        ax.plot(lim, lim, "-", color="0.35", lw=1.0, zorder=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\kappa$ Monte Carlo", fontsize=13)
        ax.set_ylabel(name, fontsize=13)
        ax.grid(True, alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)


def save(fig, name):
    for fmt in args.format:
        path = os.path.join(args.output_dir, f"{name}_{args.tag}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=args.dpi)
        print(f"Saved: {path}")


if __name__ == "__main__":

    os.makedirs(args.output_dir, exist_ok=True)
    z = np.load(os.path.join(args.data_dir, "kappa_r_mc", f"kappa_r_mc_{args.tag}.npz"))
    d = pd.DataFrame({k: z[k] for k in ["N", "delta_t", "c", "q", "r", "kappa_W", "kappa_r_mc"]})
    d = d.rename(columns={"kappa_r_mc": "kappa_mc"})

    seeds = np.random.SeedSequence(int(z["seed"])).generate_state(len(d))
    rows = [spectrum_quantities(int(N), dt, c, q, int(s))
            for N, dt, c, q, s in zip(d["N"], d["delta_t"], d["c"], d["q"], seeds)]
    r_check, d["kappa_cW"], d["q_eff"], d["kappa_cWw"] = map(np.array, zip(*rows))
    if not np.allclose(r_check, d["r"]):
        raise RuntimeError("rebuilt spectra do not match the stored r")

    d["ratio_cW"] = d["kappa_cW"] / d["kappa_mc"]
    d["err_sigma_cW"] = np.sqrt(d["ratio_cW"]) - 1
    d.to_csv(os.path.join(args.output_dir, f"fig_kcW_vs_mc_{args.tag}.csv"), index=False)

    lim = (0.95, args.kmax)
    ok = d["kappa_cW"].notna()

    # colour = one grid parameter
    for col, label, name in [("q", "$q={:g}$", "q"), ("N", "$N={:g}$", "N"), ("delta_t", r"$\Delta t={:g}$", "dt"),
                             ("c", "$c={:g}$", "c")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
        setup_axes(axes, lim)
        for val, color in zip(sorted(d[col].unique()), Q_COLORS):
            s = d[d[col] == val]
            axes[0].scatter(s["kappa_mc"], s["kappa_W"], color=color, s=12, alpha=0.6, linewidths=0, zorder=2)
            axes[1].scatter(s["kappa_mc"], s["kappa_cW"], color=color, s=12, alpha=0.6, linewidths=0, zorder=2,
                            label=label.format(val))
        axes[1].legend(frameon=False, fontsize=10, markerscale=2, loc="lower right")
        fig.tight_layout()
        save(fig, f"fig_kcW_vs_mc_{name}")

    # colour = q_eff in discrete bands (paper SI Fig. 1): bright greens (small q_eff) to light blues (paperstyle.QEFF_COLORS),
    # one colour per band; grey where kappa_cW is undefined, n_eff <= N + 1 (kappa_W panel only); every defined point
    # has q_eff > 1 + 1/N, inside the first band. Paper style, 183 mm wide.
    edges = QEFF_EDGES + [max(d["q_eff"].max(), QEFF_EDGES[-1]) * 1.01]
    cmap = ListedColormap(ps.QEFF_COLORS[:len(edges) - 1]).with_extremes(under="0.6")
    norm = BoundaryNorm(edges, cmap.N)
    color_q = d["q_eff"].where(ok, QEFF_EDGES[0] - 1.0)
    n_pan = 3 if args.tau_w else 2
    with plt.style.context(ps.CTX):
        fig, axes = plt.subplots(1, n_pan, figsize=(ps.W2, 2.45), layout="constrained")
        setup_axes(axes, lim)
        axes[0].scatter(d["kappa_mc"], d["kappa_W"], c=color_q, cmap=cmap, norm=norm, s=5, linewidths=0, zorder=2)
        sc = axes[1].scatter(d.loc[ok, "kappa_mc"], d.loc[ok, "kappa_cW"], c=color_q[ok], cmap=cmap, norm=norm,
                             s=5, linewidths=0, zorder=2)
        if args.tau_w:
            okw = d["kappa_cWw"].notna()
            axes[2].scatter(d.loc[okw, "kappa_mc"], d.loc[okw, "kappa_cWw"], c=color_q[okw], cmap=cmap, norm=norm,
                            s=5, linewidths=0, zorder=2)
            print(f"kappa_cW (tau_w) undefined: {(~okw).sum()}, above the axis: {(d['kappa_cWw'] > lim[1]).sum()}")
        for ax, letter in zip(axes, "abc"):
            ax.set_xlabel(r"$\kappa$ Monte Carlo")
            ax.yaxis.label.set_size(plt.rcParams["axes.labelsize"])
            ax.xaxis.label.set_size(plt.rcParams["axes.labelsize"])
            ax.grid(False)
            ps.panel_label(ax, letter)
        cb = fig.colorbar(sc, ax=axes, label=r"$q_{\rm eff}=n_{\rm eff}/N$", aspect=18, pad=0.015, extend="min",
                          spacing="uniform", ticks=edges[:-1])
        cb.ax.set_yticklabels([f"{e:g}" for e in edges[:-1]])
        cb.outline.set_linewidth(0.6)
        for fmt in args.format:
            path = os.path.join(args.output_dir, f"fig_kcW_vs_mc_qeff{'_tauw' if args.tau_w else ''}_{args.tag}.{fmt}")
            fig.savefig(path, dpi=600)
            print(f"Saved: {path}")

    # summary
    pd.set_option("display.width", 200)
    print(f"\npoints: {len(d)}, kappa_cW undefined (n_eff <= N + 1): {(~ok).sum()}, "
          f"kappa_cW above the axis: {(d['kappa_cW'] > lim[1]).sum()}, Monte Carlo beyond the axis: {(d['kappa_mc'] > lim[1]).sum()}")
    d["range"] = np.where(d["q_eff"] > 2, "q_eff > 2", "q_eff <= 2")
    print("\nratio kappa_cW / kappa_MC (defined points only)")
    print(d.groupby("range").agg(n=("q", "size"), defined=("ratio_cW", "count"), min=("ratio_cW", "min"),
                                 median=("ratio_cW", "median"), max=("ratio_cW", "max")).round(3).to_string())
    big = d[ok & (d["err_sigma_cW"].abs() > 0.10)]
    print(f"\n|error on sigma_r / r| > 10%: {len(big)} of {ok.sum()} defined points "
          f"(over: {(big['err_sigma_cW'] > 0).sum()}, under: {(big['err_sigma_cW'] < 0).sum()})")
    for col in ["range", "c", "q", "delta_t", "N"]:
        print(d[ok].groupby(col).apply(lambda g: f"{(g['err_sigma_cW'].abs() > 0.10).sum()}/{len(g)}").to_string(), "\n")
