"""
Paper Fig. fig:ginibre_sigma_r (Methods): Ginibre sigma_r / r.
  CSV only:
  Top left:  distribution of ||A_hat - A||_F^2 / N per c, GOE (paper: N = 100, q = 40, dt = 1).
  Top right: same for the Ginibre ensemble (paper: N = 100, q = 10, dt = 1).
  Bottom:    Ginibre sigma_r / r, with r = min Re lambda(A) and r_hat = min Re lambda(A_hat); per realisation of A,
             std of r_hat over trajectories / r, then median and 25-75% over A, one point per c.
Histograms: all estimates, 99th percentile clip.
For sigma_r all finite estimates are used, also those with r_hat < 0: keeping only the stable ones would cut the
distribution of r_hat and underestimate sigma_r.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mou_core import load_estimates, semicircle_spectrum, kappa_cW, rel_err_r

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style"))
import paperstyle as ps   # common style of the paper figures

REV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stability-inference-OU/

parser = argparse.ArgumentParser()
parser.add_argument("--q-goe", type=int, default=40)
parser.add_argument("--q-ginibre", type=int, default=10)
parser.add_argument("--tag", type=str, default="main")
parser.add_argument("--symmetrize", action="store_true", help="GOE: use the estimates made with --symmetrize")
parser.add_argument("--data-dir", type=str, default=os.path.join(REV_DIR, "results", "synthetic"))
parser.add_argument("--output-dir", type=str, default=os.path.join(REV_DIR, "figures", "other", "fig_ginibre_sigma_r"))
parser.add_argument("--format", type=str, nargs="+", default=["pdf", "png"])
parser.add_argument("--dpi", type=int, default=600)
parser.add_argument("--replot", action="store_true",
                    help="--panels: draw from the saved fig_ginibre_sigma_r_panels_plotdata.npz only")
parser.add_argument("--panels", action="store_true",
                    help="three panels as Fig. 2: q = 15, dt = 1, 0.5, 0.2 (tags ginq15dt1, ginq15dt05, ginq15dt02); "
                         "one point per A coloured by c, r_hat = min Re lambda(A_hat), no theory (fig_ginibre_sigma_r_panels)")
parser.add_argument("--projected", action="store_true",
                    help="with --cloud-goe: r_hat from the projection of Q_hat on the true soft mode, "
                         "q_hat = w_L Q_hat v_R with w_L v_R = 1 (as Fig. 2 for the GOE); suffix _proj")
parser.add_argument("--cloud-goe", action="store_true", dest="cloud_goe",
                    help="one Monte Carlo point per Ginibre A, with the GOE theory curves of Fig. 2 "
                         "(kappa_W, kappa_cW on the semicircle spectrum) overlaid (output fig_ginibre_sigma_r_cloud_goe)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


######################################################################################################################
# QUANTITIES
######################################################################################################################

def frobenius(d):
    """||A_hat - A||_F^2 / N, shape (n_c, n_A, n_sim); NaN where the estimate failed."""
    diff = d["A_hat"].astype(float) - d["A_true"][:, :, None].astype(float)
    return np.sum(diff ** 2, axis=(-2, -1)) / d["N"]


def soft_mode(d):
    """r per A, r_hat per trajectory (min real part of the spectrum), sigma_r / r per A."""
    r = np.real(np.linalg.eigvals(d["A_true"].astype(float))).min(-1)
    A_hat = d["A_hat"].astype(float)
    ok = np.isfinite(A_hat).all(axis=(-2, -1))
    r_hat = np.full(ok.shape, np.nan)
    r_hat[ok] = np.real(np.linalg.eigvals(A_hat[ok])).min(-1)
    return r, r_hat, np.nanstd(r_hat, axis=2, ddof=1) / r


######################################################################################################################
# FIGURE
######################################################################################################################

def frob_hist(ax, d, frob, name, logx=False):
    """logx: log-spaced bins and log x axis (Ginibre, heavy right tail)."""

    colors = cm.magma(np.linspace(0.15, 0.90, len(d["c_values"])))
    clip_max = np.nanpercentile(frob, 99)
    bins = np.logspace(np.log10(np.nanmin(frob)), np.log10(clip_max), 60) if logx else 60

    for i_c, c in enumerate(d["c_values"]):
        v = frob[i_c].ravel()
        v = v[np.isfinite(v) & (v <= clip_max)]
        ax.hist(v, bins=bins, density=True, histtype="step", linewidth=2, color=colors[i_c], label=f"c={c:g}")

    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(r"$\|\hat{A}-A\|_F^2/N$", fontsize=13)
    ax.set_ylabel("density", fontsize=13)
    ax.set_title(rf"{name} ($N={d['N']}$, $q={d['q']}$, $\Delta t={d['Delta_t']:g}$)", fontsize=12)
    ax.legend(frameon=False, fontsize=8)


def soft_mode_projected(d):
    """r per A and sigma_r / r per A with r_hat = -Re log(w_L Q_hat v_R) / dt, where v_R (w_L) is the right (left)
    eigenvector of the true A for the eigenvalue with the smallest real part, normalised so that w_L v_R = 1
    (rows of V^{-1}); first-order perturbation of that eigenvalue, the analogue of the GOE projection of Fig. 2."""
    n_c, n_A = d["A_true"].shape[:2]
    dt = d["Delta_t"]
    r = np.full((n_c, n_A), np.nan)
    rel = np.full((n_c, n_A), np.nan)
    for i in range(n_c):
        for j in range(n_A):
            lam, V = np.linalg.eig(d["A_true"][i, j].astype(float))
            k = np.argmin(lam.real)
            v_R, w_L = V[:, k], np.linalg.inv(V)[k]
            Q_hat = d["Q_hat"][i, j].astype(float)
            ok = np.isfinite(Q_hat).all(axis=(-2, -1))
            q_hat = np.einsum("a,kab,b->k", w_L, Q_hat[ok], v_R)
            r_hat = -np.log(q_hat).real / dt
            r[i, j] = lam[k].real
            rel[i, j] = np.nanstd(r_hat, ddof=1) / r[i, j]
    return r, rel


def cloud_goe_figure(gin, r, rel_err):
    """Ginibre sigma_r / r, one point per realisation of A, with the GOE predictions (same N, q, dt) for reference."""

    N, n, dt = gin["N"], gin["M"] - 1, gin["Delta_t"]
    c_fine = np.linspace(0.02, 0.75, 600)
    lam = [semicircle_spectrum(c, N) for c in c_fine]
    r_th = np.array([l[0] for l in lam])
    keep = r_th > 1e-3
    k_cw = np.array([float(kappa_cW(l, n, dt)[0]) for l in lam])
    k_w = n / (n - N - 1)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(r.ravel(), rel_err.ravel(), s=9, color="black", alpha=0.35, linewidths=0,
               label=r"Ginibre Monte Carlo (one point per $\mathbf{A}$)", zorder=3)
    ax.plot(r_th[keep], rel_err_r(r_th[keep], k_w, n, dt), "--", lw=2.0, color="firebrick", label=r"GOE theory, $\kappa_W$")
    ax.plot(r_th[keep], rel_err_r(r_th[keep], k_cw[keep], n, dt), "-.", lw=2.0, color="steelblue",
            label=r"GOE theory, $\kappa_{\rm cW}$")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(left=0)
    ax.set_xlabel(r"$r=\min{\rm Re}\,\lambda(\mathbf{A})$", fontsize=13)
    ax.set_ylabel(r"$\sigma_r/r$", fontsize=13)
    ax.set_title(rf"Ginibre, $N={N},\ q={gin['q']},\ \Delta t={dt:g}$"
                 + (r" (projection on the true soft mode)" if args.projected else r" ($\hat r=\min{\rm Re}\,\lambda(\hat{\mathbf{A}})$)"),
                 fontsize=12, loc="left")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    for fmt in args.format:
        path = os.path.join(args.output_dir, "fig_ginibre_sigma_r_cloud_goe" + ("_proj" if args.projected else "")
                            + f".{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=args.dpi)
        print(f"Saved: {path}")


Y_LIM_GIN = (0.0, 1.0)   # --panels: linear y range (axes only)

GIN_PANELS = [(15, "ginq15dt1"), (15, "ginq15dt05"), (15, "ginq15dt02")]


def panels_figure():
    """Paper Fig. 6: Ginibre sigma_r / r, one point per realisation of A, colour = c (colour bar), one panel per
    sampling interval; log y axis. Drawn data saved in fig_ginibre_sigma_r_panels_plotdata.npz (--replot: from it)."""

    cache = os.path.join(args.data_dir, "plotdata", "fig_ginibre_sigma_r_panels_plotdata.npz")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if args.replot:
        z = np.load(cache)
        P = [{k[3:]: z[k] for k in z.files if k.startswith(f"p{i}_")} for i in range(3)]
    else:
        P = []
        for q, tag in GIN_PANELS:
            d = load_estimates(args.data_dir, "GINIBRE", q, tag)
            r, _, rel = soft_mode(d)
            P.append({"q": q, "dt": d["Delta_t"], "c_values": d["c_values"], "r": r, "rel": rel})
        np.savez(cache, **{f"p{i}_{k}": v for i, p in enumerate(P) for k, v in p.items()})
        print(f"Saved: {cache}")

    with plt.style.context(ps.MAIN_CTX):
        fig, axes = plt.subplots(1, 3, figsize=(ps.W2, 2.35), layout="constrained", sharey=True)
        for ax, p, letter in zip(axes, P, "abc"):
            colors = ps.magma_c(len(p["c_values"]))
            for i_c in range(len(p["c_values"])):
                ax.scatter(p["r"][i_c], p["rel"][i_c], s=6, color=colors[i_c], alpha=0.8, linewidths=0)
            ax.set_ylim(*Y_LIM_GIN)
            ax.set_xlim(0, 1.0)
            ax.set_title(rf"$\Delta t={float(p['dt']):g}$", loc="right")
            ax.set_xlabel(r"$r=\min\,{\rm Re}\,\lambda(\mathbf{A})$")
            ps.panel_label(ax, letter)
            n_out = int(np.sum((p["rel"] > Y_LIM_GIN[1]) | (p["rel"] < Y_LIM_GIN[0])))
            if n_out:
                print(f"panel {letter}: {n_out} points outside the y range {Y_LIM_GIN}")
        axes[0].set_ylabel(r"$\sigma_r/r$")
        # panel titles show only what changes (dt); fixed N, q in the caption
        ps.discrete_colorbar(fig, axes, P[0]["c_values"], ps.magma_c(len(P[0]["c_values"])), r"$c$",
                             aspect=22, pad=0.015)
        for fmt in args.format:
            path = os.path.join(args.output_dir, f"fig_ginibre_sigma_r_panels.{fmt}")
            fig.savefig(path, dpi=args.dpi)
            print(f"Saved: {path}")



if __name__ == "__main__" and args.panels:
    panels_figure()
    raise SystemExit

if __name__ == "__main__" and args.cloud_goe:
    _gin = load_estimates(args.data_dir, "GINIBRE", args.q_ginibre, args.tag)
    if args.projected:
        _r, _rel = soft_mode_projected(_gin)
    else:
        _r, _, _rel = soft_mode(_gin)
    cloud_goe_figure(_gin, _r, _rel)
    raise SystemExit

if __name__ == "__main__":

    goe = load_estimates(args.data_dir, "GOE", args.q_goe, args.tag, args.symmetrize)
    gin = load_estimates(args.data_dir, "GINIBRE", args.q_ginibre, args.tag)

    frob_goe, frob_gin = frobenius(goe), frobenius(gin)
    r, r_hat, rel_err = soft_mode(gin)

    rows = [{"ensemble": "GOE", "c": c, "frob_mean": np.nanmean(frob_goe[i]), "frob_median": np.nanmedian(frob_goe[i])}
            for i, c in enumerate(goe["c_values"])]
    rows += [{"ensemble": "GINIBRE", "c": c, "frob_mean": np.nanmean(frob_gin[i]), "frob_median": np.nanmedian(frob_gin[i]),
              "r_mean": r[i].mean(), "frac_r_hat_neg": np.nanmean(r_hat[i] < 0), "rel_err_r_median": np.nanmedian(rel_err[i])}
             for i, c in enumerate(gin["c_values"])]
    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    summary.to_csv(os.path.join(args.output_dir, "fig_ginibre_sigma_r.csv"), index=False)

    # paper figure: the Ginibre sigma_r / r panel (Frobenius errors in the CSV)
    fig, ax = plt.subplots(figsize=(13, 4.75))
    colors = cm.magma(np.linspace(0.15, 0.90, len(gin["c_values"])))
    for i_c, c in enumerate(gin["c_values"]):
        lo, med, hi = np.nanpercentile(rel_err[i_c], [25, 50, 75])
        ax.errorbar(r[i_c].mean(), med, yerr=[[med - lo], [hi - med]], fmt="o", ms=7, capsize=3,
                    color=colors[i_c], label=f"c={c:g}")
    ax.set_xlabel(r"$\bar r$", fontsize=13)
    ax.set_ylabel(r"$\sigma_r/r$", fontsize=13)
    ax.set_title(r"Ginibre: $\hat r=\min{\rm Re}\,\lambda(\hat A)$ (median, 25-75% over $\mathbf{A}$)", fontsize=12)
    ax.legend(frameon=False, fontsize=8, ncol=2)

    for a in fig.axes:
        a.grid(True, alpha=0.25)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)

    fig.tight_layout()
    stem = "fig_ginibre_sigma_r" + ("_sym" if args.symmetrize else "")
    for fmt in args.format:
        path = os.path.join(args.output_dir, f"{stem}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=args.dpi)
        print(f"Saved: {path}")
