"""
Paper Fig. 1 (fig:spectra): eigenvalue spectra and condition numbers, GOE (rows 1-2) and Ginibre (rows 3-4).
Columns: true spectrum, estimated spectrum, condition number vs c (median and 25-75% over the realisations).
Rows: drift matrix A and Fisher information I.

Input: estimate.py files for GOE and Ginibre at the same q (paper: N = 100, q = 10, dt = 1, n_A = n_sim = 30).
  - spectra of A: real parts of the eigenvalues;
  - Fisher spectrum: eigenvalues of I / n from the eigenvalues of Sigma_inf and Sigma_dt (mou_core.fisher_eigs),
    histogrammed on log10 bins; the factor n does not change the condition number;
  - estimated quantities: stable reconstructions only, lambda_min(A_hat) >= 0 (caption);
  - CSV: medians per c, with kappa(I) * r_bar to check kappa(I) ~ 1/r_bar (main text, Results);
  - --fisher-block QQ: Fisher rows and condition numbers from the block I^(QQ) only (output suffix _QQ).
Diagnostic, not in the paper (fig1_eig_extremes): largest and smallest eigenvalue of A and of I / n vs c,
true and estimated, median and 25-75% over the realisations; fig1_eig_extremes_blocks: same for the blocks
I^(QQ) / n and I^(JJ) / n separately. Extremes of a Kronecker block are products of the extreme eigenvalues of
its factors (Sigma_inf, J = Sigma_dt^{-1}, Sigma_dt), so no 2N^2 spectrum is built for them.
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from mou_core import true_parameters, fisher_eigs, condition_number, load_estimates
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style"))
import paperstyle as ps   # common style of the paper figures

REV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stability-inference-OU/

parser = argparse.ArgumentParser()
parser.add_argument("--q", type=int, default=10)
parser.add_argument("--tag", type=str, default="main")
parser.add_argument("--symmetrize", action="store_true", help="use the estimates made with --symmetrize")
parser.add_argument("--data-dir", type=str, default=os.path.join(REV_DIR, "results", "synthetic"))
parser.add_argument("--output-dir", type=str, default=os.path.join(REV_DIR, "figures", "other", "fig1_spectra"))
parser.add_argument("--format", type=str, nargs="+", default=["pdf", "png"])
parser.add_argument("--dpi", type=int, default=600)
parser.add_argument("--replot", action="store_true",
                    help="draw the paper figure from the saved fig1_spectra<suffix>_plotdata.pkl only (no estimates loaded)")
parser.add_argument("--fisher-block", type=str, default="full", choices=["full", "QQ"], dest="fisher_block",
                    help="Fisher rows and condition numbers from the full I or from the block I^(QQ) only")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

BLOCK = args.fisher_block
SUFFIX = ("_sym" if args.symmetrize else "") + ("_QQ" if BLOCK == "QQ" else "")
I_LAB = r"\mathcal{I}" if BLOCK == "full" else r"\mathcal{I}^{(QQ)}"
I_HAT_LAB = r"\hat{\mathcal{I}}" if BLOCK == "full" else r"\hat{\mathcal{I}}^{(QQ)}"


######################################################################################################################
# SPECTRA AND CONDITION NUMBERS FROM THE ESTIMATES
######################################################################################################################

def spectra(d):
    """Eigenvalues of A, A_hat, Sigma_inf, Sigma_dt (true and estimated), stability mask, condition numbers."""

    n_c, n_A, n_sim, N = len(d["c_values"]), d["n_A"], d["n_sim"], d["N"]
    s = {
        "eig_A": np.empty((n_c, n_A, N), complex), "eig_Sinf": np.empty((n_c, n_A, N)), "eig_Sdt": np.empty((n_c, n_A, N)),
        "eig_A_hat": np.full((n_c, n_A, n_sim, N), np.nan, complex),
        "eig_Sinf_hat": np.full((n_c, n_A, n_sim, N), np.nan), "eig_Sdt_hat": np.full((n_c, n_A, n_sim, N), np.nan),
        "kappa_A": np.empty((n_c, n_A)), "kappa_I": np.empty((n_c, n_A)),
        "kappa_A_hat": np.full((n_c, n_A, n_sim), np.nan), "kappa_I_hat": np.full((n_c, n_A, n_sim), np.nan),
    }

    for i in range(n_c):
        for j in range(n_A):
            A = d["A_true"][i, j].astype(float)
            _, S_inf, S_dt = true_parameters(A, d["Delta_t"])
            s["eig_A"][i, j] = np.linalg.eigvals(A)
            s["eig_Sinf"][i, j], s["eig_Sdt"][i, j] = np.linalg.eigvalsh(S_inf), np.linalg.eigvalsh(S_dt)
            s["kappa_A"][i, j] = condition_number(s["eig_A"][i, j])
            s["kappa_I"][i, j] = condition_number(fisher_eigs(s["eig_Sinf"][i, j], s["eig_Sdt"][i, j], BLOCK))

            for k in range(n_sim):
                A_hat = d["A_hat"][i, j, k]
                if not np.isfinite(A_hat).all():
                    continue
                s["eig_A_hat"][i, j, k] = np.linalg.eigvals(A_hat.astype(float))
                s["eig_Sinf_hat"][i, j, k] = np.linalg.eigvalsh(d["Sigma_inf_hat"][i, j, k].astype(float))
                s["eig_Sdt_hat"][i, j, k] = np.linalg.eigvalsh(d["Sigma_dt_hat"][i, j, k].astype(float))

    s["r"] = np.real(s["eig_A"]).min(-1)
    s["stable_hat"] = np.real(s["eig_A_hat"]).min(-1) >= 0          # False also where the estimate failed (NaN)

    for i, j, k in np.argwhere(s["stable_hat"]):
        s["kappa_A_hat"][i, j, k] = condition_number(s["eig_A_hat"][i, j, k])
        s["kappa_I_hat"][i, j, k] = condition_number(fisher_eigs(s["eig_Sinf_hat"][i, j, k], s["eig_Sdt_hat"][i, j, k],
                                                                 BLOCK))

    return s


def extremes(s):
    """Largest and smallest eigenvalue of A (real part) and of I / n, per realisation; estimated: stable only.
    Extremes of fisher_eigs: products of the extreme eigenvalues of Sigma_inf and Sigma_dt."""

    def block_ext(S_inf, S_dt):
        """Extremes of the two Kronecker blocks: QQ = Sigma_inf (x) J, JJ = (1/2) Sigma_dt (x) Sigma_dt."""
        return {"QQ_max": S_inf.max(-1) / S_dt.min(-1), "QQ_min": S_inf.min(-1) / S_dt.max(-1),
                "JJ_max": 0.5 * S_dt.max(-1) ** 2, "JJ_min": 0.5 * S_dt.min(-1) ** 2}

    e = {}
    eA, eA_hat = np.real(s["eig_A"]), np.where(s["stable_hat"][..., None], np.real(s["eig_A_hat"]), np.nan)
    e["A_max"], e["A_min"] = eA.max(-1), eA.min(-1)
    e["A_hat_max"], e["A_hat_min"] = eA_hat.max(-1), eA_hat.min(-1)

    for est, (k_inf, k_dt) in [("", ("eig_Sinf", "eig_Sdt")), ("_hat", ("eig_Sinf_hat", "eig_Sdt_hat"))]:
        b = block_ext(s[k_inf], s[k_dt])
        if est:
            b = {k: np.where(s["stable_hat"], v, np.nan) for k, v in b.items()}
        for blk in ["QQ", "JJ"]:
            e[f"{blk}{est}_max"], e[f"{blk}{est}_min"] = b[f"{blk}_max"], b[f"{blk}_min"]
        e[f"I{est}_max"] = np.maximum(b["QQ_max"], b["JJ_max"])
        e[f"I{est}_min"] = np.minimum(b["QQ_min"], b["JJ_min"])

    return e


def plot_extremes_blocks(runs):
    """Extremes of the two blocks of I / n, one panel per ensemble with all 8 curves:
    colour = block (QQ, JJ), marker = max / min, dashed = true, solid = estimated."""

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    block_colors = {"QQ": ("#b34a6f", "#6d0f2d"), "JJ": ("#6baed6", "#08519c")}

    for ax, (name, d, s) in zip(axes, runs):
        e = extremes(s)
        c_values = d["c_values"]
        for blk in ["QQ", "JJ"]:
            lab = rf"\mathcal{{I}}^{{({blk})}}"
            hat_lab = rf"\hat{{\mathcal{{I}}}}^{{({blk})}}"
            for which, marker in [("max", "o"), ("min", "s")]:
                for est, color, ls in [(False, block_colors[blk][0], "--"), (True, block_colors[blk][1], "-")]:
                    lo, med, hi = q25_med_q75(e[f"{blk}{'_hat' if est else ''}_{which}"])
                    ax.fill_between(c_values, lo, hi, color=color, alpha=0.12)
                    ax.plot(c_values, med, ls, marker=marker, ms=4, lw=1.4, color=color,
                            label=rf"$\mu_{{\rm {which}}}({hat_lab if est else lab}/n)$")
        ax.set_yscale("log")
        ax.set_xlabel(r"$c$")
        ax.set_ylabel(r"$\mu$")
        ax.set_title(rf"{name}: extreme eigenvalues of the Fisher blocks ($q={d['q']}$)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.tight_layout()
    stem = "fig1_eig_extremes_blocks" + ("_sym" if args.symmetrize else "")
    for fmt in args.format:
        path = os.path.join(args.output_dir, f"{stem}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=args.dpi)
        print(f"Saved: {path}")
    plt.close(fig)


def plot_extremes(runs):

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    styles = [("max", "o", r"\max"), ("min", "s", r"\min")]

    for row, (name, d, s) in enumerate(runs):
        e = extremes(s)
        c_values = d["c_values"]
        for col, mat, lab in [(0, "A", "A"), (1, "I", r"\mathcal{I}")]:
            ax = axes[row, col]
            for which, marker, op in styles:
                for est, color, ls in [(False, "#b34a6f", "--"), (True, "#6d0f2d", "-")]:
                    key = f"{mat}{'_hat' if est else ''}_{which}"
                    lo, med, hi = q25_med_q75(e[key])
                    hat = r"\hat" if est else ""
                    ax.fill_between(c_values, lo, hi, color=color, alpha=0.15)
                    ax.plot(c_values, med, ls, marker=marker, ms=4, lw=1.4, color=color,
                            label=rf"${op}\,{hat}{{{lab}}}$" if mat == "A" else rf"$\mu_{{\rm {which}}}({hat}{{{lab}}}/n)$")
            ax.set_yscale("log")
            ax.set_xlabel(r"$c$")
            ax.set_ylabel(r"${\rm Re}\,\lambda$" if mat == "A" else r"$\mu$")
            ax.set_title(rf"{name}: extreme eigenvalues of ${lab}$" + ("" if mat == "A" else "$/n$") + f" ($q={d['q']}$)")
            ax.grid(alpha=0.25, which="both")
            ax.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    stem = "fig1_eig_extremes" + ("_sym" if args.symmetrize else "")
    for fmt in args.format:
        path = os.path.join(args.output_dir, f"{stem}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=args.dpi)
        print(f"Saved: {path}")
    plt.close(fig)


def spectrum_A(s, i_c, estimated):
    if not estimated:
        return np.real(s["eig_A"][i_c]).ravel()
    return np.real(s["eig_A_hat"][i_c][s["stable_hat"][i_c]]).ravel()


def hist_fisher(s, i_c, estimated, bins):
    """Density on log10 bins of the spectrum of I / n, accumulated realisation by realisation."""

    if not estimated:
        pairs = zip(s["eig_Sinf"][i_c], s["eig_Sdt"][i_c])
    else:
        mask = s["stable_hat"][i_c]
        pairs = zip(s["eig_Sinf_hat"][i_c][mask], s["eig_Sdt_hat"][i_c][mask])

    counts = np.zeros(len(bins) - 1)
    for S_inf, S_dt in pairs:
        e = fisher_eigs(S_inf, S_dt, BLOCK)
        counts += np.histogram(np.log10(e[np.isfinite(e) & (e > 0)]), bins=bins)[0]

    return counts / (counts.sum() * np.diff(bins)) if counts.sum() > 0 else counts


def fisher_bins(*all_s, n_bins=120):
    """Common log10 bins: extremes of the products in fisher_eigs, 0.5-99.5 percentiles over realisations."""

    lo, hi = [], []
    for s in all_s:
        for k_inf, k_dt in [("eig_Sinf", "eig_Sdt"), ("eig_Sinf_hat", "eig_Sdt_hat")]:
            S_inf, S_dt = s[k_inf], s[k_dt]
            e_min, e_max = S_inf.min(-1) / S_dt.max(-1), S_inf.max(-1) / S_dt.min(-1)          # QQ block
            if BLOCK == "full":
                e_min = np.minimum(e_min, 0.5 * S_dt.min(-1) ** 2)
                e_max = np.maximum(e_max, 0.5 * S_dt.max(-1) ** 2)
            lo.append(np.nanpercentile(np.log10(e_min[e_min > 0]), 0.5))
            hi.append(np.nanpercentile(np.log10(e_max[e_max > 0]), 99.5))
    return np.linspace(min(lo), max(hi), n_bins + 1)


def q25_med_q75(x):
    x = x.reshape(x.shape[0], -1)
    return np.nanpercentile(x, 25, axis=1), np.nanmedian(x, axis=1), np.nanpercentile(x, 75, axis=1)


######################################################################################################################
# FIGURE
######################################################################################################################

def plot_data(runs):
    """Everything Fig. 1 draws, per ensemble: spectra of A (true, stable estimates), Fisher histograms on common
    log10 bins, condition-number quartiles."""

    bins = fisher_bins(*(s for _, _, s in runs))
    out = {"bins": bins, "ensembles": []}
    for name, d, s in runs:
        c_values = d["c_values"]
        e = {"name": name, "q": d["q"], "c_values": c_values,
             "spec_A": [[spectrum_A(s, i_c, est) for est in (False, True)] for i_c in range(len(c_values))],
             "hist_I": [[hist_fisher(s, i_c, est, bins) for est in (False, True)] for i_c in range(len(c_values))]}
        for key in ["kappa_A", "kappa_A_hat", "kappa_I", "kappa_I_hat"]:
            e[key] = q25_med_q75(s[key])
        out["ensembles"].append(e)
    return out


def plot(P):
    """Paper Fig. 1, 183 mm wide: rows GOE A, GOE I, Ginibre A, Ginibre I; columns true spectrum, estimated
    spectrum, condition numbers. Colours of c: magma, one colour bar per ensemble in place of the
    legends; True / Estimated legend once."""

    bins = P["bins"]
    centers = 0.5 * (bins[:-1] + bins[1:])
    bordeaux_dark, bordeaux_light = "#6d0f2d", "#b34a6f"

    with plt.style.context(ps.MAIN_CTX):
        fig = plt.figure(figsize=(ps.W2, 7.4), layout="constrained")
        gs = fig.add_gridspec(4, 4, width_ratios=[1, 1, 0.055, 1.0])
        axes = np.array([[fig.add_subplot(gs[r, c]) for c in (0, 1, 3)] for r in range(4)])

        for row0, e in zip([0, 2], P["ensembles"]):
            name, c_values = e["name"], e["c_values"]
            colors = ps.magma_c(len(c_values))

            for i_c, c in enumerate(c_values):
                for col, est in [(0, False), (1, True)]:
                    axes[row0, col].hist(e["spec_A"][i_c][est], bins=100, density=True, histtype="step",
                                         linewidth=1.0, color=colors[i_c])
                    axes[row0 + 1, col].step(10 ** centers, e["hist_I"][i_c][est], where="mid",
                                             linewidth=1.0, color=colors[i_c])

            for row, k_true, k_est in [(row0, "kappa_A", "kappa_A_hat"), (row0 + 1, "kappa_I", "kappa_I_hat")]:
                ax = axes[row, 2]
                for key, color, ls, marker, label in [(k_true, bordeaux_light, "--", "o", "true"),
                                                      (k_est, bordeaux_dark, "-", "s", "estimated")]:
                    lo, med, hi = e[key]
                    ax.fill_between(c_values, lo, hi, color=color, alpha=0.15, linewidth=0)
                    ax.plot(c_values, med, ls, linewidth=1.25, marker=marker, markersize=3.5, color=color, label=label)
                ax.set_yscale("log")
                ax.set_xlabel(r"$c$")
                ax.set_ylabel("condition number")

            sub = r"\mathrm{GOE}" if name == "GOE" else r"\mathrm{Gin}"
            axes[row0, 0].set_title(rf"$A^*_{{{sub}}}$", loc="right")
            axes[row0, 1].set_title(rf"$\hat{{A}}_{{{sub}}}$", loc="right")
            axes[row0, 2].set_title(rf"$\kappa(A_{{{sub}}})$, $\kappa(\hat{{A}}_{{{sub}}})$", loc="right")
            axes[row0 + 1, 0].set_title(rf"${I_LAB}_{{{sub}}}/n$", loc="right")
            axes[row0 + 1, 1].set_title(rf"${I_HAT_LAB}_{{{sub}}}/n$", loc="right")
            axes[row0 + 1, 2].set_title(rf"$\kappa({I_LAB}_{{{sub}}})$, $\kappa({I_HAT_LAB}_{{{sub}}})$", loc="right")

            A_all = np.concatenate([a for pair in e["spec_A"] for a in pair])
            xmin_A = A_all.min() - 0.05 * (A_all.max() - A_all.min())
            for col in [0, 1]:
                axes[row0, col].set_xlim(xmin_A, 4)
                axes[row0, col].set_xlabel(r"$\lambda$" if name == "GOE" else r"${\rm Re}\,\lambda$")
                axes[row0 + 1, col].set_xscale("log")
                axes[row0 + 1, col].set_xlim(10 ** bins[0], 1e2)
                axes[row0 + 1, col].set_xlabel(r"$\mu$")
            axes[row0, 0].set_ylabel("density")
            axes[row0 + 1, 0].set_ylabel(r"density in $\log_{10}\mu$")
            for row in [row0, row0 + 1]:
                ymax = max(axes[row, 0].get_ylim()[1], axes[row, 1].get_ylim()[1])
                axes[row, 0].set_ylim(0, ymax)
                axes[row, 1].set_ylim(0, ymax)

            cax = fig.add_subplot(gs[row0:row0 + 2, 2])
            cb = ps.discrete_colorbar(fig, None, c_values, colors, r"$c$", cax=cax)

        axes[0, 2].legend(loc="upper left")

        stem = "fig1_spectra" + SUFFIX
        for fmt in args.format:
            path = os.path.join(args.output_dir, f"{stem}.{fmt}")
            fig.savefig(path, dpi=args.dpi)
            print(f"Saved: {path}")
        plt.close(fig)



if __name__ == "__main__" and args.replot:
    with open(os.path.join(args.data_dir, "plotdata", "fig1_spectra" + SUFFIX + "_plotdata.pkl"), "rb") as f:
        plot(pickle.load(f))
    raise SystemExit

if __name__ == "__main__":

    runs, rows = [], []
    for ensemble, name in [("GOE", "GOE"), ("GINIBRE", "Gin")]:
        d = load_estimates(args.data_dir, ensemble, args.q, args.tag, args.symmetrize)
        s = spectra(d)
        runs.append((name, d, s))
        del d["A_hat"], d["Q_hat"], d["Sigma_dt_hat"], d["Sigma_inf_hat"]

        for i_c, c in enumerate(d["c_values"]):
            rows.append({
                "ensemble": ensemble, "c": c, "r_mean": s["r"][i_c].mean(),
                "frac_unstable_hat": 1 - s["stable_hat"][i_c].mean(),
                "kappa_A_med": np.nanmedian(s["kappa_A"][i_c]), "kappa_A_hat_med": np.nanmedian(s["kappa_A_hat"][i_c]),
                "kappa_I_med": np.nanmedian(s["kappa_I"][i_c]), "kappa_I_hat_med": np.nanmedian(s["kappa_I_hat"][i_c]),
                "kappa_I_times_r": np.nanmedian(s["kappa_I"][i_c] * s["r"][i_c]),
            })

    summary = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(summary.round(4).to_string(index=False))
    summary.to_csv(os.path.join(args.output_dir, "fig1_spectra" + SUFFIX + ".csv"), index=False)

    P = plot_data(runs)
    os.makedirs(os.path.join(args.data_dir, "plotdata"), exist_ok=True)
    with open(os.path.join(args.data_dir, "plotdata", "fig1_spectra" + SUFFIX + "_plotdata.pkl"), "wb") as f:
        pickle.dump(P, f)
    plot(P)
    if BLOCK == "full":
        plot_extremes(runs)
        plot_extremes_blocks(runs)
