"""
Paper Fig. 2 (fig:projected_soft_mode_uncertainty): relative uncertainty of the soft mode, sigma_r / r, vs r.

PAPER FIGURES (see run.sh):
  main Fig. 2 = --panels --q 15 --line-colors purple_magenta --debias
  SI Fig. 2   = the same plus --tau-w
  Three panels, GOE, N = 100, q = 15, n_A = n_sim = 30, c = 0.1 ... 0.7, dt = 1, 0.5, 0.2 (tags q15dt1, q15dt05,
  q15dt02). One Monte Carlo point per realisation of A: std over trajectories of the bias-corrected r_hat
  (r_hat - bias_r) / true r. Lines: sigma_r / r with kappa_W and kappa_cW on the typical GOE spectrum
  (semicircle_spectrum). Linear axes. See panels_figure().

The rest of this description is about the single-panel mode (no --panels). Its input: estimate.py files for the GOE (N = 100, q = 10, dt = 1, n_A = n_sim = 30), runs main
(c = 0.1, 0.2, ... 0.7) and step05 (c = 0.15, 0.25, ... 0.65).
  - Monte Carlo: r = lambda_min(A), u_* its eigenvector, r_hat = -log(u_*^T Q_hat u_*) / dt per trajectory;
    per realisation of A, std of r_hat over trajectories / r; then median and 25-75% over A, one point per c.
    --stat mean: mean and std over A, and kappa_cW averaged only over the A
    with M_eff > N + 2 (the others are discarded).
    All finite estimates with u_*^T Q_hat u_* > 0 are used.
  - Theory, per realisation of A (Eq. var_lambda, Methods), then median over A:
        sigma_r / r = sqrt(kappa (e^{2 r dt} - 1) / (n r^2 dt^2)),  n = M - 1,
    with
        kappa_W  = (M-1) / (M-N-2)                                         (independent data)
        kappa_r  = 1 + [sum_{b != r} tau_rb + 2 tau_r] / n                 (first order, Eq. kappa_r)
                   tau_rb = (1 + q_r q_b) / (1 - q_r q_b),  q_a = exp(-lambda_a dt)
        kappa_cW = (M_eff-1) / (M_eff-N-2),  M_eff = n / tau_bar + 1,  tau_bar = (1/N) sum_a tau_a (arithmetic mean)
                   (infinite where n_eff <= N + 1, undefined); q_eff = n / (tau_bar N)
        kappa_r is computed for the CSV only.
        kappa_bd = kappa_W (1 + q_r) / (1 - q_r)                           (exact upper bound on kappa_r, SI Step 7;
                   drawn with --bound)
  - CSV: per c, median of r, Monte Carlo and the three predictions, kappa's and q_eff.
--estimated: no knowledge of the true A (as for real data). Per trajectory, r_hat = min Re lambda(A_hat) (estimated
eigenvalue and eigenvector), and the three predictions are evaluated by plug-in, with the real parts of the eigenvalues
of A_hat in place of lambda_a and r_hat in place of r. Only trajectories with r_hat > 0 are kept (frac_neg = fraction
with r_hat <= 0); per realisation of A, Monte Carlo = std of r_hat over them / true r, predictions = their median.
Output suffix _est.
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mou_core import load_estimates, bias_r, kappa_cW as kappa_cW_core, semicircle_spectrum, rel_err_r
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style"))
import paperstyle as ps   # common style of the paper figures

REV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stability-inference-OU/

parser = argparse.ArgumentParser()
parser.add_argument("--q", type=int, default=10)
parser.add_argument("--tags", type=str, nargs="+", default=["main", "step05"],
                    help="runs to combine: main (c = 0.1, 0.2, ... 0.7, as Fig. 1), step05 (c = 0.15, 0.25, ... 0.65, "
                         "), fine (c = 0.62 ... 0.68)")
parser.add_argument("--symmetrize", action="store_true", help="use the estimates made with --symmetrize")
parser.add_argument("--data-dir", type=str, default=os.path.join(REV_DIR, "results", "synthetic"))
parser.add_argument("--output-dir", type=str, default=os.path.join(REV_DIR, "figures", "other", "fig2_soft_mode"))
parser.add_argument("--estimated", action="store_true",
                    help="r_hat and the predictions from the eigenvalues of A_hat only (output suffix _est)")
parser.add_argument("--stat", type=str, default="median", choices=["median", "mean", "median_se"],
                    help="over A: median and 25-75%%, or mean and std (output suffix _mean)")
parser.add_argument("--bound", action="store_true",
                    help="also draw the exact upper bound kappa_r <= kappa_W (1 + q_r) / (1 - q_r) (output suffix _bound)")
parser.add_argument("--tau-variants", action="store_true", dest="tau_variants",
                    help="also kappa_cW with tau_w = sum tau^2 / sum tau and with tau_max (suffix _tauvar)")
parser.add_argument("--lmin", action="store_true",
                    help="--panels: Monte Carlo from r_hat = min Re lambda(A_hat), no projection, no bias "
                         "correction, all finite estimates (suffix _lmin)")
parser.add_argument("--debias", action="store_true",
                    help="--panels: Monte Carlo of the bias-corrected r_tilde = r_hat - b_r (mou_core.bias_r; suffix _debias)")
parser.add_argument("--tau-w", action="store_true", dest="tau_w",
                    help="--panels: also kappa_cW with the weighted mean tau_w = sum tau^2 / sum tau in place of tau_bar (suffix _tauw)")
parser.add_argument("--line-colors", type=str, default="black_teal", choices=["black_teal", "purple_magenta"],
                    dest="line_colors", help="--panels: colours of the kappa_W / kappa_cW lines (suffix _pm for purple_magenta)")
parser.add_argument("--panels", action="store_true",
                    help="three panels side by side, q = 10 (or --q 15: tags q15dt1, q15dt05, q15dt02): dt = 1 (tags main+step05), dt = 0.5 (tag q10dt05), dt = 0.2 (tag q10dt02); "
                         "one Monte Carlo point per A, kappa_W and kappa_cW on the typical GOE spectrum (output fig2_soft_mode_panels)")
parser.add_argument("--cloud", action="store_true",
                    help="Monte Carlo as one point per realisation of A instead of median and 25-75%% (suffix _cloud)")
parser.add_argument("--log", action="store_true", help="log-log axes (suffix _log)")
parser.add_argument("--logy", action="store_true", help="log y axis only (suffix _logy)")
parser.add_argument("--format", type=str, nargs="+", default=["pdf", "png"])
parser.add_argument("--dpi", type=int, default=600)
parser.add_argument("--replot", action="store_true",
                    help="--panels: draw from the saved <stem>_plotdata.npz only (no estimates loaded, no CSV)")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


######################################################################################################################
# INFLATION FACTORS
######################################################################################################################

def kappa_factors(lam, N, M, delta_t):
    """kappa_W, kappa_r (first order), kappa_cW and q_eff (of kappa_cW) for one A with eigenvalues lam (ascending)."""

    n = M - 1
    qa = np.exp(-lam * delta_t)
    q_r, q_b = qa[0], qa[1:]

    tau_rb = (1 + q_r * q_b) / (1 - q_r * q_b)
    tau_r = (1 + q_r ** 2) / (1 - q_r ** 2)
    kappa_r = 1 + (tau_rb.sum() + 2 * tau_r) / n

    kappa_W = (M - 1) / (M - N - 2)

    kappa_cW, q_eff = kappa_cW_core(lam, n, delta_t)
    kappa_cW = float(kappa_cW) if np.isfinite(kappa_cW) else np.inf   # undefined -> infinite (median over A)

    kappa_bd = kappa_W * (1 + q_r) / (1 - q_r)

    return kappa_W, kappa_r, kappa_cW, q_eff, kappa_bd


######################################################################################################################
# MONTE CARLO AND THEORY
######################################################################################################################

def soft_mode(d):
    """Per realisation of A: r, sigma_r / r (Monte Carlo) and the three predictions; kappa's and q_eff."""

    n_c, n_A, N, M, dt = len(d["c_values"]), d["n_A"], d["N"], d["M"], d["Delta_t"]
    n = M - 1
    out = {k: np.full((n_c, n_A), np.nan) for k in
           ["r", "mc", "th_W", "th_r", "th_cW", "th_bd", "kappa_W", "kappa_r", "kappa_cW", "q_eff",
            "th_cWw", "th_cWmax"]}

    for i in range(n_c):
        for j in range(n_A):
            A = d["A_true"][i, j].astype(float)
            lam, U = np.linalg.eigh(A)
            r, u = lam[0], U[:, 0]

            Q_hat = d["Q_hat"][i, j].astype(float)
            ok = np.isfinite(Q_hat).all(axis=(-2, -1))
            q_hat = np.einsum("a,kab,b->k", u, Q_hat[ok], u)
            r_hat = -np.log(q_hat[q_hat > 0]) / dt

            kW, kr, kcW, qe, kbd = kappa_factors(lam, N, M, dt)
            base = (np.exp(2 * r * dt) - 1) / (n * r ** 2 * dt ** 2)

            out["r"][i, j] = r
            out["mc"][i, j] = np.std(r_hat, ddof=1) / r
            out["th_W"][i, j], out["th_r"][i, j], out["th_cW"][i, j], out["th_bd"][i, j] = (
                np.sqrt(np.array([kW, kr, kcW, kbd]) * base))
            out["kappa_W"][i, j], out["kappa_r"][i, j], out["kappa_cW"][i, j], out["q_eff"][i, j] = kW, kr, kcW, qe
            # test variants of the effective correlation time (same kappa = n_eff / (n_eff - N - 1), n_eff = n / tau)
            qa2 = np.exp(-2 * lam * dt)
            tau = (1 + qa2) / (1 - qa2)
            for key, t in [("th_cWw", np.sum(tau ** 2) / np.sum(tau)), ("th_cWmax", tau.max())]:
                ne = n / t
                out[key][i, j] = np.sqrt(ne / (ne - N - 1) * base) if ne > N + 1 else np.inf

    return out


def debiased_mc(d):
    """Per realisation of A: std over trajectories of r_tilde = r_hat - b_r / r, with r_hat from the projection on
    the true soft mode (as soft_mode) and b_r from mou_core.bias_r on the eigenvalues of Q_hat (no knowledge of A)."""

    n_c, n_A, n_sim, dt, n = len(d["c_values"]), d["n_A"], d["n_sim"], d["Delta_t"], d["M"] - 1
    out = np.full((n_c, n_A), np.nan)
    for i in range(n_c):
        for j in range(n_A):
            lam, U = np.linalg.eigh(d["A_true"][i, j].astype(float))
            u = U[:, 0]
            r_t = np.full(n_sim, np.nan)
            for k in range(n_sim):
                Qh = d["Q_hat"][i, j, k].astype(float)
                if not np.isfinite(Qh).all():
                    continue
                q_r = u @ Qh @ u
                if q_r <= 0:
                    continue
                r_t[k] = -np.log(q_r) / dt - bias_r(q_r, np.linalg.eigvals(Qh), n, dt)
            out[i, j] = np.nanstd(r_t, ddof=1) / lam[0]
    return out


def soft_mode_estimated(d):
    """As soft_mode, without the true A: r_hat = min Re lambda(A_hat) and plug-in predictions per trajectory,
    median over the trajectories with r_hat > 0; true r only to normalise the Monte Carlo and place the point."""

    n_c, n_A, n_sim, N, M, dt = len(d["c_values"]), d["n_A"], d["n_sim"], d["N"], d["M"], d["Delta_t"]
    n = M - 1
    keys = ["th_W", "th_r", "th_cW", "kappa_W", "kappa_r", "kappa_cW", "q_eff"]
    out = {k: np.full((n_c, n_A), np.nan) for k in ["r", "mc", "frac_neg"] + keys}

    for i in range(n_c):
        for j in range(n_A):
            r = np.linalg.eigvalsh(d["A_true"][i, j].astype(float))[0]
            per_traj = {k: np.full(n_sim, np.nan) for k in keys}
            r_hat = np.full(n_sim, np.nan)

            for k in range(n_sim):
                A_hat = d["A_hat"][i, j, k].astype(float)
                if not np.isfinite(A_hat).all():
                    continue
                lam_hat = np.sort(np.real(np.linalg.eigvals(A_hat)))
                r_hat[k] = lam_hat[0]
                if r_hat[k] <= 0:
                    continue

                kW, kr, kcW, qe, _ = kappa_factors(lam_hat, N, M, dt)
                base = (np.exp(2 * r_hat[k] * dt) - 1) / (n * r_hat[k] ** 2 * dt ** 2)
                (per_traj["th_W"][k], per_traj["th_r"][k], per_traj["th_cW"][k]) = np.sqrt(np.array([kW, kr, kcW]) * base)
                per_traj["kappa_W"][k], per_traj["kappa_r"][k], per_traj["kappa_cW"][k], per_traj["q_eff"][k] = kW, kr, kcW, qe

            out["r"][i, j] = r
            out["frac_neg"][i, j] = np.mean(r_hat[np.isfinite(r_hat)] <= 0)
            out["mc"][i, j] = np.nanstd(r_hat[r_hat > 0], ddof=1) / r
            for key in keys:
                out[key][i, j] = np.nanmedian(per_traj[key])

    return out


######################################################################################################################
# FIGURE
######################################################################################################################

def plot(r, mc, lo, hi, med, order, N, q, dt, stem, bound=False, cloud=None, log=False, logy=False):
    """Monte Carlo (median and 25-75%, or mean and std, over A) and the three predictions vs r."""

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    if cloud is None:
        ax.errorbar(r, mc, yerr=[mc - lo["mc"][order], hi["mc"][order] - mc], fmt="o", ms=6, capsize=3,
                    elinewidth=1.2, color="black", label="Monte Carlo", zorder=3)
    else:
        ax.scatter(cloud["r"].ravel(), cloud["mc"].ravel(), s=9, color="black", alpha=0.35, linewidths=0,
                   label=r"Monte Carlo (one point per $\mathbf{A}$)", zorder=3)
        y_max = np.inf if (log or logy) else 1.0
        n_out = int(np.sum(cloud["mc"] > y_max))
        if not (log or logy):
            ax.set_ylim(0, y_max)
        if n_out:
            ax.text(0.98, 0.02, f"{n_out} points above {y_max:g} not shown", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=9, color="gray")
    if cloud is None:
        x_th, th = r, {k: med[k][order] for k in ["th_W", "th_r", "th_cW", "th_bd"] if k in med}
    else:
        # predictions per realisation of A, sorted by r, so that they cover the same range as the cloud
        o = np.argsort(cloud["r"].ravel())
        x_th, th = cloud["r"].ravel()[o], {k: cloud[k].ravel()[o] for k in ["th_W", "th_r", "th_cW", "th_bd"]}
    ax.plot(x_th, th["th_W"], "--", lw=2.0, color="firebrick", label=r"$\kappa_W$")
    ax.plot(x_th, th["th_cW"], "-.", lw=2.0, color="steelblue", label=r"$\kappa_{\rm cW}$")
    if args.tau_variants:
        ax.plot(x_th, med["th_cWw"][order], ":", lw=2.0, color="purple",
                label=r"$\kappa_{\rm cW}$, $\tau_w=\sum\tau^2/\sum\tau$")
        ax.plot(x_th, med["th_cWmax"][order], "-", lw=2.0, color="seagreen", label=r"$\kappa_{\rm cW}$, $\tau_{\max}$")
    if bound:
        ax.plot(x_th, th["th_bd"], ":", lw=2.0, color="gray",
                label=r"bound $\kappa_W(1+q_r)/(1-q_r)$")

    ax.set_xlabel(r"$r$", fontsize=13)
    ax.set_ylabel(r"$\sigma_r/r$", fontsize=13)
    ax.set_title(rf"Relative soft-mode uncertainty ($N={N},\ q={q},\ \Delta t={dt:g}$)", fontsize=12)
    if log:
        ax.set_xscale("log")
    if log or logy:
        ax.set_yscale("log")
    ax.grid(True, which="both" if (log or logy) else "major", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(frameon=False, fontsize=10)

    fig.tight_layout()
    for fmt in args.format:
        path = os.path.join(args.output_dir, f"{stem}.{fmt}")
        fig.savefig(path, bbox_inches="tight", dpi=args.dpi)
        print(f"Saved: {path}")


PANELS = {10: [(10, ["main", "step05"]), (10, ["q10dt05"]), (10, ["q10dt02"])],
          15: [(15, ["q15dt1"]), (15, ["q15dt05"]), (15, ["q15dt02"])]}


Y_LIM = (0.0, 1.0)   # --panels: linear y range of sigma_r / r (axes only)

LINE_COLORS = {"black_teal": ("black", "#00897b"), "purple_magenta": ("#4b0082", "#c2185b")}


def panels_figure():
    """One Monte Carlo point per realisation of A, with kappa_W and kappa_cW on the typical GOE spectrum, for the
    three runs of PANELS. The drawn data are saved in <stem>_plotdata.npz; --replot draws from that file only
    (no estimates loaded). Paper style (183 mm): colours of c in a colour bar, linear axes (log y only with --logy),
    legend once."""

    stem = ("fig2_soft_mode_panels" + ("" if args.q == 10 else f"_q{args.q}")
            + ("_logy" if args.logy else "") + ("_pm" if args.line_colors == "purple_magenta" else "")
            + ("_tauw" if args.tau_w else "") + ("_debias" if args.debias else "") + ("_lmin" if args.lmin else ""))
    cache = os.path.join(args.data_dir, "plotdata", stem + "_plotdata.npz")
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    if args.replot:
        z = np.load(cache)
        P = [{k[3:]: z[k] for k in z.files if k.startswith(f"p{i}_")} for i in range(3)]
    else:
        P, rows = [], []
        for q, tags in PANELS[args.q]:
            runs = [load_estimates(args.data_dir, "GOE", q, tag) for tag in tags]
            d = dict(runs[0])
            for k in ["A_true", "A_hat", "Q_hat", "c_values"]:
                d[k] = np.concatenate([run[k] for run in runs])
            s = soft_mode(d)
            if args.debias:
                s["mc"] = debiased_mc(d)
            if args.lmin:
                # r_hat = smallest real part of the eigenvalues of A_hat (as for the data); std over trajectories / r
                A_hat = d["A_hat"].astype(float)
                ok = np.isfinite(A_hat).all(axis=(-2, -1))
                r_l = np.full(ok.shape, np.nan)
                r_l[ok] = np.real(np.linalg.eigvals(A_hat[ok])).min(-1)
                s["mc"] = np.nanstd(r_l, axis=2, ddof=1) / s["r"]
            lo, med, hi = ({k: np.nanpercentile(v, p, axis=1) for k, v in s.items()} for p in (25, 50, 75))
            o = np.argsort(med["r"])
            # theory on the typical GOE spectrum (as Fig. 3): soft mode at r_bar(c, N), the rest on the semicircle,
            # c on a fine grid; smooth in r. kappa_cW undefined (n_eff <= N + 1) -> not drawn.
            N_, n_, dt_ = d["N"], d["M"] - 1, d["Delta_t"]
            c_fine = np.linspace(0.02, 0.75, 600)
            lam_list = [semicircle_spectrum(c, N_) for c in c_fine]
            r_th = np.array([lam[0] for lam in lam_list])
            keep = r_th > 1e-3
            k_cw = np.array([float(kappa_cW_core(lam, n_, dt_)[0]) for lam in lam_list])
            k_w = n_ / (n_ - N_ - 1)
            p = {"q": q, "dt": d["Delta_t"], "c_values": d["c_values"], "r": s["r"], "mc": s["mc"],
                 "r_th": r_th[keep], "th_W": rel_err_r(r_th[keep], k_w, n_, dt_),
                 "th_cW": rel_err_r(r_th[keep], k_cw[keep], n_, dt_)}
            if args.tau_w:
                # same kappa = n_eff / (n_eff - N - 1), n_eff = n / tau_w, tau_w = sum tau^2 / sum tau; undefined -> not drawn
                k_tw = []
                for lam in lam_list:
                    qa2 = np.exp(-2 * lam * dt_)
                    tau = (1 + qa2) / (1 - qa2)
                    ne = n_ / (np.sum(tau ** 2) / np.sum(tau))
                    k_tw.append(ne / (ne - N_ - 1) if ne > N_ + 1 else np.nan)
                p["th_tw"] = rel_err_r(r_th[keep], np.array(k_tw)[keep], n_, dt_)
            # q_eff = n / (tau_bar N) on the typical spectrum, over the simulated c range (largest at small c)
            p["qe"] = np.array([float(kappa_cW_core(semicircle_spectrum(c, N_), n_, dt_)[1])
                                for c in (d["c_values"].min(), d["c_values"].max())])
            P.append(p)
            for i in o:
                rows.append({"q": q, "Delta_t": d["Delta_t"], "c": d["c_values"][i], "r": med["r"][i],
                             "MC": med["mc"][i], "theory_kW": med["th_W"][i], "theory_kcW": med["th_cW"][i],
                             "kappa_W": med["kappa_W"][i], "kappa_cW": med["kappa_cW"][i], "q_eff": med["q_eff"][i],
                             "n_A_kcW_undefined": d["n_A"] - np.isfinite(s["kappa_cW"][i]).sum()})
        np.savez(cache, **{f"p{i}_{k}": v for i, p in enumerate(P) for k, v in p.items()})
        print(f"Saved: {cache}")
        summary = pd.DataFrame(rows)
        summary.to_csv(os.path.join(args.output_dir, "fig2_soft_mode_panels" + ("" if args.q == 10 else f"_q{args.q}")
                                    + ("_debias" if args.debias else "") + ("_lmin" if args.lmin else "") + ".csv"), index=False)
        pd.set_option("display.width", 250)
        print(summary.round(4).to_string(index=False))

    col_w, col_cw = LINE_COLORS[args.line_colors]
    with plt.style.context(ps.CTX if args.tau_w else ps.MAIN_CTX):   # SI Fig. 2 / main Fig. 2
        fig, axes = plt.subplots(1, 3, figsize=(ps.W2, 2.35), layout="constrained", sharey=True)
        for ax, p, letter in zip(axes, P, "abc"):
            colors = ps.magma_c(len(p["c_values"]))   # full scale, as Fig. 1 and Ginibre
            for i_c in range(len(p["c_values"])):
                ax.scatter(p["r"][i_c], p["mc"][i_c], s=6, color=colors[i_c], alpha=0.8, linewidths=0, zorder=3)
            ax.plot(p["r_th"], p["th_W"], "--", lw=1.25, color=col_w, label=r"$\kappa_W$", zorder=4)
            ax.plot(p["r_th"], p["th_cW"], "-.", lw=1.25, color=col_cw, label=r"$\kappa_{\rm cW}$", zorder=4)
            if "th_tw" in p:
                ax.plot(p["r_th"], p["th_tw"], ":", lw=1.5, color="#e69f00",
                        label=r"$\kappa_{\rm cW}$, $\langle\tau\rangle_w$", zorder=4)
            ax.set_ylim(*Y_LIM)
            ax.set_xlim(0, 1.0)
            # panel title: what changes between panels (dt, and q_eff over the simulated c); fixed N, q in the caption
            ax.set_title(rf"$\Delta t={float(p['dt']):g}$,  $q_{{\rm eff}}={p['qe'].min():.1f}$–${p['qe'].max():.1f}$",
                         loc="right")
            ax.set_xlabel(r"$r$")
            ps.panel_label(ax, letter)
            n_out = int(np.sum((p["mc"] > Y_LIM[1]) | (p["mc"] < Y_LIM[0])))
            if n_out:
                print(f"panel {letter}: {n_out} points outside the y range {Y_LIM}")
        axes[0].set_ylabel(r"$\sigma_r/r$")
        axes[0].legend(loc="upper right")   # fixed N, q: in the caption
        ps.discrete_colorbar(fig, axes, P[0]["c_values"], ps.magma_c(len(P[0]["c_values"])), r"$c$",
                             aspect=22, pad=0.015)
        for fmt in args.format:
            path = os.path.join(args.output_dir, stem + f".{fmt}")
            fig.savefig(path, dpi=args.dpi)
            print(f"Saved: {path}")



if __name__ == "__main__" and args.panels:
    panels_figure()
    raise SystemExit

if __name__ == "__main__":

    runs = [load_estimates(args.data_dir, "GOE", args.q, tag, args.symmetrize) for tag in args.tags]
    d = dict(runs[0])
    for k in ["A_true", "A_hat", "Q_hat", "c_values"]:
        d[k] = np.concatenate([run[k] for run in runs])
    s = soft_mode_estimated(d) if args.estimated else soft_mode(d)
    N, q, dt = d["N"], d["q"], d["Delta_t"]

    if args.stat == "median":
        lo, med, hi = ({k: np.nanpercentile(v, p, axis=1) for k, v in s.items()} for p in (25, 50, 75))
    elif args.stat == "median_se":
        # median over A, symmetric bar = standard error of the median (bootstrap over the realisations of A)
        med = {k: np.nanmedian(v, axis=1) for k, v in s.items()}
        rng = np.random.default_rng(20261112)
        idx = rng.integers(0, d["n_A"], size=(1000, d["n_A"]))
        se = {k: np.nanstd(np.nanmedian(v[:, idx], axis=2), axis=1, ddof=1) for k, v in s.items()}
        lo, hi = ({k: med[k] + sign * se[k] for k in s} for sign in (-1, 1))
    else:
        s = {k: np.where(np.isinf(v), np.nan, v) for k, v in s.items()}
        med = {k: np.nanmean(v, axis=1) for k, v in s.items()}
        std = {k: np.nanstd(v, axis=1, ddof=1) for k, v in s.items()}
        lo, hi = ({k: med[k] + sign * std[k] for k in s} for sign in (-1, 1))

    summary = pd.DataFrame({
        "c": d["c_values"], "r": med["r"],
        "MC": med["mc"], "MC_lo": lo["mc"], "MC_hi": hi["mc"],
        "theory_kW": med["th_W"], "theory_kcW": med["th_cW"],
        "theory_kcW_lo": lo["th_cW"], "theory_kcW_hi": hi["th_cW"],
        "theory_bound": med["th_bd"] if "th_bd" in med else np.nan,
        "kappa_W": med["kappa_W"], "kappa_r": med["kappa_r"], "kappa_cW": med["kappa_cW"],
        "q_eff": med["q_eff"], "n_A_kcW_divergent": d["n_A"] - np.isfinite(s["kappa_cW"]).sum(axis=1),
    })
    if args.estimated:
        summary["frac_neg"] = np.nanmean(s["frac_neg"], axis=1)
    pd.set_option("display.width", 250)
    print(f"estimated = {args.estimated}, stat = {args.stat}")
    print(summary.round(4).to_string(index=False))
    stem = ("fig2_soft_mode" + ("_sym" if args.symmetrize else "") + ("" if args.q == 10 else f"_q{args.q}")
            + ("_est" if args.estimated else "") + ("_mean" if args.stat == "mean" else "") + ("_medse" if args.stat == "median_se" else "")
            + ("_bound" if args.bound else "") + ("_tauvar" if args.tau_variants else "") + ("_cloud" if args.cloud else "") + ("_log" if args.log else "") + ("_logy" if args.logy else ""))
    summary.to_csv(os.path.join(args.output_dir, stem + ".csv"), index=False)

    order = np.argsort(med["r"])
    r = med["r"][order]
    mc = med["mc"][order]

    plot(r, mc, lo, hi, med, order, N, q, dt, stem, args.bound, s if args.cloud else None, args.log, args.logy)
