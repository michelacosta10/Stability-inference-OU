"""
Figures and summary tables for the robustness analyses (reads results/*.pkl, writes figures/ and results/*.csv).

  figures/robust_fin.pdf, figures/robust_eeg.pdf : r_hat (1/s) and sigma_r/r versus the sampling interval,
                                                   at fixed M and at fixed duration T
  figures/innovations.pdf                         : Ljung-Box rejection fractions of the fitted innovations
  results/summary_robustness.csv, results/summary_innovations.csv
"""

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ou_core as ou
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style"))
import paperstyle as ps   # common style of the paper figures (SI Figs. 5-6)

# reference categorical palette (fixed order) + neutral inks
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MARKERS = ["o", "s", "^"]
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dd"
BLUE_RAMP = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 5,
})

U_STAR = 1.5936  # root of e^u (2 - u) = 2  ->  dt* = u* / (2 r) ~ 0.80 / r


def load(name):
    path = os.path.join(ou.RESULTS_DIR, name + ".pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def curves(d):
    ks = sorted(d["results"])
    R = [d["results"][k] for k in ks]
    dt = np.array([r["delta_t"] for r in R])
    r_med = np.array([r["summary"]["r_median"] for r in R])
    r_q = np.array([np.percentile(r["r_hat"], [25, 75]) for r in R])
    rel_boot = np.array([r["summary"]["rel_boot_median"] for r in R])
    rel_th = np.array([r["summary"]["rel_theory_r"] for r in R])
    return dt, r_med, r_q, rel_boot, rel_th


def theory_kcw(stem, ks):
    """Median over windows of the kappa_cW theory of the SI (theory_kcw.py); windows with kappa_cW = NaN dropped,
    NaN where no window has it defined (point not plotted)."""
    d = load(stem + "_kcw")
    if d is None:
        raise FileNotFoundError(f"results/{stem}_kcw.pkl missing: run theory_kcw.py")
    # point shown only if kappa_cW is defined in at least half of the windows
    return np.array([d["results"][k]["summary"]["rel_theory_median"] if d["results"][k]["summary"]["frac_nan"] <= 0.5
                     else np.nan for k in ks])


# copy of the robustness figures (SI Figs. 5-6) in stability-inference-OU/figures
PAPER_FIG_DIR = os.path.join(ou.ROOT, "figures")


def dt_star(dt, r_med):
    """Self-consistent dt* = 0.80 / r_hat, with r_hat taken at the sampling interval closest to dt*."""
    cand = U_STAR / (2 * r_med)
    i = np.argmin(np.abs(np.log(dt) - np.log(cand)))
    return cand[i]


def robustness_figure(dataset, variants, out_name):
    """SI Figs. 5-6 (paper style, 183 mm): (a) r_hat vs dt, (b) sigma_r / r at fixed M, (c) at fixed T.
    Colour + marker = variant (RdPu tones; one dataset colour if there is a single variant); points = bootstrap,
    lines = kappa_cW theory (black for a single variant, the variant colour otherwise, to keep the pairs);
    thresholds grey (diagnosis solid, resolution dashed), dt* black dashed. *_compare.pdf (not in the paper) as before."""

    from matplotlib.lines import Line2D

    single = len(variants) == 1
    colors = [ps.DATASET[dataset]["color"]] if single else ps.shades(dataset, len(variants))
    with plt.style.context(ps.CTX):
        fig, axes = plt.subplots(1, 3, figsize=(ps.W2, 2.3), layout="constrained")
    figc, axc = plt.subplots(1, 2, figsize=(7.5, 3.2), constrained_layout=True)   # *_compare.pdf, not in the paper
    rows = []

    for j, (label, stem) in enumerate(variants):
        c, m = colors[j], ps.VARIANT_MARKERS[j]
        c_th = ps.INK if single else c
        cs = SERIES[j]

        stemM = stem + "_fixedM" if "deseason" not in stem else stem.replace("_deseason", "_fixedM_deseason")
        dM = load(stemM)
        if dM is not None and dM["results"]:
            dt, r_med, r_q, rel_b, rel_r = curves(dM)
            rel_t = theory_kcw(stemM, sorted(dM["results"]))
            axes[0].plot(dt, r_med, color=c, marker=m, ms=4, lw=1.25, label=label)
            axes[0].fill_between(dt, r_q[:, 0], r_q[:, 1], color=c, alpha=0.15, linewidth=0)
            axes[1].plot(dt, rel_b, color=c, marker=m, ms=4, linestyle="none", zorder=3)
            axes[1].plot(dt, rel_t, color=c_th, lw=1.0)
            axc[0].plot(dt, rel_b, color=cs, marker=MARKERS[j], linestyle="none", label=f"{label}: bootstrap")
            axc[0].plot(dt, rel_t, color=cs, linewidth=1, label=f"{label}: theory ($\\kappa_{{\\rm cW}}$)")
            axc[0].plot(dt, rel_r, color=cs, linewidth=1, linestyle="--", label=f"{label}: theory ($\\kappa_r$, old)")
            if j == 0:
                ds = dt_star(dt, r_med)
                axes[1].axvline(ds, color=ps.INK, linestyle="--", linewidth=1.0)
                right = np.log(ds / np.min(dt)) > 0.6 * np.log(np.max(dt) / np.min(dt))   # line near the right edge
                axes[1].annotate(f"$\\Delta t^*\\approx{ds:.2g}$ s", (ds, 0.85), xycoords="data",   # just below sigma/r = 1
                                 color=ps.INK, ha="right" if right else "left", va="top", fontsize=9,
                                 xytext=(-3 if right else 3, 0),
                                 textcoords="offset points")
            for k, d_, rm, rb, rt in zip(sorted(dM["results"]), dt, r_med, rel_b, rel_r):
                rows.append([dataset, label, "fixedM", k, d_, dM["results"][k]["M"], dM["results"][k]["n_windows"],
                             rm, dM["results"][k]["summary"]["frac_stable"], rb, rt,
                             dM["results"][k]["summary"]["rel_theory_cW"], dM["results"][k]["zero_frac"]])

        dT = load(stem + "_fixedT") if "deseason" not in stem else None
        if dT is not None and dT["results"]:
            dt, r_med, r_q, rel_b, rel_r = curves(dT)
            rel_t = theory_kcw(stem + "_fixedT", sorted(dT["results"]))
            axes[2].plot(dt, rel_b, color=c, marker=m, ms=4, linestyle="none", zorder=3)
            axes[2].plot(dt, rel_t, color=c_th, lw=1.0)
            axc[1].plot(dt, rel_b, color=cs, marker=MARKERS[j], linestyle="none")
            axc[1].plot(dt, rel_t, color=cs, linewidth=1)
            axc[1].plot(dt, rel_r, color=cs, linewidth=1, linestyle="--")
            for k, d_, rm, rb, rt in zip(sorted(dT["results"]), dt, r_med, rel_b, rel_r):
                rows.append([dataset, label, "fixedT", k, d_, dT["results"][k]["M"], dT["results"][k]["n_windows"],
                             rm, dT["results"][k]["summary"]["frac_stable"], rb, rt,
                             dT["results"][k]["summary"]["rel_theory_cW"], dT["results"][k]["zero_frac"]])

    with plt.style.context(ps.CTX):
        for ax, letter in zip(axes, "abc"):
            ax.set_xscale("log")
            ax.set_xlabel(r"sampling interval $\Delta t$ (s)")
            ps.panel_label(ax, letter)
        axes[0].set_ylabel(r"$\hat r$ (s$^{-1}$)")
        for ax, t in [(axes[1], "fixed number of samples $M$"), (axes[2], "fixed duration $T$")]:
            ax.set_yscale("log")
            ax.set_ylabel(r"$\sigma_r/\hat r$")
            ax.set_title(t, loc="left")
            ax.axhline(1, color=ps.GREY, linewidth=1.0, linestyle=ps.LS_DIAGNOSIS)
            ax.axhline(0.1, color=ps.GREY, linewidth=1.0, linestyle=ps.LS_RESOLUTION)
        if not single:
            axes[0].set_ylim(top=1.45 * axes[0].get_ylim()[1])   # room for the legend above the data (axes only)
            axes[0].legend(loc="upper right")
        h = [Line2D([], [], color=ps.INK, marker="o", ms=4, ls="none", label="bootstrap"),
             Line2D([], [], color=ps.INK, lw=1.0, label=r"theory ($\kappa_{\rm cW}$)")]
        axes[2].legend(handles=h, loc="upper left")

    for ax, t in [(axc[0], "fixed number of samples $M$"), (axc[1], "fixed duration $T$")]:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"sampling interval $\Delta t$ (s)")
        ax.set_ylabel(r"$\sigma_r/\hat r$")
        ax.set_title(t, loc="left", color=INK)
        ax.axhline(1, color=MUTED, linewidth=0.8)
        ax.axhline(0.1, color=MUTED, linewidth=0.8, linestyle="--")
    axc[0].legend(fontsize=6)

    os.makedirs(ou.FIG_DIR, exist_ok=True)
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    with plt.style.context(ps.CTX):   # fonts are resolved at drawing time
        fig.savefig(os.path.join(ou.FIG_DIR, out_name))
        fig.savefig(os.path.join(PAPER_FIG_DIR, out_name))
    cmp_name = out_name.replace(".pdf", "_compare.pdf")
    figc.savefig(os.path.join(ou.FIG_DIR, cmp_name), bbox_inches="tight")
    plt.close(fig)
    plt.close(figc)
    print(f"Saved figures/{out_name}, figures/{out_name}, figures/{cmp_name}")

    return rows



def innovation_figure():

    d = load("innovation_tests")
    if d is None:
        return []

    panels = []
    if "fin" in d:
        for tr in d["config"]["fin_transforms"]:
            panels.append((f"financial, {'log realised vol.' if tr == 'logrv' else '|returns| subsampled'}", "fin", tr))
    if "eeg" in d:
        panels.append(("EEG pre-ictal", "eeg", None))

    fig, axes = plt.subplots(len(panels), 2, figsize=(7.5, 2.4 * len(panels)), constrained_layout=True, squeeze=False)
    rows = []

    for i, (title, ds, tr) in enumerate(panels):
        if ds == "fin":
            ks = d["config"]["fin_factors"]
            unit = "s"
            groups = {k: [v for (t, kk, _), v in d["fin"].items() if t == tr and kk == k] for k in ks}
        else:
            ks = d["config"]["eeg_factors"]
            unit = "s"
            groups = {k: [d["eeg"][k]] for k in ks}

        for j, k in enumerate(ks):
            items = [g for g in groups[k] if "frac_rej_lin" in g]
            if not items:
                continue
            lags = items[0]["lags_phys"]
            lin = np.mean([g["frac_rej_lin"] for g in items], axis=0)
            sq = np.mean([g["frac_rej_sq"] for g in items], axis=0)
            col = BLUE_RAMP[min(j, len(BLUE_RAMP) - 1)]
            dt_lab = f"{k} s" if ds == "fin" else f"{k / 512 * 1000:.0f} ms"
            axes[i, 0].plot(lags, lin, color=col, marker="o", label=f"$\\Delta t$ = {dt_lab}")
            axes[i, 1].plot(lags, sq, color=col, marker="o", label=f"$\\Delta t$ = {dt_lab}")
            for L, a, b in zip(lags, lin, sq):
                rows.append([ds, tr, k, L, a, b,
                             np.mean([g["gauss_soft_eps"]["ex_kurt"] for g in items]),
                             np.mean([g["gauss_soft_x"]["ex_kurt"] for g in items])])

        axes[i, 0].set_title(f"{title}: innovations $\\hat\\varepsilon$", loc="left", color=INK, fontsize=9)
        axes[i, 1].set_title(f"{title}: squared innovations $\\hat\\varepsilon^2$", loc="left", color=INK, fontsize=9)
        for ax in axes[i]:
            ax.set_xscale("log")
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel(f"Ljung–Box max lag ({unit})")
        axes[i, 0].set_ylabel("fraction of variables\nrejected (BH, 5%)")
        axes[i, 1].legend(fontsize=7, loc="lower left")

    fig.savefig(os.path.join(ou.FIG_DIR, "innovations.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/innovations.pdf")

    return rows


if __name__ == "__main__":

    cols = ["dataset", "variant", "protocol", "k", "delta_t_s", "M", "n_windows", "r_hat_median_per_s",
            "frac_stable", "rel_err_boot_median", "rel_err_theory_kappa_r", "rel_err_theory_kappa_cW", "zero_frac"]
    rows = robustness_figure("fin", [("log RV", "robust_fin_logrv"), ("|r| subsampled", "robust_fin_abs_sub"),
                                     ("log RV deseasoned", "robust_fin_logrv_deseason")], "robust_fin.pdf")
    rows += robustness_figure("eeg", [("EEG subsampled", "robust_eeg_sub")], "robust_eeg.pdf")
    pd.DataFrame(rows, columns=cols).to_csv(os.path.join(ou.RESULTS_DIR, "summary_robustness.csv"), index=False)

    rows = innovation_figure()
    pd.DataFrame(rows, columns=["dataset", "transform", "k", "max_lag_s", "frac_rej_lin", "frac_rej_sq",
                                "soft_eps_ex_kurt", "soft_x_ex_kurt"]).to_csv(
        os.path.join(ou.RESULTS_DIR, "summary_innovations.csv"), index=False)
    print("Saved results/summary_robustness.csv, results/summary_innovations.csv")


def pooled_figure():
    """r_hat versus dt up to 1 minute: all days pooled and one window per full day (log-log, slope -1 guide)."""

    p = load("pooled_days_fin")
    if p is None:
        return
    df = pd.DataFrame([{k: v for k, v in o.items() if k != "r_boot"} for o in p["results"]])

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.3), constrained_layout=True)
    names = {"logrv": "log RV", "abs_sub": "|r| subsampled"}

    for j, tr in enumerate(["logrv", "abs_sub"]):
        c, m = SERIES[j], MARKERS[j]
        s = df[(df["transform"] == tr) & (df["centering"] == "deseason")].sort_values("delta_t")
        axes[0].plot(s.delta_t, s.r_hat, color=c, marker=m, label=f"{names[tr]}, 10 days pooled")
        axes[1].plot(s.delta_t, s.rel_boot, color=c, marker=m, linestyle="none", label=f"{names[tr]}: bootstrap")
        axes[1].plot(s.delta_t, s.rel_theory_r, color=c, linewidth=1, label=f"{names[tr]}: theory ($\\kappa_r$)")
        d = load(f"robust_fin_{tr}_fixedT_fullday")
        if d is not None:
            dt, r_med, r_q, _, _ = curves(d)
            axes[0].plot(dt, r_med, color=c, marker=m, linestyle="--", linewidth=1, markerfacecolor="white",
                         label=f"{names[tr]}, one window per day")

    x = np.array([1, 60])
    axes[0].plot(x, 0.8 / x, color=MUTED, linewidth=1, linestyle=":")
    axes[0].annotate(r"$\hat r\,\Delta t = 0.8$ (i.e. $\hat r \propto 1/\Delta t$)", (8, 0.8 / 8), color=MUTED,
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\hat r$ (1/s)")
    axes[0].set_title(r"(a) $\hat r$ up to 1 minute", loc="left", color=INK)
    axes[0].legend(fontsize=7)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$\sigma_r/\hat r$")
    axes[1].set_title("(b) all days pooled: data vs theory", loc="left", color=INK)
    axes[1].axhline(0.1, color=MUTED, linewidth=0.8, linestyle="--")
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"sampling interval $\Delta t$ (s)")

    fig.savefig(os.path.join(ou.FIG_DIR, "pooled_fin.pdf"), bbox_inches="tight")
    plt.close(fig)
    print("Saved figures/pooled_fin.pdf")


if __name__ == "__main__":
    pooled_figure()
