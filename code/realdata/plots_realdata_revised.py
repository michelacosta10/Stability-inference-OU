import argparse
import os
import pickle

import sys

import numpy as np
from scipy.optimize import brentq

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ou_core as oc
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style"))
import paperstyle as ps   # common style of the paper figures (Figs. 4, 5, SI Fig. 3)
from fractions import Fraction


######################################################################################################################
# ARGUMENTS
######################################################################################################################

parser = argparse.ArgumentParser(
    description="Figures for the empirical datasets from the pickle produced by estimate_realdata_MOU.py "
                "(paper Fig. 4: r_hat*dt time series, Fig. 5: datasets on the resolvability phase diagram)."
)
parser.add_argument("--dataset", type=str, required=True, help="Path to the pickle saved by estimate_realdata_MOU.py")
parser.add_argument("--output-dir", type=str, default="./figures")
parser.add_argument("--format", type=str, default="pdf", choices=["pdf", "png"])
parser.add_argument("--dpi", type=int, default=300)
parser.add_argument(
    "--figures", type=str, nargs="+", default=["all"],
    help="Which figures to generate (default: all). See FIGURES dict in this script for names."
)
parser.add_argument("--boundaries", type=str, default=os.path.join(oc.RESULTS_DIR, "fig5_boundaries.npz"),
                    help="GOE boundaries from fig5_boundaries.py (figure phase_diagram)")
parser.add_argument("--residual-bootstrap", type=str, default=None,
                    help="phase_diagram: results/fig5_residual_bootstrap.pkl; its sigma_r replaces the moving-block one "
                         "for the datasets in --residual-for")
parser.add_argument("--residual-for", type=str, nargs="+", default=["plankton"])
parser.add_argument("--c-key", type=str, default="c_hat_TW", choices=["c_hat_TW", "c_hat"],
                    help="phase_diagram x: c_hat_TW (r_hat / lam_bar, default) or c_hat = (1 - r_hat dt) / sqrt2")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)


def savefig(fig, name, tight=True):
    """tight=False for the paper figures, which are made at their final print size."""
    path = os.path.join(args.output_dir, f"{name}.{args.format}")
    fig.savefig(path, bbox_inches="tight" if tight else None, dpi=args.dpi)
    plt.close(fig)
    print(f"Saved: {path}")


def dt_text(dt):
    """Sampling interval as in the text: 1/64 rather than 8/512 or 0.015625."""
    f = Fraction(dt).limit_denominator(4096)
    return f"{f.numerator}" if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


mu_TW = -1.21
c_crit = 1 / np.sqrt(2)

STYLE = {
    "eeg":      {"label": "EEG",       "color": "royalblue",   "marker": "o", "dt_label": r"$\Delta t=1/512$ s"},
    "fin":      {"label": "Financial", "color": "tomato",      "marker": "s", "dt_label": r"$\Delta t=1$ s"},
    "plankton": {"label": "Plankton",  "color": "forestgreen", "marker": "^", "dt_label": r"$\Delta t=1/3$ d"},
}


######################################################################################################################
# THEORY: Tracy-Widom corrected r(c), <tau>_W, kappa_cW boundary
######################################################################################################################

def r_TW(c, N):
    return (1 - np.sqrt(2) * c) - (np.sqrt(2) * c) ** (1 / 3) * mu_TW * N ** (-2 / 3)


def estimate_c_from_r_TW(r_obs, N):
    try:
        return brentq(lambda c: r_TW(c, N) - r_obs, 0.0, 5.0)
    except ValueError:
        return np.nan


def tau_avg_wigner_vec(c, dt_array):

    lam_min = max(1 - np.sqrt(2) * c, 1e-6)
    lam_max = 1 + np.sqrt(2) * c

    n_pts = 200
    lam_pts, weights = np.polynomial.legendre.leggauss(n_pts)
    lam_pts = 0.5 * (lam_max - lam_min) * lam_pts + 0.5 * (lam_max + lam_min)
    weights = weights * 0.5 * (lam_max - lam_min)

    rho = np.sqrt(np.maximum(2 * c ** 2 - (lam_pts - 1) ** 2, 0)) / (np.pi * c ** 2)

    dt_array = np.atleast_1d(dt_array)
    x = np.outer(lam_pts, dt_array)
    tau = np.where(
        x > 1e-6,
        (1 + np.exp(-2 * x)) / (1 - np.exp(-2 * x)),
        1.0 / (x + 1e-10)
    )

    numerator = np.einsum('i,i,ij->j', weights, rho, tau ** 2)
    denominator = np.einsum('i,i,ij->j', weights, rho, tau)

    return numerator / denominator


def compute_rel_err_arr(r_val, q_arr, dt_val, N, tau_val):

    if r_val <= 0:
        return np.full(len(q_arr), 999.0)

    M = q_arr * N
    Meff = M / max(tau_val, 1e-6)

    denom_kappa = Meff - N - 2
    kappa = np.where(denom_kappa > 0, (Meff - 1) / denom_kappa, np.inf)

    x = r_val * dt_val
    with np.errstate(over='ignore'):
        exp2x = np.exp(min(2 * x, 500))
    g = (exp2x - 1) / (2 * x) if x > 1e-6 else 1.0

    T_eff = r_val * M * dt_val

    with np.errstate(invalid='ignore', divide='ignore'):
        rel_err = np.where(
            (T_eff > 0) & np.isfinite(kappa),
            np.sqrt(2 * kappa * g / T_eff),
            np.inf
        )

    return np.where(np.isfinite(rel_err), rel_err, 999.0)


def boundary_curve(c_line, N_val, dt_val, threshold=1.0):

    q_try = np.logspace(0, 6, 5000)
    q_stars = np.full(len(c_line), np.nan)

    for i, c in enumerate(c_line):
        r_val = r_TW(c, N_val)
        if r_val <= 0:
            continue
        tau_val = tau_avg_wigner_vec(c, np.array([dt_val]))[0]
        rel_err = compute_rel_err_arr(r_val, q_try, dt_val, N_val, tau_val)
        idx = np.where(rel_err <= threshold)[0]
        if len(idx) > 0:
            q_stars[i] = q_try[idx[0]]

    return q_stars


######################################################################################################################
# FIGURE 4: r_hat * dt across sliding windows (EEG top, financial bottom)
######################################################################################################################

def fig_rhat_timeseries(data):

    ds = data["datasets"]
    panels = [name for name in ("eeg", "fin") if name in ds]
    if not panels:
        print("No EEG/financial data: skipping.")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 4 * len(panels)), squeeze=False)
    axes = axes[:, 0]

    for ax, name in zip(axes, panels):

        d = ds[name]

        if name == "eeg":
            ax.plot(d["time_sec"], d["lambda_min"] * d["delta_t"], color=STYLE["eeg"]["color"], lw=1.5)
            ax.axvline(d["seizure_start_sec"], color="red", ls="--", lw=1.2, label="seizure onset")
            ax.set_xlabel("Time [s]")
            ax.legend(fontsize=8)
        else:
            for day, dd in d["days"].items():
                ax.plot(dd["time_sec"] / 3600, dd["lambda_min"] * d["delta_t"], lw=1.3, alpha=0.8, label=str(day))
            ax.set_xlabel("Time of day [h]")
            ax.legend(fontsize=8, ncol=2)

        ax.axhline(0, color="gray", ls="--", lw=1.2)
        ax.set_ylabel(r"$\hat{r}\,\Delta t = \lambda_{\min}(\hat{A})\,\Delta t$")
        ax.set_title(f'{STYLE[name]["label"]} ($N={d["N"]}$, {d.get("dt_label", STYLE[name]["dt_label"])})',
                     fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, ls=":")

    if "plankton" in ds:
        print(f'Plankton (single window): r_hat*dt = {data["summary"]["plankton"]["r_dt"]:.4f}')

    fig.tight_layout()
    savefig(fig, "realdata_rhat_timeseries")


def fig_rhat_physical(data):
    """Paper Fig. 4: r_hat in physical units (1/s) across sliding windows, linear axes, two columns (183 mm): two wide
    panels of equal width placed by hand, the date colour bar to the right of the financial panel.
    EEG: full recording, window centres, pre-ictal segment used in Fig. 5 shaded, seizure onset black dashed;
    financial: one curve per trading day, coloured by date (RdPu, colour bar)."""

    ds = data["datasets"]
    with plt.style.context(ps.MAIN_CTX):
        L, R, T, PH, GAP, B = 0.62, 1.0, 0.25, 1.55, 0.62, 0.45   # inches: left, right, top, panel height, gap, bottom
        H = T + 2 * PH + GAP + B
        fig = plt.figure(figsize=(ps.W2, H))
        PW = ps.W2 - L - R
        axes = [fig.add_axes([L / ps.W2, (B + PH + GAP) / H, PW / ps.W2, PH / H]),
                fig.add_axes([L / ps.W2, B / H, PW / ps.W2, PH / H])]
        cax = fig.add_axes([(L + PW + 0.1) / ps.W2, B / H, 0.1 / ps.W2, PH / H])

        d = ds["eeg"]
        col = ps.DATASET["eeg"]["color"]
        t = d["time_sec"] + d["window_size"] * d["delta_t"] / 2
        a, b = data["config"].get("fig5_eeg_segment_sec", [0.0, d["seizure_start_sec"]])
        axes[0].axvspan(a, b, color=col, alpha=0.10, lw=0, label="pre-ictal segment")
        axes[0].plot(t, d["lambda_min"], color=col, lw=1.0)
        axes[0].axvline(d["seizure_start_sec"], color=ps.INK, ls="--", lw=1.0, label="seizure onset")
        axes[0].set_xlabel("time (s)")
        axes[0].legend(loc="upper right")

        d = ds["fin"]
        days = list(d["days"])
        colors = ps.dataset_cmap("fin")(np.linspace(0.15, 1, len(days)))
        for day, color in zip(days, colors):
            dd = d["days"][day]
            axes[1].plot(dd["time_sec"] / 3600, dd["lambda_min"], lw=1.0, color=color)
        axes[1].set_xlabel("time of day (h)")
        ps.discrete_colorbar(fig, None, [f"{x.day} Mar" if x.month == 3 else str(x) for x in days], colors,
                             "trading day (2020)", fmt="{}", cax=cax)

        for ax, name in zip(axes, ["eeg", "fin"]):
            dd = ds[name]
            ax.axhline(0, color=ps.GREY, ls=":", lw=1.0)
            ax.set_ylabel(r"$\hat r = \lambda_{\min}(\hat{\mathbf{A}})$ (s$^{-1}$)")
            ax.set_title(f'{STYLE[name]["label"]} ($N={dd["N"]}$, $\\Delta t={dt_text(dd["delta_t"])}$ s, '
                         f'$q={dd["q"]}$)', loc="left")

        print(f'Plankton (single window): r_hat = {data["summary"]["plankton"]["r_hat"]:.4f} per day')
        savefig(fig, "realdata_rhat_physical", tight=False)



######################################################################################################################
# IMPLIED c FROM THE TRACY-WIDOM CORRECTED r(c, N)
######################################################################################################################

def fig_c_TW_timeseries(data):

    ds = data["datasets"]
    panels = [name for name in ("eeg", "fin") if name in ds]
    if not panels:
        print("No EEG/financial data: skipping.")
        return

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 4 * len(panels)), squeeze=False)
    axes = axes[:, 0]

    for ax, name in zip(axes, panels):

        d = ds[name]
        N = d["N"]

        if name == "eeg":
            c_TW = np.array([estimate_c_from_r_TW(r, N) for r in d["lambda_min"] * d["delta_t"]])
            ax.plot(d["time_sec"], c_TW, color=STYLE["eeg"]["color"], lw=1.5)
            ax.axvline(d["seizure_start_sec"], color="red", ls="--", lw=1.2, label="seizure onset")
            ax.set_xlabel("Time [s]")
        else:
            for day, dd in d["days"].items():
                c_TW = np.array([estimate_c_from_r_TW(r, N) for r in dd["lambda_min"] * d["delta_t"]])
                ax.plot(dd["time_sec"] / 3600, c_TW, lw=1.3, alpha=0.8, label=str(day))
            ax.set_xlabel("Time of day [h]")

        ax.axhline(c_crit, color="black", ls="--", lw=1.2, label=r"$c_{\rm crit}=1/\sqrt{2}$")
        ax.set_ylabel(r"$c_{\mathrm{TW}}$")
        ax.set_title(rf'{STYLE[name]["label"]}: implied $c_{{\mathrm{{TW}}}}(t)$', fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3, ls=":")
        ax.legend(fontsize=8, ncol=2)

    if "plankton" in ds:
        p = ds["plankton"]
        print(f'Plankton: implied c_TW = {estimate_c_from_r_TW(p["lambda_min"] * p["delta_t"], p["N"]):.6f}')

    fig.tight_layout()
    savefig(fig, "realdata_c_TW_timeseries")


######################################################################################################################
# NORMALIZED INVERSE HESSIAN TRACES Tr(H^{-1})/N^2
######################################################################################################################

def fig_hessian_traces(data):
    """SI Fig. 3 (EEG): normalised inverse-Hessian traces on sliding windows, three panels, seizure onset black
    dashed, one RdPu tone per trace; no overall title (caption). Financial version: one curve per day, RdPu by date."""

    ds = data["datasets"]
    keys = [("trace_HJJ_inv", r"$\mathrm{Tr}[(H_{JJ})^{-1}]/N^2$"),
            ("trace_HQQ_inv", r"$\mathrm{Tr}[(H_{QQ})^{-1}]/N^2$"),
            ("trace_H_inv",   r"$\mathrm{Tr}(H^{-1})/N^2$")]

    for name in ("eeg", "fin"):

        if name not in ds:
            continue

        d = ds[name]
        with plt.style.context(ps.CTX):
            fig, axes = plt.subplots(1, 3, figsize=(ps.W2, 2.0), layout="constrained")

            for i, (ax, (key, title)) in enumerate(zip(axes, keys)):

                if name == "eeg":
                    ax.plot(d["time_sec"], d[key], color=ps.shades("eeg", 3)[i], lw=1.0)
                    ax.axvline(d["seizure_start_sec"], color=ps.INK, ls="--", lw=1.0)
                    ax.set_xlabel("time (s)")
                else:
                    days = list(d["days"])
                    colors = ps.dataset_cmap("fin")(np.linspace(0.15, 1, len(days)))
                    for day, color in zip(days, colors):
                        dd = d["days"][day]
                        ax.plot(dd["time_sec"] / 3600, dd[key], lw=1.0, color=color)
                    ax.set_xlabel("time of day (h)")

                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useMathText=True)
                ax.set_title(title, loc="right")
                ps.panel_label(ax, "abc"[i])

            axes[0].set_ylabel("normalised trace")
            if name == "fin":
                ps.discrete_colorbar(fig, axes, [f"{x.day} Mar" for x in days], colors, "trading day (2020)",
                                     fmt="{}", aspect=25, pad=0.02)

            savefig(fig, f"realdata_hessian_trace_{name}", tight=False)

    if "plankton" in ds:
        p = ds["plankton"]
        print(f'Plankton: Tr(HJJ^-1)/N^2 = {p["trace_HJJ_inv"]:.6e}, Tr(HQQ^-1)/N^2 = {p["trace_HQQ_inv"]:.6e}, '
              f'Tr(H^-1)/N^2 = {p["trace_H_inv"]:.6e}')



######################################################################################################################
# FIGURE 5: empirical datasets on the resolvability phase diagram + bootstrap histograms of lambda_min
######################################################################################################################

def phase_diagram_datasets(data, threshold, name, title_suffix):

    ds = data["datasets"]
    names = [n for n in ("eeg", "fin", "plankton") if n in ds]

    c_line = np.linspace(0.01, c_crit * 0.9999, 300)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(
        nrows=len(names), ncols=2,
        width_ratios=[2.2, 1.0],
        wspace=0.35, hspace=0.45
    )

    ax = fig.add_subplot(gs[:, 0])
    hist_axes = [fig.add_subplot(gs[i, 1]) for i in range(len(names))]

    for n in names:

        d, s, st = ds[n], data["summary"][n], STYLE[n]

        q_star = boundary_curve(c_line, d["N"], d["delta_t"], threshold=threshold)

        ax.plot(
            c_line, q_star,
            color=st["color"], lw=2, ls="--", alpha=0.8,
            label=rf'boundary {st["label"]} ($N={d["N"]}$, {d.get("dt_label", st["dt_label"])})'
        )

        q_s = q_star[np.argmin(np.abs(c_line - s["c_hat"]))]

        if not np.isnan(q_s):
            ax.plot(
                [s["c_hat"], s["c_hat"]],
                [min(s["q"], q_s), max(s["q"], q_s)],
                color=st["color"], lw=1.5, ls=":", alpha=0.5
            )

        ax.errorbar(
            s["c_hat"], s["q"],
            xerr=s["sigma_r_dt"] / np.sqrt(2),
            fmt=st["marker"], color=st["color"],
            ms=12, zorder=5, capsize=4,
            label=rf'{st["label"]} ($\sigma_r/r={s["rel_err"]:.2f}$)'
        )

        ax.annotate(
            rf'{st["label"]}' + "\n" + rf'$\sigma_r/r={s["rel_err"]:.2f}$',
            xy=(s["c_hat"], s["q"]),
            # points far from c_crit sit under the legend: label them below instead of above
            xytext=(s["c_hat"] + 0.03, s["q"] / 4) if s["c_hat"] < 0.5 else (s["c_hat"] - 0.06, s["q"] * 3),
            fontsize=9, color=st["color"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=st["color"], lw=1)
        )

    ax.axvline(c_crit, color="black", ls="--", lw=1, alpha=0.6, label=r"GOE critical point $c=1/\sqrt{2}$")

    ax.set_yscale("log")
    ax.set_xlabel(r"inferred criticality $\hat{c} = (1 - \hat{r}\,\Delta t)/\sqrt{2}$", fontsize=12)
    ax.set_ylabel(r"$q = M/N$", fontsize=12)
    ax.set_title(
        rf"Empirical datasets vs resolvability boundary ({title_suffix})" + "\n"
        + r"($\bar{r}(c,N)$ with Tracy--Widom correction, $\kappa_{\rm cW}$)",
        fontsize=11
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.3, ls=":")
    # x range adapts to the data (per-variable centering puts the financial panel at c_hat ~ 0.33)
    c_min = min(data["summary"][n]["c_hat"] for n in names)
    ax.set_xlim(min(0.4, c_min - 0.05), c_crit * 1.02)

    for axh, n in zip(hist_axes, names):

        d, s, st = ds[n], data["summary"][n], STYLE[n]

        axh.hist(d["r_boot_all"], bins=40, density=True, color=st["color"], alpha=0.65,
                 edgecolor="black", linewidth=0.4)
        axh.axvline(s["r_hat"], color="black", ls="--", lw=1.5, label=r"$\hat{\lambda}_{\min}$")
        axh.axvline(0, color="gray", ls=":", lw=1)
        axh.set_title(rf'{st["label"]}: bootstrap $\lambda_{{\min}}$', fontsize=10, color=st["color"],
                      fontweight="bold")
        axh.set_xlabel(r"$\lambda_{\min}$", fontsize=10)
        axh.set_ylabel("density", fontsize=10)
        axh.grid(alpha=0.25, ls=":")
        axh.legend(fontsize=8, framealpha=0.9)

    savefig(fig, name)


def fig_phase_diagram_threshold1(data):
    phase_diagram_datasets(data, 1.0, "realdata_phase_diagram_threshold1", r"$\sigma_r/r = 1$")


def fig_phase_diagram_threshold01(data):
    phase_diagram_datasets(data, 0.1, "realdata_phase_diagram_threshold01", r"$\sigma_r/r = 0.1$")


def fig_phase_diagram(data):
    """Paper Fig. 5: all datasets on one (c_hat, q) plane, with the GOE boundaries sigma_r/r = 1 (solid) and
    0.1 (dashed) of each dataset computed with kappa_cW by fig5_boundaries.py; bootstrap lambda_min on the right.
    Colour and marker = dataset (paperstyle.DATASET), line style = threshold; legend once, details in the caption."""

    from matplotlib.lines import Line2D

    B = np.load(args.boundaries)
    ds = data["datasets"]
    boot = {n: (data["summary"][n]["rel_err"], ds[n]["r_boot_all"], "moving-block") for n in ds}
    if args.residual_bootstrap:
        with open(args.residual_bootstrap, "rb") as f:
            res = pickle.load(f)["results"]
        for n in args.residual_for:
            boot[n] = (res[n]["rel_err_residual"], np.concatenate(res[n]["r_boot"]), "residual")
    names = [n for n in ("eeg", "fin", "plankton") if n in ds]
    thr = list(B["thresholds"])
    styles = ["-", "--", ":"]

    with plt.style.context(ps.MAIN_CTX):
        fig = plt.figure(figsize=(ps.W2, 3.4), layout="constrained")
        gs = fig.add_gridspec(nrows=len(names), ncols=2, width_ratios=[2.0, 1.0])
        ax = fig.add_subplot(gs[:, 0])
        hist_axes = [fig.add_subplot(gs[i, 1]) for i in range(len(names))]
        n_c = len(B["c_grid"])

        for n in names:
            s, st = data["summary"][n], ps.DATASET[n]
            for t in range(len(thr)):
                ax.plot(B["c_grid"], B[f"{n}_q_star_{t}"], color=st["color"], lw=1.25, ls=styles[t],
                        marker=st["marker"], ms=3.5, markevery=(t * n_c // 14 + n_c // 28, n_c // 7),
                        markerfacecolor="white", markeredgewidth=0.9)

            rel = boot[n][0]
            c_pt, c_err = ((s["c_hat_TW"], rel * s["r_mod"] / np.sqrt(2)) if args.c_key == "c_hat_TW"
                           else (s["c_hat"], rel * s["r_dt"] / np.sqrt(2)))
            q_pt = s["q"] - 1 / ds[n]["N"]   # q = n / N = (M - 1) / N, the convention of the boundaries
            ax.errorbar(c_pt, q_pt, xerr=c_err, fmt=st["marker"], color=st["color"], mec=ps.INK, mew=0.6,
                        ms=7, zorder=5, capsize=2.5, elinewidth=1.0)
            off, ha = {"eeg": ((8, -12), "left"), "fin": ((-7, 6), "right"), "plankton": ((-24, 14), "right")}[n]
            ax.annotate(rf'$\sigma_r/r={rel:.2f}$', xy=(c_pt, q_pt), xytext=off, textcoords="offset points",
                        fontsize=9, color=ps.INK, ha=ha, va="center",
                        arrowprops=dict(arrowstyle="-", color=ps.GREY, lw=0.6, shrinkA=1, shrinkB=4) if n == "plankton" else None)

        c_crit_N = oc.c_from_r(0.0, ds[names[0]]["N"])
        ax.axvline(c_crit, color=ps.INK, ls=":", lw=1.0)
        ax.axvline(c_crit_N, color=ps.GREY, ls="--", lw=1.0)
        ax.set_yscale("log")
        ax.set_xlabel(r"inferred $\hat c$" if args.c_key == "c_hat_TW" else r"inferred $\hat c = (1 - \hat r\,\Delta t)/\sqrt{2}$")
        ax.set_ylabel(r"$q = n/N$")
        ax.set_xlim(B["c_grid"][0], c_crit_N + 0.005)

        h_data = [Line2D([], [], color=ps.DATASET[n]["color"], marker=ps.DATASET[n]["marker"], ls="-", lw=1.25,
                         ms=5, mec=ps.INK, mew=0.6, label=ps.DATASET[n]["label"]) for n in names]
        h_thr = [Line2D([], [], color=ps.INK, ls="-", lw=1.25, label=r"diagnosis, $\sigma_r/r=1$"),
                 Line2D([], [], color=ps.INK, ls="--", lw=1.25, label=r"resolution, $\sigma_r/r=0.1$")]
        h_crit = [Line2D([], [], color=ps.INK, ls=":", lw=1.0, label=r"$c=1/\sqrt{2}$ ($N\to\infty$)"),
                  Line2D([], [], color=ps.GREY, ls="--", lw=1.0, label=rf'$\bar r=0$ at $N={ds[names[0]]["N"]}$')]
        leg = ax.legend(handles=h_data, loc="upper left")
        ax.add_artist(leg)
        ax.legend(handles=h_thr + h_crit, loc="upper left", bbox_to_anchor=(0.25, 1.0))

        for axh, n in zip(hist_axes, names):
            s, st = data["summary"][n], ps.DATASET[n]
            axh.hist(boot[n][1], bins=40, density=True, color=st["color"], alpha=0.85,
                     edgecolor=ps.INK, linewidth=0.3)
            axh.axvline(s["r_hat"], color=ps.INK, ls="--", lw=1.25)
            axh.axvline(0, color=ps.GREY, ls=":", lw=1.0)
            axh.set_title(st["label"], loc="left")
            axh.set_ylabel("density")
        hist_axes[-1].set_xlabel(r"$\lambda_{\min}(\hat{\mathbf{A}})$")

        savefig(fig, "realdata_phase_diagram", tight=False)



######################################################################################################################
# MAIN
######################################################################################################################

FIGURES = {
    "rhat_timeseries": fig_rhat_timeseries,
    "c_TW_timeseries": fig_c_TW_timeseries,
    "hessian_traces": fig_hessian_traces,
    "phase_diagram_threshold1": fig_phase_diagram_threshold1,
    "phase_diagram_threshold01": fig_phase_diagram_threshold01,
    "phase_diagram": fig_phase_diagram,
    "rhat_physical": fig_rhat_physical,
}


if __name__ == "__main__":

    with open(args.dataset, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded real-data estimates from: {args.dataset}")
    print(f"datasets = {list(data['datasets'])}, center = {data['config']['center']}")

    for name, s in data["summary"].items():
        print(f"{name:9s}: r*dt = {s['r_dt']:.5f}, c_hat = {s['c_hat']:.5f}, "
              f"sigma_r/r = {s['rel_err']:.3f}, q = {s['q']:.1f}")

    requested = list(FIGURES.keys()) if args.figures == ["all"] else args.figures

    unknown = [name for name in requested if name not in FIGURES]
    if unknown:
        raise ValueError(f"Unknown figure name(s): {unknown}. Available: {list(FIGURES.keys())}")

    for name in requested:
        print(f"\n--- Generating figure: {name} ---")
        FIGURES[name](data)

    print(f"\nAll requested figures saved to: {args.output_dir}")
