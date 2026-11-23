"""
SI Fig. 4 (fig:varQ_data): distributions of the estimated variance of the entries of Q,

    Var(Q_hat_ij) ~ [(H_QQ)^{-1}]_(ij),(ij) = (T3^{-1})_jj (Sigma_dt_hat)_ii / n          (SI Eq. bridge)

for the diagonal (i = j) and off-diagonal (i != j) entries, n = M - lag transitions per window. The figure shows
n Var(Q_hat_ij) = (T3^{-1})_jj (Sigma_dt_hat)_ii, which is of order one and does not depend on the window length.
Same preprocessing as the other real-data figures (ou_core): per-variable centering, one global std.
  (a) EEG (Sz1, 512 Hz, N = 47, q = 100): one window per stage, centred at 1 min (pre-ictal), 3 min (onset) and
      5 min (ictal); seizure onset at 180 s.
  (b) Financial |1-s returns| (N = 47, q = 80): windows of M = qN samples with step M/4 within each trading day,
      entries of all the windows of a day pooled.
  (c) Plankton log-abundances (N = 47, M = 264): one window, estimator at lag k = 1..5, i.e. dt = k/3 day.
Gaussian kernel density (robust Silverman bandwidth) on a linear x axis, cut at the median over the curves of a
panel of their 99th percentile, so that a single long tail (EEG onset) runs off the axis; dashed lines: mean of each
distribution. --log-x: density of log10(n Var) on a logarithmic axis.

The numbers drawn are saved in results/realdata/si_fig4_var_q.npz; --replot draws from that file only (the EEG and
financial raw data are not distributed). Run from code/realdata/:
    python3 si_fig4_var_q.py --data-dir <folder with EEG_data, FINANCIAL_data, PLANKTON_data>
    python3 si_fig4_var_q.py --replot
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ou_core as oc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style"))
import paperstyle as ps

parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", type=str, default=oc.REAL_DIR,
                    help="folder with EEG_data/, FINANCIAL_data/ and PLANKTON_data/ (default: stability-inference-OU/data)")
parser.add_argument("--eeg-q", type=int, default=100)
parser.add_argument("--eeg-centres-sec", type=float, nargs=3, default=[60.0, 180.0, 300.0])
parser.add_argument("--fin-q", type=int, default=80)
parser.add_argument("--step-div", type=int, default=4)
parser.add_argument("--plankton-lags", type=int, nargs="+", default=[1, 2, 3, 4, 5])
parser.add_argument("--cache", type=str, default=os.path.join(oc.RESULTS_DIR, "si_fig4_var_q.npz"))
parser.add_argument("--output-dir", type=str, default=oc.FIG_DIR)
parser.add_argument("--replot", action="store_true", help="draw from --cache only")
parser.add_argument("--log-x", action="store_true",
                    help="logarithmic x axis, kernel density of log10(n Var) (output suffix _logx)")
args = parser.parse_args()

oc.REAL_DIR = args.data_dir
oc.EEG_DIR = os.path.join(args.data_dir, "EEG_data")
oc.FIN_FILE = os.path.join(args.data_dir, "FINANCIAL_data", "Combined_processed", "Combined_processed",
                           "returns_wide_1s.csv")

EEG_FS = 512.0
EEG_ONSET_SEC = 180.0
STAGES = ["pre-ictal", "onset", "ictal"]


######################################################################################################################
# Var(Q_hat_ij) OF ONE WINDOW
######################################################################################################################

def var_q(X, lag=1):
    """X: (N, M). Returns (diagonal, off-diagonal) entries of n Var(Q_hat_ij) = (T3^{-1})_jj (Sigma_dt_hat)_ii."""

    X = X.astype(np.float64)
    n = X.shape[1] - lag
    X0, X1 = X[:, :-lag], X[:, lag:]
    T1, T2, T3 = X1 @ X1.T / n, X1 @ X0.T / n, X0 @ X0.T / n
    S_dt = T1 - T2 @ np.linalg.solve(T3, T2.T)
    V = np.outer(np.diag(S_dt), np.diag(np.linalg.inv(T3)))
    off = ~np.eye(len(V), dtype=bool)

    return np.diag(V).copy(), V[off]


def windows(T, M, step):
    return range(0, T - M + 1, step)


######################################################################################################################
# DATASETS
######################################################################################################################

def run_eeg():
    sig = oc.standardize(oc.load_eeg(), "per-variable")        # (T, N)
    N = sig.shape[1]
    M = args.eeg_q * N
    out = {}
    for stage, t in zip(STAGES, args.eeg_centres_sec):
        a = int(round(t * EEG_FS)) - M // 2
        out[stage] = var_q(sig[a:a + M].T)
        print(f"EEG {stage}: window {a / EEG_FS:.1f}-{(a + M) / EEG_FS:.1f} s, M = {M}")
    return out, N


def run_fin():
    df = oc.load_fin_returns(cache=False)
    df = pd.DataFrame(oc.standardize(np.abs(df.values), "per-variable"), index=df.index, columns=df.columns)
    N = df.shape[1]
    M = args.fin_q * N
    out = {}
    for day, g in df.groupby(df.index.date):
        v = g.values
        res = [var_q(v[s:s + M].T) for s in windows(len(v), M, M // args.step_div)]
        if not res:
            continue
        out[str(day)] = (np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res]))
        print(f"FIN {day}: {len(res)} windows, M = {M}")
    return out, N


def run_plankton():
    X = oc.standardize(oc.load_plankton(), "per-variable")       # (264, 47)
    out = {k: var_q(X.T, lag=k) for k in args.plankton_lags}
    print(f"Plankton: M = {X.shape[0]}, N = {X.shape[1]}, lags = {args.plankton_lags}")
    return out, X.shape[1], X.shape[0]


def compute():
    data = {}
    if os.path.isdir(oc.EEG_DIR):
        eeg, N = run_eeg()
        data["eeg_N"] = N
        for s, (d, o) in eeg.items():
            data[f"eeg|{s}|diag"], data[f"eeg|{s}|off"] = d, o
    else:
        print(f"No EEG data in {oc.EEG_DIR}: panel (a) skipped")
    if os.path.exists(oc.FIN_FILE):
        fin, N = run_fin()
        data["fin_N"] = N
        for s, (d, o) in fin.items():
            data[f"fin|{s}|diag"], data[f"fin|{s}|off"] = d, o
    else:
        print(f"No financial data at {oc.FIN_FILE}: panel (b) skipped")
    pl, N, M = run_plankton()
    data["plankton_N"], data["plankton_M"] = N, M
    for k, (d, o) in pl.items():
        data[f"plankton|{k}|diag"], data[f"plankton|{k}|off"] = d, o
    data = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) else v) for k, v in data.items()}
    data.update(eeg_q=args.eeg_q, fin_q=args.fin_q)
    np.savez(args.cache, **data)
    print(f"Saved: {args.cache}")


######################################################################################################################
# FIGURE
######################################################################################################################

def groups(z, name):
    keys = [k for k in z.files if k.startswith(name + "|") and k.endswith("|diag")]
    labels = list(dict.fromkeys(k.split("|")[1] for k in keys))
    if name == "plankton":
        labels.sort(key=int)
    if name == "eeg":
        labels.sort(key=STAGES.index)
    return labels


def kde_robust(v):
    """Gaussian KDE with Silverman's robust bandwidth 0.9 min(std, IQR/1.34) n^(-1/5): a few extreme values
    (long right tails, linear axis) do not widen the kernel."""
    sd = np.std(v, ddof=1)
    iqr = np.subtract(*np.percentile(v, [75, 25]))
    h = 0.9 * min(sd, iqr / 1.34) * len(v) ** (-0.2)
    return gaussian_kde(v, bw_method=h / sd)


def panel(z, name, labels, colors, stem, legend=None, cbar=None):
    """Two panels (diagonal, off-diagonal): Gaussian kernel density of log10(n Var), Scott bandwidth, drawn on a
    logarithmic x axis (density in log10); dashed lines: means."""
    with plt.style.context(ps.CTX):
        fig, axes = plt.subplots(1, 2, figsize=(ps.W2, 2.1), layout="constrained")
        for ax, part, title in zip(axes, ["diag", "off"], ["diagonal", "off-diagonal"]):
            vals = [z[f"{name}|{g}|{part}"].astype(float) for g in labels]
            if not args.log_x:
                allv = np.concatenate(vals)
                hi = np.median([np.nanpercentile(v, 99) for v in vals])   # one group with a long tail
                lo = np.nanpercentile(allv, 0.2)                          # (EEG onset) is cut, not the others
                lo = 0.0 if lo < 0.3 * hi else lo
                grid = np.linspace(lo, hi, 400)
                for v, col, g in zip(vals, colors, labels):
                    ax.plot(grid, kde_robust(v[np.isfinite(v)])(grid), color=col, lw=1.2,
                            label=legend[labels.index(g)] if legend else None)
                    ax.axvline(np.nanmean(v), color=col, ls="--", lw=0.8)
                ax.set_xlim(lo, hi)
                ax.set_ylim(bottom=0)
                ax.set_xlabel(r"$n\,\mathrm{Var}(\hat Q_{ij})$")
                ax.set_title(title, loc="right")
                continue
            allv = np.log10(np.concatenate(vals))
            grid = np.linspace(np.nanpercentile(allv, 0.2), np.nanpercentile(allv, 99.8), 400)
            for v, col, g in zip(vals, colors, labels):
                lv = np.log10(v[np.isfinite(v) & (v > 0)])
                ax.plot(10 ** grid, gaussian_kde(lv)(grid), color=col, lw=1.2,
                        label=legend[labels.index(g)] if legend else None)
                ax.axvline(np.nanmean(v), color=col, ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            few = (grid[-1] - grid[0]) < 1.5          # label 2, 3, 5 only when the axis spans few decades
            ax.xaxis.set_minor_formatter(FuncFormatter(
                lambda x, _: f"{x:g}" if few and round(x / 10 ** np.floor(np.log10(x)), 6) in (2, 3, 5) else ""))
            ax.set_xlim(10 ** grid[0], 10 ** grid[-1])
            ax.set_ylim(bottom=0)
            ax.set_xlabel(r"$n\,\mathrm{Var}(\hat Q_{ij})$")
            ax.set_title(title, loc="right")
        axes[0].set_ylabel("density" if not args.log_x else r"density in $\log_{10}$")
        if legend:
            axes[0].legend(loc="upper right", handlelength=1.2)
        if cbar:
            ps.discrete_colorbar(fig, axes, cbar[0], colors, cbar[1], fmt="{}")
        for fmt in ["png", "pdf"]:
            path = os.path.join(args.output_dir, f"{stem}{'_logx' if args.log_x else ''}.{fmt}")
            fig.savefig(path, dpi=600)
            print(f"Saved: {path}")
        plt.close(fig)


def plot():
    z = np.load(args.cache)
    os.makedirs(args.output_dir, exist_ok=True)

    if groups(z, "eeg"):
        labels = groups(z, "eeg")
        legend = [f"{s}, {t / 60:g} min" for s, t in zip(labels, args.eeg_centres_sec)]
        panel(z, "eeg", labels, ps.shades("eeg", len(labels)), "SI_FIG4a_est_var_EEG", legend=legend)

    if groups(z, "fin"):
        days = pd.to_datetime(groups(z, "fin"))
        colors = ps.dataset_cmap("fin")(np.linspace(0.15, 1, len(days)))
        panel(z, "fin", groups(z, "fin"), colors, "SI_FIG4b_est_var_FIN",
              cbar=([f"{d.day} Mar" for d in days], "trading day (2020)"))

    lags = groups(z, "plankton")
    colors = ps.dataset_cmap("plankton")(np.linspace(0.0, 1.0, len(lags)))
    panel(z, "plankton", lags, colors, "SI_FIG4c_est_var_PLANK",
          cbar=([f"{int(k)}/3" for k in lags], r"$\Delta t$ (day)"))


if __name__ == "__main__":
    if not args.replot:
        compute()
    plot()
