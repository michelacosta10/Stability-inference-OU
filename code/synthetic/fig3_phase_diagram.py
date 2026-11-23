"""
Fig. 3 (fig:phase_diagram): GOE phase diagrams of sigma_r / r, from phase_diagram_kcw.py (--tag kcw, kappa_cW, paper)
(the --tag main option reads a Monte Carlo kappa_r file, not included).
  Grey: kappa_cW undefined (n_eff <= N + 1) or n <= N + 1.
  Row 1: diagnosis boundary sigma_r / r = 1; row 2: resolution boundary sigma_r / r = 0.1. Each row has its own
  colour scale, blue below the threshold and red above it (unresolved), with no white in between. q axes up to q_max_row1 in row 1. One colour bar
  per row, panels (a)-(f), grid linearly interpolated for drawing.
  Columns: (c, q) at dt_fixed with the gold curve q_eff = 1; (c, dt) at q_fixed and (q, dt) at c_fixed with the
  optimal sampling interval 0.80 / r_bar (white dashed, black outline).
  Linear axes.
  CSV: per c, q*(c), sigma_r / r on the gold curve and the q of the two boundaries at dt_fixed.
"""

import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "style"))
import paperstyle as ps   # common style of the paper figures

REV_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # stability-inference-OU/

parser = argparse.ArgumentParser()
parser.add_argument("--tag", type=str, default="kcw")
parser.add_argument("--data-dir", type=str, default=os.path.join(REV_DIR, "results", "synthetic"))
parser.add_argument("--output-dir", type=str, default=os.path.join(REV_DIR, "figures", "other", "fig3_phase_diagram"))
parser.add_argument("--q-max-row1", type=float, default=6.0, dest="q_max_row1")
parser.add_argument("--format", type=str, nargs="+", default=["pdf", "png"])
parser.add_argument("--dpi", type=int, default=600)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

UNDEF = "0.78"  # grey where kappa is undefined
THRESHOLDS = [(1.0, 3.0), (0.1, 1.0)]  # (threshold, upper end of the colour scale)

# RdBu_r without its white middle: blue below the threshold, red above it, a sharp jump at the threshold
_rdbu = plt.get_cmap("RdBu_r")
CMAP = ListedColormap(np.vstack([_rdbu(np.linspace(0.0, 0.38, 128)), _rdbu(np.linspace(0.62, 1.0, 128))]))


def edges(x):
    e = np.empty(len(x) + 1)
    e[1:-1] = 0.5 * (x[:-1] + x[1:])
    e[0], e[-1] = x[0] - 0.5 * (x[1] - x[0]), x[-1] + 0.5 * (x[-1] - x[-2])
    return e


def smooth(x, y, E, x_max, y_max, n=400):
    """Linear interpolation of the grid onto a fine regular one up to (x_max, y_max), for drawing only."""
    xf, yf = np.linspace(x[0], x_max, n), np.linspace(y[0], y_max, n)
    X, Y = np.meshgrid(xf, yf)
    return xf, yf, RegularGridInterpolator((y, x), E)((Y, X))


def boundary_q(q, E, level):
    """Smallest q above which sigma_r / r stays below level, per column of E (rows = q)."""
    out = np.full(E.shape[1], np.nan)
    for j in range(E.shape[1]):
        above = np.where(~(E[:, j] <= level))[0]
        if len(above) and above[-1] < len(q) - 1:
            out[j] = q[above[-1] + 1]
    return out


if __name__ == "__main__":

    z = np.load(os.path.join(args.data_dir, "phase_diagram", f"phase_diagram_{args.tag}.npz"))
    N, dt_fixed, q_fixed, c_fixed = int(z["N"]), float(z["dt_fixed"]), float(z["q_fixed"]), float(z["c_fixed"])

    labels = {1: (r"$c$", r"$q=n/N$", rf"$(c,q)$, $\Delta t={dt_fixed:g}$"),
              2: (r"$c$", r"$\Delta t$", rf"$(c,\Delta t)$, $q={q_fixed:g}$"),
              3: (r"$q=n/N$", r"$\Delta t$", rf"$(q,\Delta t)$, $c={c_fixed:g}$")}

    # paper style, 183 mm wide. Delta t* white dashed with a black outline (readable on
    # both the blue and the red side), gold q_eff = 1 with a thin black outline; one legend for the whole figure.
    with plt.style.context(ps.MAIN_CTX):
        outline = lambda w: [pe.Stroke(linewidth=w, foreground="black"), pe.Normal()]
        # square panels placed by hand (inches), with only the room needed by labels: no white space between them
        L, GAP, CB_GAP, CB_W, R = 0.46, 0.46, 0.07, 0.11, 0.62   # left, between panels, colour bar, right labels
        T, XL, LEG = 0.24, 0.40, 0.45                              # top (letters, titles), x labels, legend
        S = (ps.W2 - L - 2 * GAP - CB_GAP - CB_W - R) / 3          # side of each square panel
        H = T + 2 * S + XL + T + XL + LEG
        fig = plt.figure(figsize=(ps.W2, H))
        axes, caxes = np.empty((2, 3), object), []
        for r in range(2):
            y0 = H - T - S - r * (S + XL + T)
            for c in range(3):
                axes[r, c] = fig.add_axes([(L + c * (S + GAP)) / ps.W2, y0 / H, S / ps.W2, S / H])
            caxes.append(fig.add_axes([(L + 3 * S + 2 * GAP + CB_GAP) / ps.W2, y0 / H, CB_W / ps.W2, S / H]))
        lax = fig.add_axes([0, 0, 1, LEG / H])   # legend of the whole figure
        for row, (level, vmax) in enumerate(THRESHOLDS):
            norm = TwoSlopeNorm(vmin=0, vcenter=level, vmax=vmax)
            for col, p in enumerate([1, 2, 3]):
                ax = axes[row, col]
                x, y, E = z[f"p{p}_x"], z[f"p{p}_y"], z[f"p{p}_rel_err"]
                q_top = args.q_max_row1 if row == 0 else None
                x_max = q_top if (p == 3 and q_top) else x[-1]
                y_max = q_top if (p == 1 and q_top) else y[-1]
                xf, yf, Ef = smooth(x, y, E, x_max, y_max)
                ax.set_facecolor(UNDEF)   # NaN cells (kappa undefined) are transparent and show this colour
                im = ax.pcolormesh(xf, yf, np.clip(Ef, 0, vmax), cmap=CMAP, norm=norm, shading="nearest", rasterized=True)
                ax.contour(x, y, np.ma.masked_invalid(E), levels=[level], colors="black", linewidths=1.25)
                if p == 1:
                    ax.plot(z["c_fine"], z["q_star_fine"], color="gold", lw=1.6, path_effects=outline(2.6))
                elif p == 2:
                    ax.plot(z["c_fine"], z["p2_dt_star_0"], "--", color="white", lw=1.4, path_effects=outline(2.6))
                else:
                    ax.axhline(float(z["p3_dt_star_0"]), ls="--", color="white", lw=1.4, path_effects=outline(2.6))
                ax.set_xlim(edges(x)[0], x_max)
                ax.set_ylim(edges(y)[0] if p == 1 else 0, y_max)
                xl, yl, title = labels[p]
                ax.set_xlabel(xl)
                ax.set_ylabel(yl)
                ax.set_title(title, loc="right")
                ax.spines["top"].set_visible(True)
                ax.spines["right"].set_visible(True)
                ps.panel_label(ax, "abcdef"[3 * row + col])
            cb = fig.colorbar(im, cax=caxes[row], extend="max")
            cb.set_label(r"$\sigma_r/r$")
            cb.set_ticks(np.linspace(0, vmax, 6) if level == 1.0 else [0, 0.05, 0.1, 0.4, 0.7, 1.0])
            cb.outline.set_linewidth(0.6)

        handles = [Line2D([], [], color="black", lw=1.25,
                          label=r"boundary: $\sigma_r/r=1$ (a$-$c), $\sigma_r/r=0.1$ (d$-$f)"),
                   Patch(facecolor=UNDEF, edgecolor="none", label=r"$\kappa_{\rm cW}$ undefined"),
                   Line2D([], [], color="gold", lw=1.6, path_effects=outline(2.6), label=r"$q_{\rm eff}=1$"),
                   Line2D([], [], color="white", ls="--", lw=1.4, path_effects=outline(2.6),
                          label=r"$\Delta t^*=0.80/\bar r$")]
        lax.axis("off")
        lax.legend(handles=handles, loc="center", ncol=2, frameon=False, columnspacing=2.0)

        for fmt in args.format:
            path = os.path.join(args.output_dir, f"fig3_phase_diagram_{args.tag}.{fmt}")
            fig.savefig(path, dpi=args.dpi)
            print(f"Saved: {path}")

    # summaries
    q1, c1, E1 = z["p1_y"], z["p1_x"], z["p1_rel_err"]
    gold = pd.DataFrame({
        "c": c1, "r_bar": None, "q_star": z["p1_q_star"],
        "rel_err_on_gold": [np.interp(qs, q1, E1[:, j]) for j, qs in enumerate(z["p1_q_star"])],
        "q_diagnosis": boundary_q(q1, E1, 1.0), "q_resolution": boundary_q(q1, E1, 0.1),
    })
    if "p1_kappa_r_sem" in z:
        gold["max_kappa_sem_rel"] = np.nanmax(z["p1_kappa_r_sem"] / z["p1_kappa_r"], axis=0)
    gold["r_bar"] = (1 - np.sqrt(2) * c1) - c1 / np.sqrt(2) * float(z["mu_TW"]) * N ** (-2 / 3)

    pd.set_option("display.width", 250)
    print(gold.round(4).to_string(index=False))
    gold.to_csv(os.path.join(args.output_dir, f"fig3_gold_{args.tag}.csv"), index=False)
