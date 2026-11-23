"""
Shared look of the paper figures: style file, print widths, colours, panel letters, discrete colour bars.

    import sys; sys.path.insert(0, "<stability-inference-OU>/code/style"); import paperstyle as ps
    ps.use()
    fig = plt.figure(figsize=(ps.W2, h), layout="constrained")

Colours
  - simulations (Figs. 1, 2, 6, SI 1-2): magma 0.15-0.90 for the values of c, q_eff bands in SI Fig. 1;
  - real data (Figs. 4, 5, SI 3, 5, 6): one colour per dataset (EEG orange, financial pink, plankton turquoise) and
    its shades for everything about that dataset, plus markers, so they can be told apart also in grey scale;
  - theory, thresholds and reference lines: black or grey, distinguished by line style.
"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(HERE, "paper.mplstyle")
CTX = ["default", STYLE]   # plt.style.context(ps.CTX): matplotlib defaults + paper style, whatever the script set before
MAIN_CTX = CTX   # main and SI figures use the same sizes (paper.mplstyle)

MM = 1 / 25.4
W1 = 89 * MM      # one column
W2 = 183 * MM     # two columns

PANEL_FONTSIZE = 9

# reference lines (real-data figures and Fig. 3)
INK = "black"
GREY = "0.45"
LS_DIAGNOSIS = "-"        # sigma_r / r = 1
LS_RESOLUTION = "--"      # sigma_r / r = 0.1

# real data: one colour per dataset, chosen with different lightness so they stay distinct in grey scale
# (orange, pink, turquoise; markers differ too); everything about a dataset uses shades of its colour.
DATASET = {
    "eeg":      {"label": "EEG",       "color": "#ff9f1c", "marker": "o", "ls": "-"},   # bright orange
    "fin":      {"label": "Financial", "color": "#f768a1", "marker": "s", "ls": "-"},   # pink
    "plankton": {"label": "Plankton",  "color": "#81d8d0", "marker": "^", "ls": "-"},   # turquoise
}


def _mix(c, other, t):
    return tuple((1 - t) * np.array(matplotlib.colors.to_rgb(c)) + t * np.array(matplotlib.colors.to_rgb(other)))


def dataset_cmap(name):
    """Sequential map in the shades of one dataset: light tint -> main colour -> dark shade."""
    c = DATASET[name]["color"]
    if name == "eeg":   # warm yellow -> orange, no dark shades
        return LinearSegmentedColormap.from_list("eeg_shades", ["#ffd23f", "#ffa62b", "#f77f00"])
    if name == "fin":   # light pink -> pink -> violet
        return LinearSegmentedColormap.from_list("fin_shades", ["#fbb4c9", "#f768a1", "#c51b8a", "#7a0177"])
    return LinearSegmentedColormap.from_list(f"{name}_shades", [_mix(c, "white", 0.6), c, _mix(c, "black", 0.55)])


def shades(name, n=3):
    """n well separated shades of a dataset colour (dark to light), e.g. variants of one dataset in a panel."""
    return [dataset_cmap(name)(x) for x in np.linspace(1.0, 0.0 if name in ("fin", "eeg") else 0.3, n)]


VARIANT_MARKERS = ["o", "s", "^"]

# SI Fig. 1: bands of q_eff, bright green (small q_eff) through teal, cyan, blue to dark blue (large q_eff)
QEFF_COLORS = ["#00d26a", "#00b3a4", "#00b8f0", "#2a7fff", "#2f3fd8", "#141a7a"]

# simulations: colours of c in Figs. 1, 2, 6
def magma_c(n):
    return plt.get_cmap("magma")(np.linspace(0.15, 0.90, n))


def use():
    plt.style.use(CTX)


def panel_label(ax, letter, x=None, y=1.0, **kw):
    """Bold panel letter, 9 pt, above the top-left corner of the axes (offset in points, independent of size)."""
    from matplotlib.transforms import offset_copy
    tr = offset_copy(ax.transAxes, fig=ax.figure, x=-22 if x is None else x, y=4, units="points")
    ax.text(0, y, letter, transform=tr, fontsize=PANEL_FONTSIZE, fontweight="bold", va="bottom", ha="left", **kw)


def discrete_colorbar(fig, axes, values, colors, label, fmt="{:g}", max_ticks=None, **kw):
    """Colour bar with one block per value (e.g. the values of c), ticks at the block centres."""
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(values) + 1) - 0.5, cmap.N)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, **kw)
    idx = np.arange(len(values))
    if max_ticks and len(values) > max_ticks:
        step = int(np.ceil(len(values) / max_ticks))
        idx = idx[::step]
    cb.set_ticks(idx)
    cb.set_ticklabels([fmt.format(values[i]) for i in idx])
    cb.ax.tick_params(length=0)
    cb.outline.set_linewidth(0.6)
    cb.set_label(label)
    return cb


def save(fig, paths):
    for p in paths:
        fig.savefig(p)
        print(f"Saved: {p}")
