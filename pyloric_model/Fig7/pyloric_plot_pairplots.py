"""Copyright (c) 2026 Irene Elices. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

"""
pyloric_plot_pairplots.py
===========================
Generates the interval pairplot figures (normal R² pairplot and
lag-1 shift pairplot) for two pyloric simulations: one with
modulation and one without, from the pickles produced by
pyloric_precompute.py.

Each pairplot type is saved as two separate figures (one per
simulation).

Usage
---
    python pyloric_plot_pairplots.py --pkl-mod mod_data.pkl --pkl-nomod nomod_data.pkl
    python pyloric_plot_pairplots.py --pkl-mod mod_data.pkl --pkl-nomod nomod_data.pkl \\
        --out-pair-mod my_pairplot_mod.svg

Generated plots
---------------
    {base_mod}_pairplot.svg            pairplot of intervals (R²) — with modulation
    {base_mod}_pairplot_shift.svg      lag-1 pairplot — with modulation
    {base_nomod}_pairplot.svg          pairplot of intervals (R²) — without modulation
    {base_nomod}_pairplot_shift.svg    lag-1 pairplot — without modulation
"""

import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import MaxNLocator
from scipy import stats

# ─── global configuration ───────────────────────────────────────────────────
plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'


# ─── interval plots ──────────────────────────────────────────────────────────

def plot_pairplot(intervals, label="", out="pyloric_pairplot.eps"):
    order = ["Periodo", "LP_ABPD_delay", "ABPD_LP_delay", "LP_burst", "ABPD_burst", "PY_duration"]
    order = [k for k in order if k in intervals and len(intervals[k]) > 1]

    ylabels = {
        "Periodo":        "LP Period (ms)",
        "LP_ABPD_delay":  "LPPD delay (ms)",
        "ABPD_LP_delay":  "PDLP delay (ms)",
        "LP_burst":       "LP burst (ms)",
        "ABPD_burst":     "PD burst (ms)",
        "PY_duration":    "PY burst (ms)",
    }

    data = np.column_stack([intervals[k] * 1000.0 for k in order])
    n    = len(order)
    N    = data.shape[0]

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, axes = plt.subplots(n, n, figsize=(3.6 * n, 3.6 * n))
    if n == 1:
        axes = np.array([[axes]])

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            x  = data[:, col]
            y  = data[:, row]

            if row == col:
                ax.hist(x, bins=20, color="#818181", edgecolor="white")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            elif row > col:
                ax.scatter(x, y, s=5, color="#555555")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            else:
                slope, intercept, r, p, _ = stats.linregress(x, y)
                r2     = r**2
                col_bg = cmap(float(np.clip(r2, 0, 1)))
                ax.set_facecolor(col_bg)
                lum = 0.299*col_bg[0] + 0.587*col_bg[1] + 0.114*col_bg[2]
                tc  = "white" if lum < 0.5 else "black"
                ax.text(0.5, 0.5, f"{r2:.2f}",
                        transform=ax.transAxes, fontsize=30,
                        ha="center", va="center", color=tc)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            if row == n - 1:
                ax.set_xlabel(ylabels[order[col]])
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(ylabels[order[row]])
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=20)

    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    plt.savefig(out)
    plt.close(fig)


def plot_pairplot_shift(intervals, label="", out="pyloric_pairplot_shift.eps"):
    order = ["Periodo", "LP_ABPD_delay", "ABPD_LP_delay", "LP_burst", "ABPD_burst", "PY_duration"]
    order = [k for k in order if k in intervals and len(intervals[k]) > 2]

    ylabels = {
        "Periodo":        "LP Period (ms)",
        "LP_ABPD_delay":  "LPPD delay (ms)",
        "ABPD_LP_delay":  "PDLP delay (ms)",
        "LP_burst":       "LP burst (ms)",
        "ABPD_burst":     "PD burst (ms)",
        "PY_duration":    "PY burst (ms)",
    }

    arrays = {k: intervals[k] * 1000.0 for k in order}
    X = np.column_stack([arrays[k][:-1] for k in order])
    Y = np.column_stack([arrays[k][1:]  for k in order])

    n = len(order)

    cmap = plt.cm.Blues
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    fig, axes = plt.subplots(n, n, figsize=(3.6 * n, 3.6 * n))
    if n == 1:
        axes = np.array([[axes]])

    for row in range(n):
        for col in range(n):
            ax = axes[row, col]
            if row == col:
                x = X[:, col]; y = Y[:, row]
                ax.scatter(x, y, s=5, color="#555555")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            elif row > col:
                x = X[:, col]; y = Y[:, row]
                ax.scatter(x, y, s=5, color="#555555")
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            else:
                x = X[:, row]; y = Y[:, col]
                slope, intercept, r, p, _ = stats.linregress(x, y)
                r2     = r**2
                col_bg = cmap(float(np.clip(r2, 0, 1)))
                ax.set_facecolor(col_bg)
                lum = 0.299*col_bg[0] + 0.587*col_bg[1] + 0.114*col_bg[2]
                tc  = "white" if lum < 0.5 else "black"
                ax.text(0.5, 0.5, f"{r2:.2f}",
                        transform=ax.transAxes, fontsize=30,
                        ha="center", va="center", color=tc)
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

            if row == n - 1:
                ax.set_xlabel(f"{ylabels[order[col]]}")
            else:
                ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(f"{ylabels[order[row]]}")
            else:
                ax.tick_params(labelleft=False)
            ax.tick_params(labelsize=20)

    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    plt.savefig(out)
    plt.close(fig)


# ─── argparse ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    
    p.add_argument("--pkl-mod", required=True,
                   help="Pickle generated by pyloric_data.py (with-modulation simulation)")
    p.add_argument("--pkl-nomod", required=True,
                   help="Pickle generated by pyloric_data.py (without-modulation simulation)")
    p.add_argument("--out-pair-mod",   default=None,
                   help="Path for the with-modulation pairplot (default: {base_mod}_pairplot.svg)")
    p.add_argument("--out-shift-mod",  default=None,
                   help="Path for the with-modulation shift pairplot (default: {base_mod}_pairplot_shift.svg)")
    p.add_argument("--out-pair-nomod", default=None,
                   help="Path for the without-modulation pairplot (default: {base_nomod}_pairplot.svg)")
    p.add_argument("--out-shift-nomod", default=None,
                   help="Path for the without-modulation shift pairplot (default: {base_nomod}_pairplot_shift.svg)")
    return p.parse_args()


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n  Loading pickle (with modulation): {args.pkl_mod}")
    with open(args.pkl_mod, "rb") as f:
        data_mod = pickle.load(f)
    intervals_mod = data_mod["intervals"]
    label_mod     = data_mod["label"]
    base_mod      = label_mod
    print(f"  Simulation: {label_mod}")

    print(f"\n  Loading pickle (without modulation): {args.pkl_nomod}")
    with open(args.pkl_nomod, "rb") as f:
        data_nomod = pickle.load(f)
    intervals_nomod = data_nomod["intervals"]
    label_nomod     = data_nomod["label"]
    base_nomod      = label_nomod
    print(f"  Simulation: {label_nomod}")

    # ── Pairplots — with modulation ───────────────────────────────────────────
    print("\n  Interval pairplot (with modulation)...")
    plot_pairplot(intervals_mod, label=label_mod,
                  out=args.out_pair_mod or f"{base_mod}_pairplot.svg")

    print("  Lag-1 shift pairplot (with modulation)...")
    plot_pairplot_shift(intervals_mod, label=label_mod,
                        out=args.out_shift_mod or f"{base_mod}_pairplot_shift.svg")

    # ── Pairplots — without modulation ────────────────────────────────────────
    print("\n  Interval pairplot (without modulation)...")
    plot_pairplot(intervals_nomod, label=label_nomod,
                  out=args.out_pair_nomod or f"{base_nomod}_pairplot.svg")

    print("  Lag-1 shift pairplot (without modulation)...")
    plot_pairplot_shift(intervals_nomod, label=label_nomod,
                        out=args.out_shift_nomod or f"{base_nomod}_pairplot_shift.svg")

    print("\n  Done.")


if __name__ == "__main__":
    main()
