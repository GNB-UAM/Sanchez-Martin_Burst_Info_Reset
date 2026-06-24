"""Copyright (c) 2026 Irene Elices. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

"""
pyloric_plot_currents.py
==========================
Generates the per-cycle current sum figures and the cycle-pair
current figures (cycles 10 and 90) for a modulated (with-modulation)
pyloric simulation, from the pickle produced by pyloric_precompute.py.

Usage
---
    python pyloric_plot_currents.py ../simulation_circuit0_mod_data.pkl
    python pyloric_plot_currents.py data.pkl --cycle-pairs 10 90
    python pyloric_plot_currents.py data.pkl --out-cycle-sums-period sums_period.svg

Generated plots
---------------
    {base}_curr_cycle_lppddelay.svg    per-cycle current sum — LPPD delay
    {base}_cycle_pair_{N}.svg          cycle pair (one figure per N in --cycle-pairs)
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# ─── global configuration ───────────────────────────────────────────────────
plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'

NEURON_NAMES = ["AB/PD", "LP", "PY"]

# Membrane area used by simulator.pyx to normalize membrane_conds and
# synaptic_conds before saving them (density, mS/cm² -> µA/cm² currents).
# I_synaptic in the raw currents .h5 inherits that normalization;
# multiplying back by this area converts it to absolute µA.
AREA_CM2 = 0.628e-3


# ─── per-cycle current plots ─────────────────────────────────────────────────

def plot_current_sums_per_cycle(sums, valid_cycles, label="",
                                out="pyloric_curr_cycle_sums.png",
                                window='period'):
    """
    sums: dict with keys 'ion_per_cycle', 'syn_per_cycle', 'ext_per_cycle'
          as saved by pyloric_precompute.py in curr_sums[window].
    """
    WINDOW_TITLES = {
        'period'  : 'Full period [LP_on -> LP_on_next]',
        'lp_burst': 'LP burst [LP_on -> LP_off]',
        'lp_pd'   : 'LPPD delay [LP_off -> ABPD_on]',
        'abpd_end': 'PDLP delay [ABPD_off -> LP_on_next]',
    }

    ion_per_cycle = sums['ion_per_cycle']
    syn_per_cycle = sums['syn_per_cycle']
    ext_per_cycle = sums['ext_per_cycle']

    n_cyc    = ion_per_cycle.shape[1]
    has_iext = (ext_per_cycle is not None
                and not np.all(np.isnan(ext_per_cycle))
                and np.any(ext_per_cycle != 0))

    if n_cyc == 0:
        print(f"  [curr_per_cycle/{window}] No cycles in the sums.")
        return

    # valid_cycles is only needed for the durations in the top panel
    # — reconstructed from ion_per_cycle shape, the full array isn't needed

    cycle_idx = np.arange(n_cyc)
    C_ION = "#308b7a"
    C_SYN = "#6d34ad"
    C_EXT = "#000000"

    show_ext = (has_iext and ext_per_cycle is not None
                and not np.all(np.isnan(ext_per_cycle))
                and np.any(ext_per_cycle != 0))
    n_rows        = 3 + 1 + (1 if show_ext else 0)
    height_ratios = ([1] if show_ext else []) + [2, 2, 2, 2]

    fig, axes = plt.subplots(n_rows, 1,
                             figsize=(19, 4 * 3 + 3 + (2 if show_ext else 0)),
                             sharex=True,
                             gridspec_kw={'hspace': 0.08,
                                          'height_ratios': height_ratios,
                                          'left': 0.12, 'right': 0.88,
                                          'top': 0.93, 'bottom': 0.07})
    if n_rows == 1:
        axes = [axes]

    # ── I_ext row ─────────────────────────────────────────────────────────────
    if show_ext:
        ax_e = axes[0]
        for ni in range(3):
            row = ext_per_cycle[ni]
            if np.all(np.isnan(row)) or np.nansum(np.abs(row)) == 0:
                continue
            ax_e.plot(cycle_idx, row,
                      color='black', lw=0.9, marker='o', markersize=2,
                      markeredgewidth=0, alpha=0.85,
                      label=f'I_ext {NEURON_NAMES[ni]}')
        ax_e.yaxis.grid(True, color='#EEEEEE', lw=0.4)
        ax_e.set_axisbelow(True)
        ax_e.set_ylabel('Σ I_ext (µA)', fontsize=20, color=C_EXT, labelpad=10)
        ax_e.tick_params(labelsize=20)
        ax_e.spines['top'].set_visible(False)
        ax_e.spines['right'].set_visible(False)
        #ax_e.legend(loc='upper right', fontsize=15, frameon=False, ncol=3)

    # ── durations panel ──────────────────────────────────────────────────────
    ax_dur = axes[1 if show_ext else 0]
    ax_per = ax_dur.twinx()
    C_PER  = '#7f7f7f'
    C_LPPD = "#ff36daff"
    C_PY   = "#2ca02c"

    dur_lppd = np.full(n_cyc, np.nan)
    dur_per  = np.full(n_cyc, np.nan)
    dur_py   = np.full(n_cyc, np.nan)
    for ci, cyc in enumerate(valid_cycles):
        lp_on,   lp_off   = cyc['lp']
        abpd_on, abpd_off = cyc['abpd']
        py_on,   py_off   = cyc['py']
        lp_next           = cyc['lp_next_on']
        dur_lppd[ci] = (abpd_on - lp_off) * 1000.0
        dur_per[ci]  = (lp_next - lp_on)  * 1000.0
        dur_py[ci]   = (py_off  - py_on)  * 1000.0

    l_lppd, = ax_dur.plot(cycle_idx, dur_lppd, color=C_LPPD, lw=0.9,
                          marker='o', markersize=2, markeredgewidth=0,
                          label='LPPD delay')
    l_py,   = ax_dur.plot(cycle_idx, dur_py,   color=C_PY,   lw=0.9,
                          marker='o', markersize=2, markeredgewidth=0,
                          label='Burst PY')
    l_per,  = ax_per.plot(cycle_idx, dur_per,  color=C_PER,  lw=0.9,
                          marker='o', markersize=2, markeredgewidth=0,
                          label='Period')

    ax_dur.yaxis.grid(True, color='#EEEEEE', lw=0.4)
    ax_dur.set_axisbelow(True)
    ax_dur.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    ax_per.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    ax_dur.set_ylabel('LPPD delay\nPY burst (ms)', fontsize=20, color='k', labelpad=10)
    ax_dur.tick_params(labelsize=20)
    ax_dur.spines['top'].set_visible(False)
    ax_per.set_ylabel('Period (ms)', fontsize=20, color=C_PER, labelpad=10)
    ax_per.tick_params(labelsize=20, colors=C_PER)
    ax_per.spines['right'].set_edgecolor(C_PER)
    ax_per.spines['top'].set_visible(False)
    ax_dur.legend(handles=[l_lppd, l_py, l_per], loc='upper left',
                  fontsize=15, frameon=False, ncol=3)

    # ── per-neuron panels ────────────────────────────────────────────────────
    neuron_start_ax = 2 if show_ext else 1
    for ni in range(3):
        ax   = axes[neuron_start_ax + ni]
        ax_r = ax.twinx()

        l_ion, = ax.plot(cycle_idx, -ion_per_cycle[ni],
                         color=C_ION, lw=1.2, marker='o', markersize=2.5,
                         markeredgewidth=0, label='Σ I_ion')
        l_syn, = ax_r.plot(cycle_idx, syn_per_cycle[ni],
                           color=C_SYN, lw=1.2, marker='o', markersize=2.5,
                           markeredgewidth=0, label='Σ I_syn')

        ax.yaxis.grid(True, color='#EEEEEE', lw=0.4)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        ax_r.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax_r.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        ax.set_ylabel(f"{NEURON_NAMES[ni]} Σ I_ion (µA)",
                      fontsize=20, color=C_ION, labelpad=10)
        ax.tick_params(labelsize=20, colors=C_ION)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_edgecolor(C_ION)
        ax_r.set_ylabel(f"{NEURON_NAMES[ni]} Σ I_syn (µA)", fontsize=20, color=C_SYN, labelpad=10)
        ax_r.tick_params(labelsize=20, colors=C_SYN)
        ax_r.spines['right'].set_edgecolor(C_SYN)
        ax_r.spines['top'].set_visible(False)
        #ax.legend(handles=[l_ion, l_syn], loc='upper right',
        #          fontsize=15, frameon=False, ncol=2)

        if ni == 0:
            ax.set_title(
                f"Per-cycle current sum — {WINDOW_TITLES[window]}\n{label}",
                fontsize=10, fontweight='bold')

    axes[-1].tick_params(labelsize=20, color='black')
    axes[-1].set_xlabel("Cycle #", fontsize=20, color='black')
    plt.savefig(out)
    plt.close(fig)


# ─── currents file loading (shared, single read) ─────────────────────────────

def load_currents_file(curr_h5_path):
    """
    Loads the synaptic-currents file once. Returns a dict with
    I_synaptic (absolute µA), t_curr_s (seconds), conn_labels, and
    conn_post, so callers never re-read the file from disk.

    I_synaptic is stored in the raw .h5 as a current density (µA/cm²),
    since it comes from synaptic_conds, which the simulator normalizes
    by membrane area before saving. We revert that normalization here.
    """
    curr_path = str(curr_h5_path)
    if curr_path.endswith('.h5') or curr_path.endswith('.hdf5'):
        import h5py
        with h5py.File(curr_path, 'r') as hf:
            I_synaptic  = hf['I_synaptic'][:]
            t_curr_ms   = hf['t'][:]
            conn_labels = list(hf['conn_labels'][:].astype(str))
            conn_post   = list(hf['conn_post'][:].astype(int))
    else:
        data        = np.load(curr_path, allow_pickle=False)
        I_synaptic  = data['I_synaptic']
        t_curr_ms   = data['t']
        conn_labels = list(data['conn_labels'].astype(str))
        conn_post   = list(data['conn_post'].astype(int))

    I_synaptic = I_synaptic * AREA_CM2   # density (µA/cm²) -> absolute (µA)

    return {
        "I_synaptic":  I_synaptic,
        "t_curr_s":    t_curr_ms / 1000.0,
        "conn_labels": conn_labels,
        "conn_post":   conn_post,
    }


# ─── min/max & min/mean percentage summary across all cycles ────────────────

def compute_synaptic_minmax_summary(cycles_data, currents):
    """
    For every valid cycle and every synaptic connection, computes
    (min_abs, max_abs, mean_abs) of |I_synaptic| over the full cycle
    window [lp_on, lp_next_on], then averages the resulting
    min/max and min/mean percentages across all cycles.

    `currents` is the dict returned by load_currents_file().

    Returns a dict keyed by connection label:
        {label: {"min_max_pct_mean": ..., "min_mean_pct_mean": ..., "n_cycles": ...}}
    """
    I_synaptic  = currents["I_synaptic"]
    t_curr_s    = currents["t_curr_s"]
    conn_labels = currents["conn_labels"]
    conn_post   = currents["conn_post"]

    I_abs       = np.abs(I_synaptic)          # (n_conn, n_t) — computed once
    n_conn      = I_abs.shape[0]

    # accumulators per connection
    sum_min_max_pct  = np.zeros(n_conn)
    sum_min_mean_pct = np.zeros(n_conn)
    n_valid          = np.zeros(n_conn, dtype=int)

    for cyc in cycles_data:
        t_lo = cyc['lp'][0]
        t_hi = cyc['lp_next_on']
        # searchsorted on the sorted time axis avoids a full boolean scan per cycle
        i0, i1 = np.searchsorted(t_curr_s, (t_lo, t_hi))
        if i1 <= i0:
            continue

        window   = I_abs[:, i0:i1]
        min_abs  = window.min(axis=1)
        max_abs  = window.max(axis=1)
        mean_abs = window.mean(axis=1)

        valid_max  = max_abs > 0
        valid_mean = mean_abs > 0

        sum_min_max_pct[valid_max]   += (min_abs[valid_max]  * 100) / max_abs[valid_max]
        sum_min_mean_pct[valid_mean] += (min_abs[valid_mean] * 100) / mean_abs[valid_mean]
        n_valid[valid_max & valid_mean] += 1

    summary = {}
    for k in range(n_conn):
        if n_valid[k] == 0:
            continue
        summary[conn_labels[k]] = {
            "min_max_pct_mean":  sum_min_max_pct[k]  / n_valid[k],
            "min_mean_pct_mean": sum_min_mean_pct[k] / n_valid[k],
            "n_cycles":          int(n_valid[k]),
            "post":              conn_post[k],
        }
    return summary


def print_synaptic_minmax_summary(summary):
    print("\n  Synaptic current min/max & min/mean — averaged over all cycles:")
    # group by receiving neuron (post), not by the order connections were stored in
    items_sorted = sorted(summary.items(), key=lambda kv: kv[1]["post"])
    current_post = None
    for lbl, s in items_sorted:
        if s["post"] != current_post:
            current_post = s["post"]
            print(f"    [{NEURON_NAMES[current_post]} receives]")
        print(f"      {lbl}: min/max = {s['min_max_pct_mean']:.2f}%, "
              f"min/mean = {s['min_mean_pct_mean']:.2f}%  (n={s['n_cycles']} cycles)")


# ─── cycle-pair plot ──────────────────────────────────────────────────────────

def plot_cycle_pair_currents(cycles_data, currents,
                             cycle_idx=0, label="",
                             out="pyloric_cycle_pair_currents.png"):
    valid  = cycles_data
    cyc_a  = valid[cycle_idx]
    cyc_b  = valid[cycle_idx + 1]
 
    t_win_lo = cyc_a['lp'][0]
    t_win_hi = cyc_b['lp_next_on']
    margin   = (t_win_hi - t_win_lo) * 0.04
    t_win_lo -= margin
    t_win_hi += margin
 
    I_synaptic  = currents["I_synaptic"]
    I_synaptic  = I_synaptic * 1e3  # nA
    t_curr_s    = currents["t_curr_s"]
    conn_labels = currents["conn_labels"]
    conn_post   = currents["conn_post"]
 
    mask_c     = (t_curr_s >= t_win_lo) & (t_curr_s <= t_win_hi)
    mask_c_idx = np.where(mask_c)[0]
    t_c        = t_curr_s[mask_c_idx]
 
    syn_by_post = {0: [], 1: [], 2: []}
    for k, (lbl, post) in enumerate(zip(conn_labels, conn_post)):
        syn_by_post[post].append(k)
 
    ANN = {
        "ABPD_burst":    "#1f77b4",
        "LP_burst":      "#ff7f0e",
        "PY_duration":   "#2ca02c",
    }
 
    def _annotate_cycle(ax, cyc):
        lp_on,   lp_off   = cyc['lp']
        abpd_on, abpd_off = cyc['abpd']
        py_on,   py_off   = cyc['py']

        ax.axvspan(abpd_on, abpd_off, alpha=0.18, color=ANN["ABPD_burst"], zorder=1)
        ax.axvspan(lp_on,   lp_off,   alpha=0.18, color=ANN["LP_burst"],   zorder=1)
        ax.axvspan(py_on,   py_off,   alpha=0.18, color=ANN["PY_duration"],zorder=1)
        ax.axvline(lp_on, color="#AAAAAA", lw=0.8, ls=":", zorder=2)

    # ── compute ymax per neuron to decide on broken-axis mode ────────────────
    cmap_t20  = plt.colormaps['tab20b']
    ymax_by_ni = {}
    for ni in range(3):
        if syn_by_post[ni]:
            y = np.concatenate([I_synaptic[k][mask_c_idx] for k in syn_by_post[ni]])
            ymax_by_ni[ni] = float(np.nanmax(y))
        else:
            ymax_by_ni[ni] = 0.0
 
    # broken axis per neuron, independently
    use_broken = {ni: ymax_by_ni[ni] > 7 for ni in range(3)}
 
    # ── create figure ─────────────────────────────────────────────────────────
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
 
    n_rows = 3
    d_cut  = 0.015
    fig    = plt.figure(figsize=(8.25, 5.25 * n_rows))
    outer  = GridSpec(n_rows, 1, figure=fig,
                      hspace=0.1, left=0.22, right=0.95, top=0.90, bottom=0.10)
    syn_axes = {}
    ref_ax   = None
    for ni in range(n_rows):
        if use_broken[ni]:
            sub_gs    = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[ni],
                                               height_ratios=[3, 7], hspace=0.1)
            sharex_kw = dict(sharex=ref_ax) if ref_ax is not None else {}
            ax_hi     = fig.add_subplot(sub_gs[0], **sharex_kw)
            ax_lo     = fig.add_subplot(sub_gs[1], sharex=ax_hi)
            if ref_ax is None:
                ref_ax = ax_hi
            for sp in ['top', 'bottom', 'right']:
                ax_hi.spines[sp].set_visible(False)
            ax_hi.spines['left'].set_visible(True)
            ax_hi.tick_params(bottom=False, labelbottom=False, top=False, labeltop=False)
            ax_hi.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
            for sp in ['top', 'right']:
                ax_lo.spines[sp].set_visible(False)
            #ax_lo.tick_params(labelsize=12)
            ax_lo.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
            kw = dict(color='k', clip_on=False, lw=0.8)
            ax_hi.plot([-d_cut, +d_cut], [-d_cut, +d_cut], transform=ax_hi.transAxes, **kw)
            ax_hi.plot([1-d_cut, 1+d_cut], [-d_cut, +d_cut], transform=ax_hi.transAxes, **kw)
            ax_lo.plot([-d_cut, +d_cut], [1-d_cut, 1+d_cut], transform=ax_lo.transAxes, **kw)
            ax_lo.plot([1-d_cut, 1+d_cut], [1-d_cut, 1+d_cut], transform=ax_lo.transAxes, **kw)
            syn_axes[ni] = (ax_lo, ax_hi)
        else:
            sharex_kw = dict(sharex=ref_ax) if ref_ax is not None else {}
            ax_lo = fig.add_subplot(outer[ni], **sharex_kw)
            if ref_ax is None:
                ref_ax = ax_lo
            ax_lo.spines['top'].set_visible(False)
            ax_lo.spines['right'].set_visible(False)
            ax_lo.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both', integer=True))
            syn_axes[ni] = (ax_lo, None)
        if ni == 0:
            syn_axes[ni][0].set_title("I_synaptic (nA)", fontsize=10, fontweight='bold')
 
    # ── plotting ─────────────────────────────────────────────────────────────
    for ni in range(n_rows):
        ax_lo, ax_hi = syn_axes[ni]
        syn_idxs     = syn_by_post[ni]
        y_all        = []
 
        if syn_idxs:
            N_s    = len(syn_idxs)
            colors = {k: cmap_t20(i / max(N_s - 1, 1)) for i, k in enumerate(syn_idxs)}

            targets = (ax_lo, ax_hi) if use_broken[ni] else (ax_lo,)
            for ax_b in targets:
                for k in syn_idxs:
                    ax_b.plot(t_c, I_synaptic[k][mask_c_idx],
                              lw=0.9, label=conn_labels[k], color=colors[k])
            y_all = np.concatenate([I_synaptic[k][mask_c_idx] for k in syn_idxs])
 
        if use_broken[ni]:
            ymax_ni = ymax_by_ni[ni]
            if len(y_all) > 0:
                ymin   = float(np.nanmin(y_all))
                margin = abs(ymin) * 0.05 + 0.1
                ax_lo.set_ylim(ymin - margin, 3)
                ax_hi.set_ylim(ymax_ni - 2, ymax_ni + 0.1)
            else:
                ax_lo.set_ylim(-1, 3)
                ax_hi.set_ylim(3, 6)
            for ax_b in (ax_lo, ax_hi):
                ax_b.axhline(0, color='#CCCCCC', lw=0.4)
                ax_b.yaxis.grid(True, color='#EEEEEE', lw=0.4)
                ax_b.set_axisbelow(True)
            ax_hi.legend(loc='upper right', fontsize=12, frameon=False)
            ax_lo.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, min_n_ticks=3))
            ax_lo.set_ylabel("Currents (nA)", rotation=90)
        else:
            ax_lo.axhline(0, color='#CCCCCC', lw=0.4)
            ax_lo.yaxis.grid(True, color='#EEEEEE', lw=0.4)
            ax_lo.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True, min_n_ticks=3))
            ax_lo.set_axisbelow(True)
            ax_lo.set_ylabel("Currents (nA)", rotation=90)
            ax_lo.legend(loc='upper right', fontsize=12, frameon=False)
 
    # ── interval annotations ──────────────────────────────────────────────────
    for ni in range(n_rows):
        ax_lo, ax_hi = syn_axes[ni]
        targets = (ax_lo, ax_hi) if use_broken[ni] else (ax_lo,)
        for ax_b in targets:
            ax_b.autoscale_view(scaley=False)
        for ax_b in targets:
            for cyc in (cyc_a, cyc_b):
                _annotate_cycle(ax_b, cyc)
 
    syn_axes[n_rows - 1][0].set_xlabel('Time (s)')
 
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Patch(facecolor=ANN["ABPD_burst"],  alpha=0.5, label="AB/PD burst"),
        Patch(facecolor=ANN["LP_burst"],    alpha=0.5, label="LP burst"),
        Patch(facecolor=ANN["PY_duration"], alpha=0.5, label="PY burst"),
        Line2D([0], [0], color="#AAAAAA", lw=0.8, ls=":", label="LP onset (cycle)"),
    ], loc='lower center', ncol=2, fontsize=10, framealpha=0.9,
       bbox_to_anchor=(0.5, 0.0))
 
    fig.suptitle(
        f"I_synaptic — cycles {cycle_idx} and {cycle_idx+1} — {label}\n"
        f"(t = {cyc_a['lp'][0]:.3f} … {cyc_b['lp_next_on']:.3f} s)",
        fontsize=11, fontweight='bold')
 
    plt.savefig(out)
    plt.close(fig)


# ─── argparse ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pkl_file", nargs="?", default="../simulation_circuit0_ramp_ABPD_amp0.00010-desc-15_dur140s_data.pkl",
                   help="Pickle generated by pyloric_data.py (with-modulation simulation). ")

    p.add_argument("--out-cycle-sums-period", default=None)
    p.add_argument("--out-cycle-sums-burst",  default=None)
    p.add_argument("--out-cycle-sums-delay",  default=None)
    p.add_argument("--cycle-pairs", type=int, nargs="+", default=[10, 90],
                   help="First-cycle indices (0-based) for the cycle-pair plots (default: 10 90)")
    return p.parse_args()


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n  Loading pickle: {args.pkl_file}")
    with open(args.pkl_file, "rb") as f:
        data = pickle.load(f)

    valid_cycles = data["valid_cycles"]
    label        = data["label"]
    curr_h5      = data["curr_h5"]

    pkl_dir = os.path.dirname(os.path.abspath(args.pkl_file))
    curr_h5 = os.path.join(pkl_dir, os.path.basename(curr_h5))

    base = label
    print(f"  Simulation: {label}")
    print(f"  Valid cycles: {len(valid_cycles)}")

    # ── Load the currents file once, reused by every step below ──────────────
    currents = None
    if curr_h5 is not None:
        print(f"\n  Loading currents file: {curr_h5}")
        currents = load_currents_file(curr_h5)

    # ── Synaptic min/max & min/mean summary across all cycles ────────────────
    if currents is not None and len(valid_cycles) > 0:
        print("\n  Computing synaptic current min/max summary over all cycles...")
        summary = compute_synaptic_minmax_summary(valid_cycles, currents)
        print_synaptic_minmax_summary(summary)

    # ── Per-cycle currents ────────────────────────────────────────────────────
    curr_sums = data.get("curr_sums")
    if curr_sums is not None:
        print("\n  Per-cycle current sum plots...")

        print("  Per-cycle sum — LP->AB/PD delay...")
        plot_current_sums_per_cycle(
            curr_sums["lp_pd"], valid_cycles, label=label,
            out=args.out_cycle_sums_delay or f"{base}_curr_cycle_lppddelay.svg",
            window="lp_pd")

        # ── Cycle-pair plots ───────────────────────────────────────────────────
        n_avail = len(valid_cycles)
        print(f"\n  [cycle_pair] Available valid cycles: {n_avail}")
        if n_avail < 2:
            print("  [cycle_pair] Not enough cycles.")
        else:
            for cidx in args.cycle_pairs:
                if cidx < 0 or cidx + 1 >= n_avail:
                    print(f"  [cycle_pair] cycle_idx={cidx} out of range (0 … {n_avail-2}).")
                    continue
                out_cp = f"{base}_cycle_pair_{cidx}.svg"
                print(f"  Plotting currents for cycles {cidx}/{cidx+1} → {out_cp}")
                import traceback as _tb
                try:
                    plot_cycle_pair_currents(
                        valid_cycles, currents,
                        cycle_idx=cidx,
                        label=label,
                        out=out_cp)
                except Exception as _e:
                    print(f"  [cycle_pair] EXCEPTION: {_e}")
                    _tb.print_exc()

    elif curr_h5 is None:
        print("\n  [info] No curr_sums in the pickle and no currents file — "
              "skipping current plots.")

    print("\n  Done.")


if __name__ == "__main__":
    main()