"""Copyright (c) 2026 Irene Elices. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

"""
pyloric_plot_voltage.py
========================
Generates the voltage trace figures (full trace + 2 zooms) for a
modulated (with-modulation) pyloric simulation, from the pickle
produced by pyloric_precompute.py.

Usage
---
    python pyloric_plot_voltage.py simulation_circuit0_mod_data.pkl
    python pyloric_plot_voltage.py data.pkl --plot-decimate 20

Generated plots
---------------
    {base}_voltage_traces.svg          full voltage traces
    {base}_voltage_traces_zoom1.svg    zoom 1
    {base}_voltage_traces_zoom2.svg    zoom 2
"""

import os
import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ─── global configuration ───────────────────────────────────────────────────
plt.rcParams.update({'font.size': 20})
plt.rcParams['svg.fonttype'] = 'none'

NEURON_NAMES = ["AB/PD", "LP", "PY"]
PALETTE      = ["#1f77b4", "#ff7f0e", "#2ca02c"]


# ─── voltage plot ────────────────────────────────────────────────────────────
# (re-reads the file directly to avoid loading the full array into memory)

def plot_voltage_traces(path, label="", out="pyloric_voltage_traces.png",
                        decimate=10, I_ext=None,
                        t_start_ms=None, t_end_ms=None,
                        line_width=0.5, figsize=(10, 7)):
    sl = slice(None, None, decimate)
    if path.endswith('.h5') or path.endswith('.hdf5'):
        import h5py
        with h5py.File(path, 'r') as hf:
            dt_h5 = float(hf.attrs['dt'])
            N_tot = hf['t'].shape[0]
            if t_start_ms is not None or t_end_ms is not None:
                i0 = max(0, int((t_start_ms or 0) / dt_h5))
                i1 = min(N_tot, int((t_end_ms or N_tot * dt_h5) / dt_h5) + 1)
            else:
                i0, i1 = 0, N_tot
            idx     = slice(i0, i1, decimate)
            t_vec   = hf['t'][idx]
            voltage = np.stack([hf['V_ABPD'][idx], hf['V_LP'][idx], hf['V_PY'][idx]])
    else:
        d = np.load(path, allow_pickle=True)
        t_ms_full = d['t']
        if t_start_ms is not None or t_end_ms is not None:
            t0   = t_start_ms if t_start_ms is not None else t_ms_full[0]
            t1   = t_end_ms   if t_end_ms   is not None else t_ms_full[-1]
            mask = (t_ms_full >= t0) & (t_ms_full <= t1)
            t_vec   = t_ms_full[mask][::decimate]
            voltage = np.stack([d['V_ABPD'][mask][::decimate],
                                d['V_LP'][mask][::decimate],
                                d['V_PY'][mask][::decimate]])
        else:
            t_vec   = d['t'][sl]
            voltage = np.stack([d['V_ABPD'][sl], d['V_LP'][sl], d['V_PY'][sl]])

    if t_vec[-1] > 1e4:
        t_vec = t_vec / 1000.0

    has_Iext = I_ext is not None and np.any(I_ext != 0)
    n_panels = 4 if has_Iext else 3

    # height proportional to the number of panels
    w, h_per_panel = figsize[0], figsize[1] / 4
    auto_figsize = (w, h_per_panel * n_panels)

    fig, axes = plt.subplots(n_panels, 1, figsize=auto_figsize,
                             sharex=True, layout='constrained')
    for j in range(3):
        axes[j].plot(t_vec, voltage[j], lw=line_width, color=PALETTE[j])
        axes[j].set_ylabel(f'{NEURON_NAMES[j]} (mV)', fontsize=20, rotation=90)
        axes[j].spines['top'].set_visible(False)
        axes[j].spines['right'].set_visible(False)
        axes[j].yaxis.grid(True, color='#EEEEEE', lw=0.4)
        axes[j].set_axisbelow(True)

    if has_Iext:
        ax = axes[3]
        # crop I_ext to the same window as t_vec
        N_tot_iext = I_ext.shape[1]
        dt_iext    = None  # inferred from path if h5
        if path.endswith('.h5') or path.endswith('.hdf5'):
            import h5py as _h5_iext
            with _h5_iext.File(path, 'r') as _hf:
                _dt = float(_hf.attrs['dt'])
            i0_i = max(0, int((t_start_ms or 0) / _dt)) if t_start_ms is not None else 0
            i1_i = min(N_tot_iext, int((t_end_ms or N_tot_iext * _dt) / _dt) + 1) if t_end_ms is not None else N_tot_iext
        else:
            i0_i, i1_i = 0, N_tot_iext
        I_dec = I_ext[:, i0_i:i1_i:decimate] * 1000000
        n     = min(I_dec.shape[1], len(t_vec))
        for j in range(3):
            if np.any(I_dec[j, :n] != 0):
                ax.plot(t_vec[:n], I_dec[j, :n], lw=line_width,
                        color=PALETTE[j], label=NEURON_NAMES[j])
        ax.set_ylabel('I_ext (pA)', fontsize=20, rotation=90)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, color='#EEEEEE', lw=0.4)
        ax.set_axisbelow(True)

    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1].set_xlabel('Time (s)', fontsize=20)
    fig.suptitle(f'Voltage — {label}  (dec={decimate})', fontsize=11, fontweight='bold')
    plt.savefig(out)
    plt.close(fig)


# ─── argparse ────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pkl_file",
                   help="Pickle generated by pyloric_data.py (with-modulation simulation)")
    p.add_argument("--plot-decimate", type=int, default=10,
                   help="Subsampling factor for voltage traces (default: 10)")
    p.add_argument("--zoom1-start", type=float, default=15100,
                   help="Zoom 1 window start (ms, default: 15100)")
    p.add_argument("--zoom1-end",   type=float, default=19300,
                   help="Zoom 1 window end (ms, default: 19300)")
    p.add_argument("--zoom2-start", type=float, default=133300,
                   help="Zoom 2 window start (ms, default: 133300)")
    p.add_argument("--zoom2-end",   type=float, default=137500,
                   help="Zoom 2 window end (ms, default: 137500)")
    return p.parse_args()


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n  Loading pickle: {args.pkl_file}")
    with open(args.pkl_file, "rb") as f:
        data = pickle.load(f)

    label   = data["label"]
    h5_file = data["h5_file"]
    params  = data["params"]

    pkl_dir = os.path.dirname(os.path.abspath(args.pkl_file))
    h5_file = os.path.join(pkl_dir, os.path.basename(h5_file))


    base = label
    print(f"  Simulation: {label}")
    print(f"  Parameters used in precompute: {params}")

    t_start_ms = params.get("t_start")
    t_end_ms   = params.get("t_end")

    # ── Voltage traces ────────────────────────────────────────────────────────
    print("\n  Full voltage plot...")
    # load I_ext from the voltage file to pass it to the plots
    _I_ext_plot = None
    try:
        _vp = str(h5_file)
        if _vp.endswith('.h5') or _vp.endswith('.hdf5'):
            import h5py as _h5p
            with _h5p.File(_vp, 'r') as _hf:
                if 'I_ext_ABPD' in _hf:
                    _I_ext_plot = np.stack([_hf['I_ext_ABPD'][:],
                                            _hf['I_ext_LP'][:],
                                            _hf['I_ext_PY'][:]])
        else:
            _dv = np.load(_vp, allow_pickle=True)
            if 'I_ext_ABPD' in _dv:
                _I_ext_plot = np.stack([_dv['I_ext_ABPD'],
                                        _dv['I_ext_LP'],
                                        _dv['I_ext_PY']])
    except Exception as _e:
        print(f"  [warn] Could not load I_ext for plotting: {_e}")

    plot_voltage_traces(h5_file, label=label,
                        out=f"{base}_voltage_traces.svg",
                        decimate=args.plot_decimate,
                        I_ext=_I_ext_plot,
                        t_start_ms=t_start_ms, t_end_ms=t_end_ms,
                        line_width=0.7, figsize=(20, 8))

    print("  Voltage plot zoom1...")
    plot_voltage_traces(h5_file, label=label,
                        out=f"{base}_voltage_traces_zoom1.svg",
                        decimate=args.plot_decimate,
                        I_ext=_I_ext_plot,
                        t_start_ms=args.zoom1_start, t_end_ms=args.zoom1_end,
                        line_width=1, figsize=(10, 8))

    print("  Voltage plot zoom2...")
    plot_voltage_traces(h5_file, label=label,
                        out=f"{base}_voltage_traces_zoom2.svg",
                        decimate=args.plot_decimate,
                        I_ext=_I_ext_plot,
                        t_start_ms=args.zoom2_start, t_end_ms=args.zoom2_end,
                        line_width=1, figsize=(10, 8))

    print("\n  Done.")


if __name__ == "__main__":
    main()
