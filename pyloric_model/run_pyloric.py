"""Copyright (c) 2026 Irene Elices. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

import numpy as np
import pyximport
pyximport.install(
    setup_args={'include_dirs': np.get_include()},
    reload_support=True,
    language_level=3,
)
# ───────────────────────────────────────────────────────────────────────────

import pandas as pd
import matplotlib.pyplot as plt

from pyloric import create_prior, simulate, summary_stats
from pyloric.utils import show_traces
from pyloric_iext import make_I_ext, combine_I_ext, simulate_chunked

# ── Pandas display options ─────────────────────────
pd.set_option("display.max_rows", 10)
pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 2000)
pd.set_option("display.float_format", "{:,.4f}".format)

# ── Parameters ──────────────────────────────────────────────────────────────
dt    = 0.025
t_max = 150000 # 300000 ms

# Tuned parameters from Deistler et al. 2022 (PNAS)
# Path to the .pkl file (relative or absolute)
PARAMS_FILE = "close_to_xo_circuit_parameters_min_burst_condition_078.pkl"
# Alternative (no minimum-burst condition):
# PARAMS_FILE = "close_to_xo_circuit_parameters_078.pkl"

p = pd.read_pickle(PARAMS_FILE)
print(f"Loaded {len(p)} tuned parameter sets from the paper")

# How many circuits to simulate (pick a small number to start)
N_CIRCUITS = 1
p = p.iloc[:N_CIRCUITS]

# ── Define the external current here ───────────────────────────────────────
# Pick ONE block and comment out the rest, or combine them with combine_I_ext()

# --- No current ---
#I = None
#I_label = 'noIext'

# --- DC pulse on AB/PD ---
# I = make_I_ext(t_max, dt, neuron='AB/PD', kind='step',
#                amp=0.5, t_start=3000, t_end=70000)
# I_label = 'step_ABPD_amp0.5_t3000-70000'

# --- Triangular ramp on AB/PD (rises then falls back to 0) ---
#I = make_I_ext(t_max, dt, neuron='AB/PD', kind='triangle',
#               amp=0.00012, t_start=1000, t_peak=45500, t_end=91000)
#I_label = 'triangle_ABPD_amp0.00012_t1000-45500-91000'

# --- Sinusoidal on LP ---
#I = make_I_ext(t_max, dt, neuron='AB/PD', kind='sinusoidal',
#                 amp=0.00006, freq=0.3, t_start=0, t_end=500000, offset=0.00005)
#I_label = 'sin_ABPD_amp0.00006_freq0.3Hz'

# --- Ramp on AB/PD: 0 -> 80 pA over 130 s ---
I = make_I_ext(t_max, dt, neuron='AB/PD', kind='ramp',
               v_start=0.00010, v_end=-0.000015, t_start=10000, t_end=140000)
I_label = 'ramp_ABPD_amp0.00010-desc-15_dur140s'

# --- Pulse train on PY ---
# I = make_I_ext(t_max, dt, neuron='PY', kind='pulse_train',
#                amp=2.0, t_start=2000, t_end=9000, period=500, width=50)
# I_label = 'pulsetrain_PY_amp2.0_p500_w50'

# --- Noise on AB/PD ---
# I = make_I_ext(t_max, dt, neuron='AB/PD', kind='noise',
#                amp=1.0, std=0.5, t_start=2000, t_end=9000, seed=42)
# I_label = 'noise_ABPD_amp1.0_std0.5'

# --- Multiple neurons at once ---
# I_abpd = make_I_ext(t_max, dt, neuron='AB/PD', kind='step',
#                     amp=2.0, t_start=2000, t_end=5000)
# I_lp   = make_I_ext(t_max, dt, neuron='LP', kind='sinusoidal',
#                     amp=1.0, freq=2.0, t_start=4000, t_end=8000)
# I = combine_I_ext(I_abpd, I_lp)
# I_label = 'step_ABPD_sin_LP'

# ── Simulate in chunks (avoids OOM for long t_max) ─────────────────────────
# chunk_ms: duration of each chunk in ms. Adjust based on available RAM:
#   10000 ms -> ~80 MB/chunk    (safe on any machine)
#   50000 ms -> ~400 MB/chunk
CHUNK_MS = 100000

for i in range(len(p)):
    simulate_chunked(
        params       = p.iloc[i],
        I_ext        = I,
        dt           = dt,
        t_max        = t_max,
        chunk_ms     = CHUNK_MS,
        seed         = 0,
        save_path    = f'simulation_circuit{i}_{I_label}',  # output: _voltages.h5 and _currents.h5
        save_currents= True,
        verbose      = True,
    )

# ── Plot (first 10000 ms only, as a quick diagnostic) ──────────────────────
names  = ['AB/PD', 'LP', 'PY']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
PLOT_MS = 10000   # ms to display — adjust to see more

for i in range(len(p)):
    import h5py
    with h5py.File(f'simulation_circuit{i}_{I_label}_voltages.h5', 'r') as hf:
        t_vec   = hf['t'][:]
        voltage = np.stack([hf['V_ABPD'][:], hf['V_LP'][:], hf['V_PY'][:]])

    # Crop for plotting
    mask    = t_vec <= PLOT_MS
    t_plot  = t_vec[mask]
    v_plot  = voltage[:, mask]

    has_Iext = (I is not None) and np.any(I != 0)
    n_panels = 4 if has_Iext else 3

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.5 * n_panels),
                             sharex=True, gridspec_kw={'hspace': 0.08})

    for j in range(3):
        axes[j].plot(t_plot, v_plot[j], lw=0.6, color=colors[j])
        axes[j].set_ylabel(f'{names[j]}\n(mV)', fontsize=9)
        axes[j].spines['top'].set_visible(False)
        axes[j].spines['right'].set_visible(False)

    if has_Iext:
        ax = axes[3]
        for j in range(3):
            if np.any(I[j, mask] != 0):
                ax.plot(t_plot, I[j, mask], lw=0.8, color=colors[j], label=names[j])
        ax.set_ylabel('I_ext\n(µA)', fontsize=9)
        ax.legend(loc='upper right', fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (ms)', fontsize=10)
    fig.suptitle(f'Circuit {i} — {I_label} (first {PLOT_MS} ms)', fontsize=10)
    plt.tight_layout()

    fname_fig = f'simulation_circuit{i}_{I_label}.png'
    plt.savefig(fname_fig, dpi=150, bbox_inches='tight')
    print(f'Figure saved: {fname_fig}')
    plt.close(fig)   # avoid blocking in headless environments