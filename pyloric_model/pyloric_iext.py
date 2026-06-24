"""Copyright (c) 2026 Irene Elices. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

"""
pyloric_iext.py
===============
Injects external currents into the pyloric simulator (mackelab).

The simulator already supports Ix (external current), but the public
`simulate()` API leaves it at zero. Here we wrap `sim_time` directly to
pass an arbitrary waveform to any neuron.
"""

import gc
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# Neuron indices
# ─────────────────────────────────────────────────────────────────────────────
NEURONS = {'AB/PD': 0, 'ABPD': 0, 'abpd': 0,
           'LP': 1, 'lp': 1,
           'PY': 2, 'py': 2}

N_NEURONS = 3


# ─────────────────────────────────────────────────────────────────────────────
# Waveform generators
# ─────────────────────────────────────────────────────────────────────────────

def make_I_ext(
    T: float,
    dt: float,
    neuron: int | str = 0,
    kind: str = 'step',
    amp: float = 1.0,
    **kwargs,
) -> np.ndarray:
    """
    Generates an external-current array of shape (3, N_timesteps).
    """
    # Resolve neuron index
    if isinstance(neuron, str):
        neuron = NEURONS[neuron]

    N = int(T / dt)
    t = np.arange(N) * dt

    I_ext = np.zeros((N_NEURONS, N))
    wave  = np.zeros(N)

    if kind == 'step':
        t_start = kwargs.get('t_start', 0.0)
        t_end   = kwargs.get('t_end',   T)
        mask = (t >= t_start) & (t < t_end)
        wave[mask] = amp

    elif kind == 'ramp':
        t_start = kwargs.get('t_start', 0.0)
        t_end   = kwargs.get('t_end',   T / 2)
        t_off   = kwargs.get('t_off',   T)
        v_start = kwargs.get('v_start', None)
        v_end   = kwargs.get('v_end',   None)
        # if v_start/v_end given, ignore amp
        if v_start is not None or v_end is not None:
            v0 = v_start if v_start is not None else 0.0
            v1 = v_end   if v_end   is not None else amp
        else:
            v0, v1 = 0.0, amp
        # ramp phase
        mask_up = (t >= t_start) & (t < t_end)
        wave[mask_up] = v0 + (v1 - v0) * (t[mask_up] - t_start) / (t_end - t_start)
        # plateau at final value
        mask_on = (t >= t_end) & (t < t_off)
        wave[mask_on] = v1
        # after t_off -> 0

    elif kind == 'ramp_down':
        t_start = kwargs.get('t_start', 0.0)
        t_end   = kwargs.get('t_end',   T)
        mask = (t >= t_start) & (t < t_end)
        wave[mask] = amp * (1 - (t[mask] - t_start) / (t_end - t_start))

    elif kind == 'triangle':
        # Ramps 0 -> amp between t_start/t_peak, then amp -> 0 between t_peak/t_end
        t_start = kwargs.get('t_start', 0.0)
        t_peak  = kwargs.get('t_peak',  T / 2)
        t_end   = kwargs.get('t_end',   T)
        mask_up   = (t >= t_start) & (t < t_peak)
        mask_down = (t >= t_peak)  & (t < t_end)
        wave[mask_up]   = amp * (t[mask_up]   - t_start) / (t_peak - t_start)
        wave[mask_down] = amp * (1 - (t[mask_down] - t_peak) / (t_end - t_peak))

    elif kind == 'sinusoidal':
        t_start = kwargs.get('t_start', 0.0)
        t_end   = kwargs.get('t_end',   T)
        freq    = kwargs.get('freq',    1.0)   # Hz
        offset  = kwargs.get('offset',  0.0)
        mask = (t >= t_start) & (t < t_end)
        # freq in Hz, t in ms -> convert to seconds
        wave[mask] = offset + amp * np.sin(2 * np.pi * freq * t[mask] / 1000.0)

    elif kind == 'noise':
        t_start = kwargs.get('t_start', 0.0)
        t_end   = kwargs.get('t_end',   T)
        std     = kwargs.get('std',     0.5)
        seed    = kwargs.get('seed',    None)
        rng  = np.random.default_rng(seed)
        mask = (t >= t_start) & (t < t_end)
        wave[mask] = amp + rng.normal(0, std, size=mask.sum())

    elif kind == 'pulse_train':
        t_start = kwargs.get('t_start', 0.0)
        t_end   = kwargs.get('t_end',   T)
        period  = kwargs.get('period',  500.0)   # ms
        width   = kwargs.get('width',   50.0)    # ms
        mask_region = (t >= t_start) & (t < t_end)
        t_rel = t - t_start
        mask_pulse = mask_region & ((t_rel % period) < width)
        wave[mask_pulse] = amp

    elif kind == 'custom':
        waveform = kwargs.get('waveform', None)
        if waveform is None:
            raise ValueError("kind='custom' requires waveform=<array>")
        waveform = np.asarray(waveform, dtype=float)
        if len(waveform) != N:
            raise ValueError(
                f"waveform has {len(waveform)} points, expected {N} "
                f"(T={T} ms, dt={dt} ms)"
            )
        wave = waveform

    else:
        raise ValueError(
            f"Unknown kind='{kind}'. Options: "
            "'step', 'ramp', 'ramp_down', 'triangle', 'sinusoidal', 'noise', 'pulse_train', 'custom'"
        )

    I_ext[neuron] = wave
    return I_ext


def combine_I_ext(*arrays: np.ndarray) -> np.ndarray:
    """
    Sums several I_ext arrays (useful for injecting into multiple neurons
    at once, or combining several waveforms on the same neuron).

    Example
    -------
    I_abpd = make_I_ext(T, dt, neuron=0, kind='step', amp=2.0, t_start=2000, t_end=5000)
    I_lp   = make_I_ext(T, dt, neuron=1, kind='sinusoidal', amp=1.0, freq=2.0)
    I_total = combine_I_ext(I_abpd, I_lp)
    """
    result = np.zeros_like(arrays[0])
    for a in arrays:
        result = result + a
    return result


# ─────────────────────────────────────────────────────────────────────────────
# simulate wrapper
# ─────────────────────────────────────────────────────────────────────────────

def simulate_with_Iext(
    params,
    I_ext: np.ndarray,
    dt: float = 0.025,
    t_max: float = 11000.0,
    temperature: float = 283.0,
    seed: int = None,
    **simulate_kwargs,
) -> dict:
    """
    Runs the pyloric simulation with an arbitrary external current.

    Calls `pyloric.simulate` (patched interface.py) internally, passing
    I_ext through to the Cython solver.

    Parameters
    ----------
    params      : pd.Series with circuit parameters (output of prior.sample())
    I_ext       : array (3, N_timesteps), current in µA
    dt          : timestep (ms). Default 0.025 ms
    t_max       : total duration (ms). Default 11000 ms (same as interface.py)
    temperature : temperature in Kelvin. Default 283 K (~10°C)
    seed        : optional random seed
    **simulate_kwargs : extra arguments passed to pyloric.simulate

    Returns
    -------
    dict with keys: 'voltage' (3, N), 'dt', 't_max'
        and optionally 'energy', 'membrane_conds', etc.
    """
    try:
        from pyloric import simulate
    except ImportError:
        raise ImportError(
            "pyloric package not found. "
            "Install with: pip install -e .  (from the pyloric-main folder)"
        )

    N = int(t_max / dt)

    # Check dimensions
    if I_ext.shape != (N_NEURONS, N):
        raise ValueError(
            f"I_ext has shape {I_ext.shape}, expected ({N_NEURONS}, {N}). "
            f"Make sure make_I_ext(T=t_max, dt=dt, ...) used the same values."
        )

    out  = simulate(
        params,
        dt=dt,
        t_max=t_max,
        temperature=temperature,
        seed=seed,
        I_ext=I_ext,
        **simulate_kwargs,
    )

    return out


def _extract_final_state(data: dict, n_neurons: int = 3, n_synapses: int = 7) -> list:
    """
    Extracts the final state of a simulation to use as init for the next chunk.

    sim_time expects init in this format:
        init[j] = [V, Ca, mNa, mCaT, mCaS, mA, mKCa, mKd, mH, 0, hNa, hCaT, hCaS, hA]
        init[n_neurons] = [sx_0, sx_1, ..., sx_6]   (synaptic states)

    Parameters
    ----------
    data : dict returned by sim_time (with keys 'Vs', 'Cas', 'logs', 'n_Kd')
    """
    logs = data['logs']
    Vs   = data['Vs']    # (3, N)
    Cas  = data['Cas']   # (3, N)

    state = []
    for j in range(n_neurons):
        neuron_state = [
            Vs[j, -1],               # 0  V
            Cas[j, -1],              # 1  Ca
            logs['mNa'][j, -1],      # 2  mNa
            logs['mCaT'][j, -1],     # 3  mCaT
            logs['mCaS'][j, -1],     # 4  mCaS
            logs['mA'][j, -1],       # 5  mA
            logs['mKCa'][j, -1],     # 6  mKCa
            logs['mKd'][j, -1],      # 7  mKd
            logs['mH'][j, -1],       # 8  mH
            0.0,                     # 9  (unused in sim_time)
            logs['hNa'][j, -1],      # 10 hNa
            logs['hCaT'][j, -1],     # 11 hCaT
            logs['hCaS'][j, -1],     # 12 hCaS
            logs['hA'][j, -1],       # 13 hA
        ]
        state.append(neuron_state)

    # Final synaptic state (sx has shape (7, N))
    sx_final = [logs['s'][k, -1] for k in range(n_synapses)]
    state.append(sx_final)

    return state


def simulate_chunked(
    params,
    I_ext: np.ndarray | None = None,
    dt: float = 0.025,
    t_max: float = 11000.0,
    chunk_ms: float = 10000.0,
    temperature: float = 283.0,
    seed: int = None,
    save_path: str = 'simulation_chunked',
    save_currents: bool = True,
    verbose: bool = True,
    **simulate_kwargs,
) -> None:
    """
    Simulates the pyloric circuit in chunks to avoid the OOM killer,
    saving voltages and currents to disk at the end of each chunk.

    Returns nothing — data is written directly to .h5 files:
        {save_path}_voltages.h5   — voltages for all neurons
        {save_path}_currents.h5   — ionic and synaptic currents
                                     (only if save_currents=True)

    Voltage file contents: t, V_ABPD, V_LP, V_PY, dt, t_max
    Currents file uses the same format as pyloric_currents.py's save_currents().

    Parameters
    ----------
    params      : pd.Series with circuit parameters
    I_ext       : array (3, N_total) of external current, or None
    dt          : timestep (ms)
    t_max       : total simulation duration (ms)
    chunk_ms    : duration of each chunk (ms). Default 10000 ms.
                  Adjust based on available RAM:
                    10000 ms  -> ~80 MB of membrane_conds per chunk
                    50000 ms  -> ~400 MB
    temperature : temperature in Kelvin
    seed        : random seed for noise (first chunk only)
    save_path   : prefix for output files (no extension)
    save_currents: if True, also computes and saves ionic and synaptic
                  currents. If False, only saves voltages (faster).
    verbose     : print progress
    **simulate_kwargs : extra arguments passed to pyloric.simulate
    """
    try:
        from pyloric import simulate
        from pyloric.simulator import sim_time
        from pyloric.utils import (
            build_conns, build_synapse_q10s, ensure_array_not_scalar,
            membrane_conductances_replaced_with_defaults,
            q10s_replaced_with_defaults, synapses_replaced_with_defaults,
        )
    except ImportError:
        raise ImportError("pyloric package not found.")

    if save_currents:
        try:
            from pyloric_currents import extract_currents
            from pyloric_currents import (
                DEFAULT_CONN_LABELS, DEFAULT_CONN_POST, DEFAULT_E_SYN,
                CHANNEL_NAMES, NEURON_NAMES,
            )
        except ImportError:
            raise ImportError("pyloric_currents.py not found.")

    N_total    = int(t_max / dt)
    N_chunk    = int(chunk_ms / dt)
    n_chunks   = int(np.ceil(N_total / N_chunk))

    if verbose:
        print(f"[simulate_chunked] t_max={t_max} ms, chunk={chunk_ms} ms, "
              f"n_chunks={n_chunks}, N_total={N_total}")

    # ── Prepare model parameters (same as interface.py) ───────────────────────
    defaults_dict = {
        "membrane_gbar": [["PM", "PM_4", 0.628e-3],
                          ["LP", "LP_3", 0.628e-3],
                          ["PY", "PY_4", 0.628e-3]],
        "Q10_gbar_mem"  : [1.5]*8,
        "Q10_gbar_syn"  : [1.5, 1.5],
        "Q10_tau_m"     : [2.4],
        "Q10_tau_h"     : [2.8],
        "Q10_tau_CaBuff": [2.0],
        "Q10_tau_syn"   : [1.7, 1.7],
    }
    for key in defaults_dict:
        defaults_dict[key] = ensure_array_not_scalar(defaults_dict[key])

    setup_dict = {
        "membrane_gbar"  : [[True]*8, [True]*8, [True]*8],
        "Q10_gbar_mem"   : [False]*8,
        "Q10_gbar_syn"   : [False, False],
        "Q10_tau_m"      : [False],
        "Q10_tau_h"      : [False],
        "Q10_tau_CaBuff" : [False],
        "Q10_tau_syn"    : [False, False],
    }
    for key in setup_dict:
        setup_dict[key] = ensure_array_not_scalar(setup_dict[key])

    membrane_pd  = membrane_conductances_replaced_with_defaults(params, defaults_dict)
    synaptic_pd  = synapses_replaced_with_defaults(params, defaults_dict)
    q10_pd       = q10s_replaced_with_defaults(params, defaults_dict)

    membrane_q10_gbar  = q10_pd["Q10 gbar"].to_numpy()[0, :8]
    synapse_q10_gbar   = build_synapse_q10s(q10_pd["Q10 gbar"].to_numpy()[0, 8:10])
    synapse_q10_tau    = build_synapse_q10s(q10_pd["Q10 tau"].to_numpy()[0, 3:5])
    q10_tau_m          = q10_pd["Q10 tau"]["m"].to_numpy().tolist() * 7
    q10_tau_h          = q10_pd["Q10 tau"]["h"].to_numpy().tolist() * 4
    q10_tau_cabuff     = q10_pd["Q10 tau"]["CaBuff"].to_numpy().tolist()
    q10_tau_m[2]       = 2.0   # CaS  (Caplan 2014)
    q10_tau_m[4]       = 1.6   # KCa  (Caplan 2014)

    modelx = np.reshape(membrane_pd.to_numpy(), (3, 8))
    conns  = build_conns(-np.exp(synaptic_pd.to_numpy()[0]))

    # Noise: generate all at once for reproducibility
    rng   = np.random.RandomState(seed=seed)
    noise = rng.normal(scale=simulate_kwargs.pop('noise_std', 0.001),
                       size=(3, N_total))

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    # ── Create HDF5 files with pre-allocated datasets ──────────────────────────
    # Write chunk by chunk without accumulating anything in RAM
    path_v = Path(f'{save_path}_voltages.h5')
    path_c = Path(f'{save_path}_currents.h5') if save_currents else None

    with h5py.File(path_v, 'w') as fv:
        fv.create_dataset('t',     shape=(N_total,),    dtype='f4')
        fv.create_dataset('V_ABPD',shape=(N_total,),    dtype='f4')
        fv.create_dataset('V_LP',  shape=(N_total,),    dtype='f4')
        fv.create_dataset('V_PY',  shape=(N_total,),    dtype='f4')
        fv.attrs['dt']    = dt
        fv.attrs['t_max'] = t_max
        if I_ext is not None:
            fv.create_dataset('I_ext_ABPD', data=I_ext[0].astype('f4'))
            fv.create_dataset('I_ext_LP',   data=I_ext[1].astype('f4'))
            fv.create_dataset('I_ext_PY',   data=I_ext[2].astype('f4'))

    if save_currents:
        # chunk_size for compression: ~1 MB per chunk on disk
        _cz_1d = min(N_total, 131072)                          # 128k timesteps
        _cz    = {'compression': 'gzip', 'compression_opts': 4,
                  'shuffle': True}
        with h5py.File(path_c, 'w') as fc:
            fc.attrs['dt']    = dt
            fc.attrs['t_max'] = t_max
            # t: stored for reference only; Vs is read from the voltages file
            fc.create_dataset('t',
                              shape=(N_total,), dtype='f4',
                              chunks=(_cz_1d,), **_cz)
            # I_ionic: (3 neurons, 8 channels, N)
            fc.create_dataset('I_ionic',
                              shape=(3, 8, N_total), dtype='f4',
                              chunks=(1, 1, _cz_1d), **_cz)
            # I_synaptic: (7 synapses, N)  — I_syn_total is recomputed during analysis
            fc.create_dataset('I_synaptic',
                              shape=(7, N_total), dtype='f4',
                              chunks=(1, _cz_1d), **_cz)
            # Metadata (small, no compression)
            fc.create_dataset('channel_names',  data=np.array(CHANNEL_NAMES,  dtype='S'))
            fc.create_dataset('neuron_names',   data=np.array(NEURON_NAMES,   dtype='S'))
            fc.create_dataset('conn_labels',    data=np.array(DEFAULT_CONN_LABELS, dtype='S'))
            fc.create_dataset('conn_post',      data=np.array(DEFAULT_CONN_POST))
            fc.create_dataset('e_syn_per_conn', data=np.array(DEFAULT_E_SYN))

    init = None   # first chunk starts from default conditions

    for ch in range(n_chunks):
        i0 = ch * N_chunk
        i1 = min(i0 + N_chunk, N_total)
        N_ch = i1 - i0
        t_ch = np.arange(N_ch) * dt

        if verbose:
            print(f"  Chunk {ch+1}/{n_chunks}  "
                  f"[{i0*dt/1000:.1f}–{i1*dt/1000:.1f} s] ...", end=' ', flush=True)

        I_ch = noise[:, i0:i1].copy()
        if I_ext is not None:
            I_ch = I_ch + I_ext[:, i0:i1]

        data = sim_time(
            dt,
            t_ch,
            I_ch,
            modelx,
            conns_=conns,
            g_q10_conns_gbar=synapse_q10_gbar,
            g_q10_conns_tau=synapse_q10_tau,
            g_q10_memb_gbar=membrane_q10_gbar,
            g_q10_memb_tau_m=q10_tau_m,
            g_q10_memb_tau_h=q10_tau_h,
            g_q10_memb_tau_CaBuff=q10_tau_cabuff,
            temp=temperature,
            num_energy_timesteps=0,
            num_energyscape_timesteps=N_ch if save_currents else 0,
            init=init,
            start_val_input=0.0,
            verbose=False,
        )

        # ── Write voltages directly to HDF5 ────────────────────────────────────
        t_chunk = np.arange(i0, i1, dtype='f4') * dt
        with h5py.File(path_v, 'a') as fv:
            fv['t'][i0:i1]      = t_chunk
            fv['V_ABPD'][i0:i1] = data['Vs'][0].astype('f4')
            fv['V_LP'][i0:i1]   = data['Vs'][1].astype('f4')
            fv['V_PY'][i0:i1]   = data['Vs'][2].astype('f4')

        # ── Compute and write currents directly to HDF5 ────────────────────────
        if save_currents:
            out_ch = {
                'voltage'         : data['Vs'],
                'membrane_conds'  : data['membrane_conds'],
                'synaptic_conds'  : data['synaptic_conds'],
                'reversal_calcium': data['reversal_calcium'],
                'dt'              : dt,
                't_max'           : N_ch * dt,
            }
            curr_ch = extract_currents(out_ch, dt=dt)
            with h5py.File(path_c, 'a') as fc:
                fc['t'][i0:i1]             = t_chunk
                fc['I_ionic'][:, :, i0:i1] = curr_ch['I_ionic'].astype('f4')
                fc['I_synaptic'][:, i0:i1] = curr_ch['I_synaptic'].astype('f4')

        init = _extract_final_state(data)

        # Explicitly free memory
        del data, I_ch, t_chunk
        if save_currents:
            del out_ch, curr_ch
        gc.collect()

        if verbose:
            print('OK')

    if verbose:
        print(f'  Voltages → {path_v}')
        if save_currents:
            print(f'  Currents → {path_c}')
        print('[simulate_chunked] Done.')

    if verbose:
        print('[simulate_chunked] Done.')




def plot_simulation(out: dict, I_ext: np.ndarray = None, dt: float = 0.025,
                    title: str = '', save_path: str = None):
    """
    Standard plot of the 3 neurons plus the injected current (if given).

    Parameters
    ----------
    out       : output dict from simulate / simulate_with_Iext
    I_ext     : array shape (3, N) — if given, drawn as the bottom panel
    dt        : timestep (ms)
    title     : plot title
    save_path : path to save the figure (None = display only)
    """
    # Compatible with both versions of the package
    Vs = out.get('voltage', out.get('Vs'))
    if Vs is None:
        raise KeyError("Output dict has neither 'voltage' nor 'Vs'.")

    N  = Vs.shape[1]
    t  = np.arange(N) * dt

    names  = ['AB/PD', 'LP', 'PY']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    n_panels = 3 + (1 if I_ext is not None else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 2.5 * n_panels),
                              sharex=True, gridspec_kw={'hspace': 0.08})

    for j in range(3):
        axes[j].plot(t, Vs[j], lw=0.6, color=colors[j])
        axes[j].set_ylabel(f'{names[j]}\n(mV)', fontsize=9)
        axes[j].spines['top'].set_visible(False)
        axes[j].spines['right'].set_visible(False)

    if I_ext is not None:
        ax = axes[3]
        for j in range(3):
            if np.any(I_ext[j] != 0):
                ax.plot(t, I_ext[j], lw=0.8, color=colors[j],
                        label=names[j])
        ax.set_ylabel('I_ext\n(µA)', fontsize=9)
        ax.legend(loc='upper right', fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (ms)', fontsize=10)
    if title:
        fig.suptitle(title, fontsize=11, y=1.01)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved to: {save_path}')

    plt.show()
    return fig, axes