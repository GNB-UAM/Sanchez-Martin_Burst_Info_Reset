"""Copyright (c) 2026 Irene Elices. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

"""
pyloric_data.py
=====================
Loads the voltage files (.h5 or .npz), detects bursts, computes
per-cycle intervals, and serializes the results into a pickle file
for later use by pyloric_plots.py.

Usage
---
    python pyloric_data.py                          # auto-detects simulation_circuit*.h5
    python pyloric_data.py simulation_circuit0_noIext.h5
    python pyloric_data.py --skip 5
    python pyloric_data.py --min-isi 30
    python pyloric_data.py --t-start 10000 --t-end 100000
    python pyloric_data.py --out data.pkl

Output
------
    {base}_data.pkl  with the keys:
        intervals     : dict of 1D arrays (Periodo, LP_burst, ABPD_burst,
                        PY_duration, ABPD_LP_delay, LP_ABPD_delay, t_cycle)
        valid_cycles  : list of dicts {lp, abpd, py, lp_next_on}
        label         : str, base name of the file
        h5_file      : str, path to the voltage file
        curr_h5      : str or None, path to the currents file
        params        : dict with the parameters used (thr_up, thr_down,
                        min_isi, skip, t_start, t_end)
"""

import sys
import glob
import pickle
import argparse
import numpy as np
import os
import h5py

# ─── default configuration ────────────────────────────────────────────────────
THR_UP   =  -10.0
THR_DOWN =  -55.0
MIN_ISI  =   30.0

NEURON_NAMES = ["AB/PD", "LP", "PY"]


AREA_CM2 = 0.628e-3


# ─── helpers (reused from pyloric_analysis.py) ───────────────────────────────

def find_file(pattern, explicit=None):
    if explicit:
        return explicit
    for pat in [pattern.replace('.npz', '.h5'), pattern]:
        matches = sorted(glob.glob(pat))
        if matches:
            return matches[-1]
    return None


def load_h5(path, t_start_ms=None, t_end_ms=None):
    
    path = str(path)

    if path.endswith('.npz'):
        data  = np.load(path)
        t_ms  = data['t']
        V     = np.stack([data['V_ABPD'], data['V_LP'], data['V_PY']])
        dt    = float(data['dt'])
        t_max = float(data['t_max'])
    elif path.endswith('.h5') or path.endswith('.hdf5'):
        
        with h5py.File(path, 'r') as hf:
            t_ms  = hf['t'][:]
            V     = np.stack([hf['V_ABPD'][:], hf['V_LP'][:], hf['V_PY'][:]])
            dt    = float(hf.attrs['dt'])
            t_max = float(hf.attrs['t_max'])
    else:
        raise ValueError(f"Unrecognized format: {path}. Use .npz or .h5")

    I_ext = None
    if path.endswith('.npz') and 'I_ext_ABPD' in data:
        I_ext = np.stack([data['I_ext_ABPD'], data['I_ext_LP'], data['I_ext_PY']])
    elif (path.endswith('.h5') or path.endswith('.hdf5')):
        
        with h5py.File(path, 'r') as hf:
            if 'I_ext_ABPD' in hf:
                I_ext = np.stack([hf['I_ext_ABPD'][:], hf['I_ext_LP'][:], hf['I_ext_PY'][:]])

    label = os.path.basename(path)
    for suffix in ('_voltages.h5', '_voltages.npz', '.h5', '.npz'):
        if label.endswith(suffix):
            label = label[:-len(suffix)]
            break

    if t_start_ms is not None or t_end_ms is not None:
        t0   = t_start_ms if t_start_ms is not None else t_ms[0]
        t1   = t_end_ms   if t_end_ms   is not None else t_ms[-1]
        mask = (t_ms >= t0) & (t_ms <= t1)
        t_ms = t_ms[mask]
        V    = V[:, mask]
        if I_ext is not None:
            I_ext = I_ext[:, mask]

    t_s = t_ms / 1000.0
    return t_s, V, I_ext, dt, label


def detect_bursts(t, v, thr_up=None, thr_down=None, min_isi_s=None):
    if thr_up    is None: thr_up    = THR_UP
    if thr_down  is None: thr_down  = THR_DOWN
    if min_isi_s is None: min_isi_s = 0.0

    bursts          = []
    in_spike        = False
    in_burst        = False
    t_first         = None
    t_last          = None
    t_silence_start = None

    for i in range(len(v)):
        vi = v[i]
        ti = t[i]

        if not in_spike and vi >= thr_up:
            in_spike = True
            if not in_burst:
                in_burst = True
                t_first  = ti
            t_last          = ti
            t_silence_start = None
        elif in_spike and vi < thr_up:
            in_spike = False

        if in_burst and not in_spike and vi <= thr_down:
            if t_silence_start is None:
                t_silence_start = ti
            elif (ti - t_silence_start) >= min_isi_s:
                bursts.append((t_first, t_last))
                in_burst        = False
                t_first         = None
                t_last          = None
                t_silence_start = None
        elif in_burst and vi > thr_down:
            t_silence_start = None

    if in_burst and t_last is not None:
        bursts.append((t_first, t_last))

    return bursts


def compute_intervals(t, V, skip=0, min_isi_s=None):
    bursts_abpd = detect_bursts(t, V[0], min_isi_s=min_isi_s)
    bursts_lp   = detect_bursts(t, V[1], min_isi_s=min_isi_s)
    bursts_py   = detect_bursts(t, V[2], min_isi_s=min_isi_s)

    bursts_abpd = bursts_abpd[skip:]
    bursts_lp   = bursts_lp[skip:]
    bursts_py   = bursts_py[skip:]

    abpd_ons = np.array([b[0] for b in bursts_abpd])
    py_ons   = np.array([b[0] for b in bursts_py])

    def first_burst_in(burst_list, burst_ons, t_lo, t_hi):
        i0 = np.searchsorted(burst_ons, t_lo, side='left')
        i1 = np.searchsorted(burst_ons, t_hi, side='left')
        if i1 <= i0:
            return None
        return burst_list[i0]

    keys    = ["t_cycle", "Periodo", "LP_burst", "ABPD_burst",
               "PY_duration", "ABPD_LP_delay", "LP_ABPD_delay"]
    results     = {k: [] for k in keys}
    cycles_data = []

    for i in range(1, len(bursts_lp) - 1):
        lp_on,      lp_off      = bursts_lp[i]
        lp_on_prev, lp_off_prev = bursts_lp[i - 1]
        lp_on_next, _           = bursts_lp[i + 1]
        periodo                 = lp_on_next - lp_on

        t_lo = lp_on
        t_hi = lp_on_next

        ba = first_burst_in(bursts_abpd, abpd_ons, t_lo, t_hi)
        bp = first_burst_in(bursts_py,   py_ons,   t_lo, t_hi)

        if ba is None or bp is None:
            cycles_data.append({
                "valid":      False,
                "lp":         (lp_on, lp_off),
                "lp_next_on": lp_on_next,
                "reason":     f"{'ABPD' if ba is None else 'PY'} not found",
            })
            continue

        abpd_on, abpd_off = ba
        py_on,   py_off   = bp

        lp_burst      = lp_off   - lp_on
        abpd_burst    = abpd_off - abpd_on
        py_duration   = py_off   - py_on
        abpd_lp_delay = lp_on_next - abpd_off
        lp_abpd_delay = abpd_on - lp_off

        if lp_burst <= 0 or abpd_burst <= 0 or py_duration <= 0:
            cycles_data.append({
                "valid":      False,
                "lp":         (lp_on, lp_off),
                "lp_next_on": lp_on_next,
                "reason":     "duration <= 0",
            })
            continue

        results["t_cycle"].append(lp_on)
        results["Periodo"].append(periodo)
        results["LP_burst"].append(lp_burst)
        results["ABPD_burst"].append(abpd_burst)
        results["PY_duration"].append(py_duration)
        results["ABPD_LP_delay"].append(abpd_lp_delay)
        results["LP_ABPD_delay"].append(lp_abpd_delay)

        cycles_data.append({
            "valid":      True,
            "lp":         (lp_on, lp_off),
            "abpd":       (abpd_on, abpd_off),
            "py":         (py_on, py_off),
            "lp_next_on": lp_on_next,
        })

    return {k: np.array(v) for k, v in results.items()}, cycles_data


def print_interval_summary(intervals):
    print("\n  Temporal intervals (mean ± SD, in seconds):")
    print(f"  {'Variable':<20}  {'n':>5}  {'mean':>9}  {'SD':>9}  {'CV%':>7}")
    print("  " + "-"*57)
    for k, arr in intervals.items():
        if k == "t_cycle":
            continue
        if len(arr) == 0:
            print(f"  {k:<20}  {'—':>5}")
            continue
        m, s = arr.mean(), arr.std()
        cv   = 100 * s / m if m != 0 else float("nan")
        print(f"  {k:<20}  {len(arr):>5}  {m:>9.4f}  {s:>9.4f}  {cv:>6.2f}%")
    print()


# ─── currents file auto-detection ────────────────────────────────────────────

def find_currents(h5_file, explicit=None):
    
    if explicit:
        return explicit
    for ext_v, ext_c in [('_voltages.h5',  '_currents.h5'),
                          ('_voltages.npz', '_currents.npz'),
                          ('.h5',           '_currents.h5'),
                          ('.h5',           '_currents.npz')]:
        candidate = h5_file.replace(ext_v, ext_c)
        if candidate != h5_file and os.path.exists(candidate):
            return candidate
    for pattern in ['*_currents.h5', '*_currents.npz']:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    return None


# ─── argparse ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5_file", nargs="?", default=None,
                   help="voltages .h5/.npz (auto-detected if omitted)")
    p.add_argument("--skip",      type=int,   default=5,
                   help="Initial cycles to discard (default: 5)")
    p.add_argument("--thr-up",    type=float, default=THR_UP)
    p.add_argument("--thr-down",  type=float, default=THR_DOWN)
    p.add_argument("--min-isi",   type=float, default=MIN_ISI,
                   help="Minimum refractory period between bursts (ms, default: 30)")
    p.add_argument("--t-start",   type=float, default=None,
                   help="Start of the analysis window (ms)")
    p.add_argument("--t-end",     type=float, default=None,
                   help="End of the analysis window (ms)")
    p.add_argument("--currents-h5", default=None,
                   help="Path to the currents file (auto-detected if omitted)")
    p.add_argument("--out",       default=None,
                   help="Path to the output pickle (default: {base}_data.pkl)")
    return p.parse_args()


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    global THR_UP, THR_DOWN, MIN_ISI
    THR_UP   = args.thr_up
    THR_DOWN = args.thr_down
    MIN_ISI  = args.min_isi
    min_isi_s = MIN_ISI / 1000.0

    h5_file = find_file("simulation_circuit*.h5", args.h5_file)
    if not h5_file:
        sys.exit("ERROR: No voltage file found (.h5 or .npz)")

    print(f"\n  Voltage file : {h5_file}")
    t, V, I_ext, dt, label = load_h5(h5_file,
                                       t_start_ms=args.t_start,
                                       t_end_ms=args.t_end)
    print(f"  Points  : {len(t)}")
    print(f"  dt_out  : {t[1]-t[0]:.5f} s   t_max = {t[-1]:.1f} s")
    print(f"  Thresholds: ↑{THR_UP} mV  ↓{THR_DOWN} mV   min_isi={MIN_ISI:.0f} ms")

    # ── Detected bursts (info) ────────────────────────────────────────────────
    print("\n  Bursts detected (before skip):")
    for ni, name in enumerate(NEURON_NAMES):
        b = detect_bursts(t, V[ni], min_isi_s=min_isi_s)
        print(f"    {name}: {len(b)} bursts  "
              f"(mean period = {t[-1]/max(len(b),1)*1000:.1f} ms)")

    # ── Intervals ──────────────────────────────────────────────────────────────
    print(f"\n  Computing intervals (skip={args.skip})...")
    intervals, cycles_data = compute_intervals(t, V, skip=args.skip,
                                               min_isi_s=min_isi_s)
    n_cyc = len(intervals["Periodo"])
    print(f"  Complete cycles found: {n_cyc}")

    if n_cyc == 0:
        print("\n  ERROR: No complete cycles found.")
        print(f"   - Adjust --min-isi (currently {MIN_ISI:.0f} ms)")
        print(f"   - Adjust --thr-up  (currently {THR_UP} mV)")
        print(f"   - Adjust --thr-down (currently {THR_DOWN} mV)")
        print(f"   - Reduce --skip if the simulation is short")
        return

    print_interval_summary(intervals)

    # ── valid_cycles: filtered subset ─────────────────────────────────────────
    t0_s = (args.t_start / 1000.0) if args.t_start is not None else -np.inf
    t1_s = (args.t_end   / 1000.0) if args.t_end   is not None else  np.inf
    valid_cycles = [c for c in cycles_data
                    if c["valid"]
                    and c["lp"][0]      >= t0_s
                    and c["lp_next_on"] <= t1_s]
    print(f"  Valid cycles in analyzed window: {len(valid_cycles)}")

    # ── Currents auto-detection ───────────────────────────────────────────────
    curr_h5 = find_currents(h5_file, args.currents_h5)
    if curr_h5:
        print(f"  Currents file: {curr_h5}")
    else:
        print("  [info] No currents file found.")

    # ── Per-cycle current sums ────────────────────────────────────────────────
    curr_sums = None
    if curr_h5:
        print("  Computing per-cycle current sums...")
        try:
            # import the loading function from the plots module itself
            # or replicate it inline to avoid depending on pyloric_plots
            curr_path = str(curr_h5)
            if curr_path.endswith('.h5') or curr_path.endswith('.hdf5'):
                
                with h5py.File(curr_path, 'r') as hf:
                    I_ionic    = hf['I_ionic'][:]
                    I_synaptic = hf['I_synaptic'][:]
                    t_curr_ms  = hf['t'][:]
                    conn_post  = list(hf['conn_post'][:].astype(int))
                    conn_labels = (list(hf['conn_labels'][:].astype(str))
                                   if 'conn_labels' in hf else None)
            else:
                _d = np.load(curr_path, allow_pickle=False)
                I_ionic    = _d['I_ionic']
                I_synaptic = _d['I_synaptic']
                t_curr_ms  = _d['t']
                conn_post  = list(_d['conn_post'].astype(int))
                conn_labels = (list(_d['conn_labels'].astype(str))
                               if 'conn_labels' in _d else None)

            # I_ionic / I_synaptic are stored as current densities (µA/cm²),
            # since they come from membrane_conds / synaptic_conds, which the
            # simulator normalizes by membrane area. Revert that here to work
            # with absolute µA. I_ext (below, from the voltage file) is left
            # untouched: the simulator never area-normalizes it.
            I_ionic    = I_ionic    * AREA_CM2
            I_synaptic = I_synaptic * AREA_CM2

            t_curr_s  = t_curr_ms / 1000.0
            I_ion_sum = I_ionic.sum(axis=1)

            n_conn = I_synaptic.shape[0]
            syn_by_post = {0: [], 1: [], 2: []}
            for k, post in enumerate(conn_post):
                syn_by_post[post].append(k)
            N = I_synaptic.shape[1]
            I_syn_sum = np.zeros((3, N))
            for ni in range(3):
                if syn_by_post[ni]:
                    I_syn_sum[ni] = I_synaptic[np.array(syn_by_post[ni], dtype=int), :].sum(axis=0)

            vp = str(h5_file)
            I_ext_data = None
            t_volt_s   = None
            if vp.endswith('.h5') or vp.endswith('.hdf5'):
                
                with h5py.File(vp, 'r') as hf:
                    t_volt_s = hf['t'][:] / 1000.0
                    if 'I_ext_ABPD' in hf:
                        I_ext_data = np.stack([hf['I_ext_ABPD'][:],
                                               hf['I_ext_LP'][:],
                                               hf['I_ext_PY'][:]])
            else:
                _dv = np.load(vp, allow_pickle=True)
                t_volt_s = _dv['t'] / 1000.0
                if 'I_ext_ABPD' in _dv:
                    I_ext_data = np.stack([_dv['I_ext_ABPD'],
                                           _dv['I_ext_LP'],
                                           _dv['I_ext_PY']])

            has_iext = (I_ext_data is not None and np.any(I_ext_data != 0))

            def _get_window_times(cyc, window):
                lp_on,   lp_off   = cyc['lp']
                abpd_on, abpd_off = cyc['abpd']
                lp_next           = cyc['lp_next_on']
                if window == 'period':   return lp_on, lp_next
                if window == 'lp_burst': return lp_on, lp_off
                if window == 'lp_pd':    return lp_off, abpd_on
                raise ValueError(f"window desconocida: {window!r}")

            curr_sums = {}
            for window in ('period', 'lp_burst', 'lp_pd'):
                n_cyc_v = len(valid_cycles)
                ion_per = np.full((3, n_cyc_v), np.nan)
                syn_per = np.full((3, n_cyc_v), np.nan)
                ext_per = np.full((3, n_cyc_v), np.nan) if has_iext else None
                for ci, cyc in enumerate(valid_cycles):
                    try:
                        w_lo, w_hi = _get_window_times(cyc, window)
                    except (KeyError, ValueError):
                        continue
                    if w_hi <= w_lo:
                        continue
                    mask_c = (t_curr_s >= w_lo) & (t_curr_s < w_hi)
                    if mask_c.sum() == 0:
                        continue
                    for ni in range(3):
                        ion_per[ni, ci] = I_ion_sum[ni][mask_c].sum()
                        syn_per[ni, ci] = I_syn_sum[ni][mask_c].sum()
                    if has_iext and t_volt_s is not None:
                        mask_v = (t_volt_s >= w_lo) & (t_volt_s < w_hi)
                        if mask_v.sum() > 0:
                            for ni in range(3):
                                ext_per[ni, ci] = I_ext_data[ni][mask_v].sum()
                curr_sums[window] = {
                    'ion_per_cycle': ion_per,
                    'syn_per_cycle': syn_per,
                    'ext_per_cycle': ext_per,
                }
            print(f"  Sums computed for windows: {list(curr_sums.keys())}")
        except Exception as _e:
            import traceback
            print(f"  [warn] Error computing current sums: {_e}")
            traceback.print_exc()
            curr_sums = None

    # ── Save pickle ───────────────────────────────────────────────────────────
    out_pkl = args.out or f"{label}_data.pkl"
    payload = {
        "intervals"   : intervals,
        "valid_cycles": valid_cycles,
        "curr_sums"   : curr_sums,
        "label"       : label,
        "h5_file"    : h5_file,
        "curr_h5"    : curr_h5,
        "params"      : {
            "thr_up"  : THR_UP,
            "thr_down": THR_DOWN,
            "min_isi" : MIN_ISI,
            "skip"    : args.skip,
            "t_start" : args.t_start,
            "t_end"   : args.t_end,
        },
    }
    with open(out_pkl, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n  Pickle saved: {out_pkl}")


if __name__ == "__main__":
    main()