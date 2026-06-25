"""
pyloric_currents.py
===================
Extracts and saves the ionic and synaptic currents of a pyloric simulation.

The simulator (simulator.pyx) computes membrane conductances at each
timestep and returns them in `membrane_conds` (shape: 3 neurons x 8 channels
x N). Synaptic conductances are in `synaptic_conds` (shape: 7 synapses x N).

This module converts them to currents (µA/cm², density) by multiplying by
(V - E_rev) and saves them to .h5 files separate from the voltage data.
Use the AREA_CM2 constant in pyloric_data.py to convert to absolute µA.

Neurons  : 0=AB/PD, 1=LP, 2=PY
Channels : 0=Na, 1=CaT, 2=CaS, 3=A, 4=KCa, 5=Kd, 6=H, 7=leak

Synapses — row order in synaptic_conds (7 rows)
------------------------------------------------
The order is set by build_conns() in pyloric/utils.py. Verified order (see
DEFAULT_CONN_LABELS / DEFAULT_CONN_POST below):

    0: AB/PD -> LP   (glut, E=-70)
    1: AB/PD -> LP   (chol, E=-80)
    2: AB/PD -> PY   (glut, E=-70)
    3: AB/PD -> PY   (chol, E=-80)
    4: LP    -> AB/PD(glut, E=-70)
    5: LP    -> PY   (glut, E=-70)
    6: PY    -> LP   (glut, E=-70)


Note — track_currents
----------------------
`membrane_conds`, `synaptic_conds`, and `reversal_calcium` are only
returned when calling simulate() with track_currents=True (interface.py
maps it internally to num_energyscape_timesteps=len(t)).

With simulate_with_Iext use:
    out = simulate_with_Iext(..., track_currents=True)
or pass num_energyscape_timesteps=N directly.
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Model constants (Prinz 2004 / mackelab)
# ─────────────────────────────────────────────────────────────────────────────

# Membrane channels — order confirmed in simulator.pyx (line ~487):
#   membrane_conds[:, 0, :] = Na
#   membrane_conds[:, 1, :] = CaT
#   membrane_conds[:, 2, :] = CaS
#   membrane_conds[:, 3, :] = A
#   membrane_conds[:, 4, :] = KCa
#   membrane_conds[:, 5, :] = Kd
#   membrane_conds[:, 6, :] = H
#   membrane_conds[:, 7, :] = leak
CHANNEL_NAMES = ['Na', 'CaT', 'CaS', 'A', 'KCa', 'Kd', 'H', 'leak']
NEURON_NAMES  = ['AB/PD', 'LP', 'PY']

# Fixed reversal potentials (mV) — Ca is handled separately (Nernst)
E_REV_FIXED = {
    'Na'  : 50.0,
    'CaT' : None,   # Nernst, time-dependent -> use reversal_calcium
    'CaS' : None,   # same
    'A'   : -80.0,
    'KCa' : -80.0,
    'Kd'  : -80.0,
    'H'   : -20.0,
    'leak': -50.0,
}

# Indices on axis 1 of membrane_conds (neuron, channel, time)
CH_IDX = {name: i for i, name in enumerate(CHANNEL_NAMES)}

# ─────────────────────────────────────────────────────────────────────────────
# Synapses
# Exact order from build_conns() in pyloric/utils/circuit_parameters.py:
#
#   idx  post      pre    type   params[]  E_syn
#    0   LP(1)   AB/PD(0) glut  params[0]  -70
#    1   LP(1)   AB/PD(0) chol  params[1]  -80
#    2   PY(2)   AB/PD(0) glut  params[2]  -70
#    3   PY(2)   AB/PD(0) chol  params[3]  -80
#    4   AB/PD(0) LP(1)   glut  params[4]  -70
#    5   PY(2)    LP(1)   glut  params[5]  -70
#    6   LP(1)    PY(2)   glut  params[6]  -70
#
# Synapse names (_synapse_names in circuit_parameters.py):
#   ["AB-LP", "PD-LP", "AB-PY", "PD-PY", "LP-PD", "LP-PY", "PY-LP"]
#   AB and PD are both neurons of the AB/PD ganglion (neuron 0 here).
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONN_LABELS = [
    'AB-LP (glut)',   # 0  post=1  pre=0
    'PD-LP (chol)',   # 1  post=1  pre=0
    'AB-PY (glut)',   # 2  post=2  pre=0
    'PD-PY (chol)',   # 3  post=2  pre=0
    'LP-PD (glut)',   # 4  post=0  pre=1
    'LP-PY (glut)',   # 5  post=2  pre=1
    'PY-LP (glut)',   # 6  post=1  pre=2
]

# Postsynaptic neuron (0=AB/PD, 1=LP, 2=PY)
DEFAULT_CONN_POST = [1, 1, 2, 2, 0, 2, 1]

# Synaptic reversal potential by type (mV)
E_SYN_GLUT = -70.0
E_SYN_CHOL = -80.0

# Parallel to DEFAULT_CONN_LABELS
DEFAULT_E_SYN = [E_SYN_GLUT, E_SYN_CHOL,
                 E_SYN_GLUT, E_SYN_CHOL,
                 E_SYN_GLUT, E_SYN_GLUT, E_SYN_GLUT]


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def inspect_conn_order(params=None, dt: float = 0.025, t_max: float = 500.0) -> None:
    """
    Prints the real synapse order in synaptic_conds by querying
    build_conns from pyloric/utils.py.

    Uses a prior sample if no params are given. Run this once in your
    environment to verify DEFAULT_CONN_LABELS before analyzing data.

    Usage
    -----
    from pyloric_currents import inspect_conn_order
    inspect_conn_order()
    """
    try:
        from pyloric.utils import build_conns
        from pyloric import create_prior
        import numpy as np
    except ImportError:
        print("[inspect_conn_order] Could not import pyloric. "
              "Install it from pyloric-main.")
        return

    # Dummy conductances — only the structure/order matters.
    # IMPORTANT: same default kwargs as interface.py, so row order
    # matches the real synaptic_conds.
    fake_syn = np.array([-1.0] * 7)
    conns = build_conns(
        -np.exp(fake_syn),
        Esglut=[-70, -70, -70, -70, -70],
        kminusglut=[40, 40, 40, 40, 40],
        Eschol=[-80, -80],
        kminuschol=[100, 100],
        Vth=[-35, -35, -35, -35, -35, -35, -35],
        Delta=[5, 5, 5, 5, 5, 5, 5],
    )

    neuron_names = ['AB/PD', 'LP', 'PY']
    print("\n[inspect_conn_order] Real row order in synaptic_conds:")
    print(f"  {'idx':>4}  {'post':>8}  {'pre':>8}  {'E_syn (mV)':>12}  {'k_minus':>10}")
    print("  " + "-" * 55)
    for k, row in enumerate(conns):
        npost, npre, g, Es, km = int(row[0]), int(row[1]), row[2], row[3], row[4]
        print(f"  {k:>4}  {neuron_names[npost]:>8}  {neuron_names[npre]:>8}  "
              f"{Es:>12.1f}  {km:>10.1f}")
    print()
    print("  Compare against DEFAULT_CONN_LABELS / DEFAULT_CONN_POST in pyloric_currents.py.")
    print("  Update those constants if they differ.\n")


def extract_currents(
    out: dict,
    dt: float = 0.025,
    conn_labels: list[str] | None = None,
    conn_post: list[int] | None = None,
    e_syn_per_conn: list[float] | None = None,
) -> dict:
    """
    Computes ionic and synaptic currents from the simulate() output.

    Parameters
    ----------
    out            : dict returned by pyloric.simulate or simulate_with_Iext.
                     Must contain 'voltage' (or 'Vs'), 'membrane_conds',
                     'synaptic_conds', 'reversal_calcium'.
                     -> Requires having called with track_currents=True.
    dt             : Timestep (ms).
    conn_labels    : Labels for the 7 synapses (list of strings).
                     Default: DEFAULT_CONN_LABELS.
    conn_post      : Postsynaptic neuron index for each synapse.
                     Default: DEFAULT_CONN_POST.
    e_syn_per_conn : Reversal potential (mV) for each synapse.
                     Default: DEFAULT_E_SYN  (-70 glut, -80 chol).

    Returns
    -------
    dict with:
        't'              : np.ndarray (N,)      — time vector (ms)
        'Vs'             : np.ndarray (3, N)    — membrane potential (mV)
        'I_ionic'        : np.ndarray (3, 8, N) — ionic currents (µA/cm², density)
                           axis 1 -> [Na, CaT, CaS, A, KCa, Kd, H, leak]
        'I_synaptic'     : np.ndarray (7, N)    — current per connection (µA/cm², density)
        'I_syn_total'    : np.ndarray (3, N)    — synaptic sum per neuron
        'channel_names'  : list[str]
        'neuron_names'   : list[str]
        'conn_labels'    : list[str]
        'conn_post'      : list[int]
        'e_syn_per_conn' : list[float]
        'dt'             : float
    """
    if conn_labels is None:
        conn_labels = DEFAULT_CONN_LABELS
    if conn_post is None:
        conn_post = DEFAULT_CONN_POST
    if e_syn_per_conn is None:
        e_syn_per_conn = DEFAULT_E_SYN

    # ── Simulator tensors ────────────────────────────────────────────────────
    Vs     = out.get('voltage', out.get('Vs'))   # interface.py uses 'voltage'
    g_memb = out.get('membrane_conds')           # (3, 8, N)  — mS/cm² (density)
    g_syn  = out.get('synaptic_conds')           # (7, N)     — mS/cm² (density)
    ECa_t  = out.get('reversal_calcium')         # (3, N)

    if Vs is None:
        raise KeyError("Output dict has neither 'voltage' nor 'Vs'.")
    if g_memb is None or g_syn is None or ECa_t is None:
        raise KeyError(
            "Missing 'membrane_conds', 'synaptic_conds', or 'reversal_calcium'.\n"
            "Call simulate() with track_currents=True  (or simulate_with_Iext\n"
            "with num_energyscape_timesteps=N)."
        )

    n_neurons, n_channels, N = g_memb.shape
    t = np.arange(N) * dt

    # ── Ionic currents  I = g x (V - E_rev) ──────────────────────────────────
    # Shape -> (3 neurons, 8 channels, N timesteps)
    I_ionic = np.zeros((n_neurons, n_channels, N))

    for j in range(n_neurons):
        V_j = Vs[j]                        # (N,)
        for ch_name, ch_i in CH_IDX.items():
            g = g_memb[j, ch_i]            # (N,)
            if ch_name in ('CaT', 'CaS'):
                E = ECa_t[j]              # (N,) Nernst, time-dependent
            else:
                E = E_REV_FIXED[ch_name]  # fixed scalar
            I_ionic[j, ch_i] = g * (V_j - E)

    # ── Synaptic currents  I_syn_k = g_syn_k x (V_post - E_syn_k) ───────────
    # synaptic_conds returns the dynamic conductance g(t) with the negative
    # sign already baked in (build_conns receives -exp(...)). I = g·(V-E).
    #
    # Quick diagnostic: if PD->LP looks flat/inactive, run:
    #   inspect_conn_order()   <- confirms which index is which synapse
    #   print(g_syn[:, :100].min(axis=1))  <- if a row is always ~0,
    #                                          its conn_post may be wrong
    n_syn = g_syn.shape[0]
    I_synaptic = np.zeros((n_syn, N))

    for k in range(n_syn):
        npost  = conn_post[k] if k < len(conn_post) else 0
        V_post = Vs[npost]
        E_k    = e_syn_per_conn[k] if k < len(e_syn_per_conn) else E_SYN_GLUT
        I_synaptic[k] = g_syn[k] * (V_post - E_k)

    # ── Total synaptic sum per neuron ────────────────────────────────────────
    I_syn_total = np.zeros((n_neurons, N))
    for k in range(n_syn):
        npost = conn_post[k] if k < len(conn_post) else 0
        I_syn_total[npost] += I_synaptic[k]

    return {
        't'             : t,
        'Vs'            : Vs,
        'I_ionic'       : I_ionic,
        'I_synaptic'    : I_synaptic,
        'I_syn_total'   : I_syn_total,
        'channel_names' : CHANNEL_NAMES,
        'neuron_names'  : NEURON_NAMES,
        'conn_labels'   : conn_labels,
        'conn_post'     : conn_post,
        'e_syn_per_conn': e_syn_per_conn,
        'dt'            : dt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Save / load
# ─────────────────────────────────────────────────────────────────────────────

def save_currents(curr: dict, path: str | Path = 'pyloric_currents.h5') -> None:
    """
    Saves the currents dict to an .h5 file.

    The file contains:
        t              — time vector (ms)
        Vs             — voltages (3, N)
        I_ionic        — ionic currents (3, 8, N)   µA/cm² (density)
                         axis 1: [Na, CaT, CaS, A, KCa, Kd, H, leak]
        I_synaptic     — synaptic currents (7, N)   µA/cm² (density)
        I_syn_total    — synaptic sum per neuron (3, N)
        channel_names  — channel names (8 strings)
        neuron_names   — neuron names (3 strings)
        conn_labels    — synapse labels (7 strings)
        conn_post      — postsynaptic neuron indices (7 ints)
        e_syn_per_conn — reversal potential per synapse (7 floats)
        dt             — timestep (ms)

    Usage
    -----
    save_currents(curr, 'results/currents_run01.h5')
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        t              = curr['t'],
        Vs             = curr['Vs'],
        I_ionic        = curr['I_ionic'],
        I_synaptic     = curr['I_synaptic'],
        I_syn_total    = curr['I_syn_total'],
        channel_names  = np.array(curr['channel_names']),
        neuron_names   = np.array(curr['neuron_names']),
        conn_labels    = np.array(curr['conn_labels']),
        conn_post      = np.array(curr['conn_post']),
        e_syn_per_conn = np.array(curr.get('e_syn_per_conn', DEFAULT_E_SYN)),
        dt             = curr['dt'],
    )
    print(f'[pyloric_currents] Saved to: {path}')


def load_currents(path: str | Path) -> dict:
    """
    Loads an .h5 file produced by save_currents and returns the dict.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=False)

    curr = {
        't'             : data['t'],
        'Vs'            : data['Vs'],
        'I_ionic'       : data['I_ionic'],
        'I_synaptic'    : data['I_synaptic'],
        'I_syn_total'   : data['I_syn_total'],
        'channel_names' : list(data['channel_names'].astype(str)),
        'neuron_names'  : list(data['neuron_names'].astype(str)),
        'conn_labels'   : list(data['conn_labels'].astype(str)),
        'conn_post'     : list(data['conn_post'].astype(int)),
        'e_syn_per_conn': list(data['e_syn_per_conn'].astype(float)),
        'dt'            : float(data['dt']),
    }
    return curr


def diagnose_synaptic(curr: dict, t_range=None) -> None:
    """
    Prints amplitude statistics for each synaptic current.

    Useful to detect an inactive synapse (e.g. PD->LP): if its range is
    ~0 or its std is much smaller than the rest, conn_post or the
    current calculation likely has a bug.

    Usage
    -----
    curr = load_currents('..._currents.h5')  # or dict from extract_currents
    diagnose_synaptic(curr)
    diagnose_synaptic(curr, t_range=(1000, 5000))  # time zoom (ms)
    """
    t      = curr['t']
    I_syn  = curr['I_synaptic']   # (7, N)
    labels = curr['conn_labels']
    posts  = curr['conn_post']

    if t_range is not None:
        mask  = (t >= t_range[0]) & (t <= t_range[1])
        I_syn = I_syn[:, mask]

    neuron_names = ['AB/PD', 'LP', 'PY']
    print("\n[diagnose_synaptic] Synaptic current amplitudes:")
    print(f"  {'idx':>3}  {'label':<20}  {'post':>8}  {'min':>10}  {'max':>10}  "
          f"{'std':>10}  {'mean':>10}")
    print("  " + "-" * 72)
    for k in range(I_syn.shape[0]):
        lbl  = labels[k] if k < len(labels) else f'syn{k}'
        post = posts[k]  if k < len(posts)  else '?'
        pname = neuron_names[post] if isinstance(post, int) and post < 3 else str(post)
        row = I_syn[k]
        print(f"  {k:>3}  {lbl:<20}  {pname:>8}  "
              f"{row.min():>10.4f}  {row.max():>10.4f}  "
              f"{row.std():>10.4f}  {row.mean():>10.4f}")
    print()
    print("  If a synapse that should be active has std~0 or range~0,")
    print("  its conn_post or index in synaptic_conds is likely wrong.")
    print("  Run inspect_conn_order() to verify the real order.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_currents(
    curr: dict,
    neuron: int | str = 0,
    t_range: tuple[float, float] | None = None,
    include_synaptic: bool = True,
    save_path: str | None = None,
):
    """
    Plots ionic (and synaptic) currents for a given neuron.

    Parameters
    ----------
    curr            : dict returned by extract_currents or load_currents
    neuron          : int (0,1,2) or string ('AB/PD','LP','PY')
    t_range         : (t_start, t_end) in ms, for a time zoom
    include_synaptic: if True, adds a panel with the total synaptic current
    save_path       : path to save the figure (None = display only)
    """
    import matplotlib.pyplot as plt

    neuron_names  = curr['neuron_names']
    channel_names = curr['channel_names']

    if isinstance(neuron, str):
        neuron = neuron_names.index(neuron)

    t        = curr['t']
    I_ionic  = curr['I_ionic'][neuron]   # (8, N)
    I_syn    = curr['I_syn_total'][neuron]  # (N,)
    Vs       = curr['Vs'][neuron]           # (N,)

    # Time zoom
    if t_range is not None:
        mask = (t >= t_range[0]) & (t <= t_range[1])
        t       = t[mask]
        I_ionic = I_ionic[:, mask]
        I_syn   = I_syn[mask]
        Vs      = Vs[mask]

    n_panels = 1 + len(channel_names) + (1 if include_synaptic else 0)
    fig, axes = plt.subplots(n_panels, 1,
                             figsize=(14, 1.8 * n_panels),
                             sharex=True,
                             gridspec_kw={'hspace': 0.05})

    # Voltage
    axes[0].plot(t, Vs, lw=0.6, color='k')
    axes[0].set_ylabel('V (mV)', fontsize=8)
    axes[0].set_title(f'Neuron {neuron_names[neuron]}', fontsize=10)

    # One ionic current per panel
    colors = plt.cm.tab10.colors
    for ch_i, ch_name in enumerate(channel_names):
        ax = axes[ch_i + 1]
        ax.plot(t, I_ionic[ch_i], lw=0.6, color=colors[ch_i % 10])
        ax.set_ylabel(f'I_{ch_name}\n(µA/cm²)', fontsize=7)
        ax.axhline(0, color='gray', lw=0.4, ls='--')

    # Total synaptic current
    if include_synaptic:
        ax = axes[-1]
        ax.plot(t, I_syn, lw=0.6, color='purple')
        ax.set_ylabel('I_syn\n(µA/cm²)', fontsize=7)
        ax.axhline(0, color='gray', lw=0.4, ls='--')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (ms)', fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[pyloric_currents] Figure saved to: {save_path}')

    plt.show()
    return fig, axes


def plot_synaptic_currents(
    curr: dict,
    t_range: tuple[float, float] | None = None,
    save_path: str | None = None,
):
    """
    Plots the 7 individual synaptic currents.
    """
    import matplotlib.pyplot as plt

    t        = curr['t']
    I_syn    = curr['I_synaptic']    # (7, N)
    labels   = curr['conn_labels']

    if t_range is not None:
        mask = (t >= t_range[0]) & (t <= t_range[1])
        t     = t[mask]
        I_syn = I_syn[:, mask]

    n_syn = I_syn.shape[0]
    fig, axes = plt.subplots(n_syn, 1,
                             figsize=(14, 1.8 * n_syn),
                             sharex=True,
                             gridspec_kw={'hspace': 0.05})

    colors = plt.cm.Set1.colors
    for k in range(n_syn):
        ax = axes[k]
        ax.plot(t, I_syn[k], lw=0.6, color=colors[k % len(colors)])
        ax.set_ylabel(labels[k] if k < len(labels) else f'syn{k}', fontsize=8)
        ax.axhline(0, color='gray', lw=0.4, ls='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel('Time (ms)', fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[pyloric_currents] Figure saved to: {save_path}')

    plt.show()
    return fig, axes