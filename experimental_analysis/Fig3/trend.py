"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib.collections import LineCollection

#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})#Consistent fontsize for all figures

#Route to the analysed_data.pkl data file, by default in the previous folder to this script
path = "../intervals_data.pkl"
df_data = pd.read_pickle(path)



def interval_plotter(interval, x_label, y_label):
    fig = plt.figure(figsize=(10,5), constrained_layout = True)
    ax = fig.add_subplot(111)

    ax.plot(interval)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig



def invariant_plotter(x,y,shift, x_label, y_label):
    y_shifted = y[shift:]  
    x_trimmed = x[:-shift] if shift > 0 else x

    fig = plt.figure(figsize=(5,5), constrained_layout = True)
    ax = fig.add_subplot(111)

    ax.scatter(x_trimmed,y_shifted, s = 5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_box_aspect(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    return fig



def colored_interval_plotter(interval, x_label, y_label, cmap='jet', show_colorbar=True):
    fig = plt.figure(figsize=(10,5), constrained_layout=True)
    ax = fig.add_subplot(111)

    interval = np.asarray(interval)
    x = np.arange(len(interval))

    # Build line segments
    points = np.array([x, interval]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create colored line
    norm = plt.Normalize(0, len(interval))
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(np.arange(len(interval)))

    ax.add_collection(lc)

    # Set limits manually for LineCollection
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(interval.min(), interval.max())

    # Labels and style
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_colorbar:
        fig.colorbar(lc, ax=ax, label='Cycle')

    return fig



def colored_invariant_plotter(x, y, shift, x_label, y_label, cmap='jet', show_colorbar=False):
    y_shifted = y[shift:]  
    x_trimmed = x[:-shift] if shift > 0 else x

    # Color by index
    c = np.arange(len(x_trimmed))

    fig = plt.figure(figsize=(5,5), constrained_layout=True)
    ax = fig.add_subplot(111)

    sc = ax.scatter(x_trimmed, y_shifted, c=c, cmap=cmap, s=5)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_box_aspect(1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if show_colorbar:
        fig.colorbar(sc, ax=ax, label='Cycle')

    return fig




##
# EXP WITH NO TREND
#
exp = "10"


figure = colored_interval_plotter(df_data[exp]["intervals"]["LPPD1_delay"], x_label = "Cycle #", y_label="LPPD delay (ms)")
plt.savefig(f"interval_exp{exp}.svg")



figure = colored_invariant_plotter(df_data[exp]["intervals"]["LP_period"], df_data[exp]["intervals"]["LPPD1_delay"], shift = 0, x_label = "Cycle $i$ LP period (ms)", y_label="Cycle $i$ LPPD delay (ms)")
plt.savefig(f"invariant_i_exp{exp}.svg")

figure = colored_invariant_plotter(df_data[exp]["intervals"]["LP_period"], df_data[exp]["intervals"]["LPPD1_delay"], shift = 1, x_label = "Cycle $i$ LP period (ms)", y_label="Cycle $i$+1 LPPD delay (ms)")
plt.savefig(f"invariant_i+1_exp{exp}.svg")



##
# EXP WITH TREND
#
exp = "6"


figure = colored_interval_plotter(df_data[exp]["intervals"]["LPPD1_delay"], x_label = "Cycle #", y_label="LPPD delay (ms)")
plt.savefig(f"interval_exp{exp}.svg")



figure = colored_invariant_plotter(df_data[exp]["intervals"]["LP_period"], df_data[exp]["intervals"]["LPPD1_delay"], shift = 0, x_label = "Cycle $i$ LP period (ms)", y_label="Cycle $i$ LPPD delay (ms)")
plt.savefig(f"invariant_i_exp{exp}.svg")

figure = colored_invariant_plotter(df_data[exp]["intervals"]["LP_period"], df_data[exp]["intervals"]["LPPD1_delay"], shift = 1, x_label = "Cycle $i$ LP period (ms)", y_label="Cycle $i$+1 LPPD delay (ms)")
plt.savefig(f"invariant_i+1_exp{exp}.svg")



