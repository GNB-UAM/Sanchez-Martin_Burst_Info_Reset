"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from matplotlib.ticker import MaxNLocator


#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})#Consistent fontsize for all figures

#Route to the analysed_data.pkl data file, by default in the previous folder to this script
path = "../intervals_data.pkl"
df_data = pd.read_pickle(path)


exp = "10"


x1 = df_data[exp]["intervals"]["LP_period"]
y1 = df_data[exp]["intervals"]["LPPD1_interval"]
x1_label = "LP period (ms)"
y1_label = "LPPD interval (ms)"

x2 = df_data[exp]["intervals"]["LP_period"]
y2 = df_data[exp]["intervals"]["LPPD1_delay"]
x2_label = "LP period (ms)"
y2_label = "LPPD delay (ms)"

x3 = df_data[exp]["intervals"]["LP_period"]
y3 = df_data[exp]["intervals"]["PD1_burst"]
x3_label = "LP period (ms)"
y3_label = "PD burst (ms)"




fig, axes = plt.subplots(1, 3, figsize=(16, 7))
dotsize = 8

axes[0].scatter(x1, y1, s = dotsize, edgecolors='none', linewidths=0, color = "#555555")
axes[0].set_box_aspect(1)
axes[0].set_xlabel(x1_label)
axes[0].set_ylabel(y1_label)
slope, intercept, r, p, stderr = linregress(x1, y1)
r2 = r**2
axes[0].text(
            0.05, 0.95,
            f"$R^2$ = {r2:.3f}",
            transform=axes[0].transAxes,
            va='top'
        )

axes[1].scatter(x2, y2, s = dotsize, edgecolors='none', linewidths=0, color = "#555555")
axes[1].set_box_aspect(1)
axes[1].set_xlabel(x2_label)
axes[1].set_ylabel(y2_label)
slope, intercept, r, p, stderr = linregress(x2, y2)
r2 = r**2
axes[1].text(
            0.05, 0.95,
            f"$R^2$ = {r2:.3f}",
            transform=axes[1].transAxes,
            va='top'
        )


axes[2].scatter(x3, y3, s = dotsize, edgecolors='none', linewidths=0, color = "#555555")
axes[2].set_box_aspect(1)
axes[2].set_xlabel(x3_label)
axes[2].set_ylabel(y3_label)
slope, intercept, r, p, stderr = linregress(x3, y3)
r2 = r**2
axes[2].text(
            0.95, 0.95,
            f"$R^2$ = {r2:.3f}",
            transform=axes[2].transAxes,
            ha='right',
            va='top'
        )


for ax in axes:
    ax.spines[['top', 'right']].set_visible(False)

    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(3))


plt.tight_layout()
plt.savefig("invariants.svg")