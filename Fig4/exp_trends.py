"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 16})#Consistent fontsize for all figures

#Route to the analysed_data.pkl data file, by default in the previous folder to this script
path = "../intervals_data.pkl"
df_data = pd.read_pickle(path)


regions_13 = [(0,200),(2000,2200), (3100,3300), (4800, 5000), (6000,6200), (7200,7400), (8200,8400), (10000,10200)]

exp = "13"
regions = regions_13



delay = df_data[exp]["intervals"]["LPPD1_delay"]
period = df_data[exp]["intervals"]["LP_period"]


def r2_shifted(x, y, shift):
    x = np.asarray(x)
    y = np.asarray(y)
    
    # shift y forward (truncate so lengths match)
    y_shifted = y[shift:]  
    x_trimmed = x[:-shift] if shift > 0 else x

    corr = np.corrcoef(x_trimmed, y_shifted)[0, 1]
    r2 = corr ** 2
    return r2


def plot_r2_by_regions(signal1, signal2, regions, shift=0):
    signal1 = np.asarray(signal1)
    signal2 = np.asarray(signal2)

    r2_values = []

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot full signals
    ax.plot(signal1, label="LP period", linewidth=1.5, color = "#303030") #color = "#303030" grey
    ax.plot(signal2, label="LPPD delay", linewidth=1.5, color = "#b4b4b4ff")#ffc3f4ff" pink

    # Compute global y-range for text placement
    y_min = min(signal1.min(), signal2.min())
    y_max = max(signal1.max(), signal2.max())
    y_range = y_max - y_min

    for start, end in regions:
        x_chunk = signal1[start:end]
        y_chunk = signal2[start:end]

        r2 = r2_shifted(x_chunk, y_chunk, shift)
        r2_values.append(r2)

        # Normalize intensity (clip in case of NaN)
        alpha = 0 if np.isnan(r2) else np.clip(r2, 0, 1)

        # Shade region
        # Fill (data-dependent transparency)
        ax.axvspan(
            start,
            end,
            facecolor='red',
            alpha=alpha * 0.7,
            linewidth=0  # no border here
        )

        # Borders
        #ax.axvline(start, color='black', linewidth=0.2)
        #ax.axvline(end, color='black', linewidth=0.2)

        # Annotate R²
        ax.text(
            (start + end) / 2,
            y_max + 0.13 * y_range,
            f"{r2:.2f}" if not np.isnan(r2) else "nan",
            ha='center',
            va='top',
            fontsize=12,
            color='red'
        )
    ax.legend()
    ax.set_xlabel("Cycle #")
    ax.set_ylabel("Interval (ms)")

    return r2_values, fig, ax


r2_vals, fig, ax = plot_r2_by_regions(period, delay, regions, shift=1)
plt.tight_layout()
plt.savefig("./trends_exp13.svg")



