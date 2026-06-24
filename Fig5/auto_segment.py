"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d, median_filter
import seaborn as sns

#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})#Consistent fontsize for all figures

#Route to the analysed_data.pkl data file, by default in the previous folder to this script
path = "../intervals_data.pkl"
df_data = pd.read_pickle(path)


#keep only the experiments with invariants (5 discarded because of minor than 0.08 CV)
exps_to_analyze = ["1", "2", "3", "4", "6", "7", "10", "11", "12", "13", "14", "15", "16", "17", "18", "18p", "19", "20", "22", "22p", "23", "24"]
exps_to_analyze = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "18p", "19", "20","21", "22", "22p", "23", "24", "25"]
df_data = df_data.loc[:, df_data.columns.isin(exps_to_analyze)]

plt.rcParams['svg.fonttype'] = 'none'

def moving_average(signal, window=5):
    signal = np.asarray(signal, dtype=float)

    if window < 1:
        raise ValueError("Window size must be >= 1")

    kernel = np.ones(window) / window

    smoothed = np.convolve(signal, kernel, mode='same')

    return smoothed


def filtered_slope(data,
                   filter_type='median',
                   window=11,
                   slope_method='linregress',#'polyfit', 'linregress', 'first-last'
                   standardize=None):  # None, 'zscore', 'mean', 'norm'

    data = np.asarray(data)

    # Filtering
    if filter_type == 'mean':
        filtered = uniform_filter1d(data, size=window, mode='nearest')
    elif filter_type == 'median':
        filtered = median_filter(data, size=window, mode='nearest')
    elif filter_type == 'moving_avg':
        filtered = moving_average(data, window=5)
    else:
        raise ValueError("filter_type must be 'mean' 'median' or 'moving_avg'")

    # Standardization / normalization
    if standardize == 'zscore':
        std = filtered.std()
        if std > 0:
            filtered = (filtered - filtered.mean()) / std

    elif standardize == 'mean':
        mean = filtered.mean()
        if mean != 0:
            filtered = filtered / mean

    elif standardize == 'norm':#min-max
        data_range = filtered.max() - filtered.min()
        if data_range > 0:
            filtered = (filtered - filtered.min()) / data_range

    elif standardize is not None:
        raise ValueError("standardize must be None, 'zscore', 'mean', or 'norm'")

    x = np.arange(len(filtered))

    # Slope calculation
    if slope_method == 'polyfit':
        slope = abs(np.polyfit(x, filtered, 1)[0])

    elif slope_method == 'linregress':
        slope = abs(linregress(x, filtered).slope)

    elif slope_method == 'first-last':
        slope = abs((filtered[-1] - filtered[0]) / (x[-1] - x[0]))

    else:
        raise ValueError(
            "slope_method must be 'polyfit', 'linregress', or 'first-last'"
        )

    return slope


def calculate_R2(x, y, shift = 1):
    
    x = np.asarray(x)
    y = np.asarray(y)

    if shift > 0:
        x_cropped = x[:-shift]
        y_shifted = y[shift:]

    elif shift < 0:
        x_cropped = x[-shift:]
        y_shifted = y[:shift]

    else:
        x_cropped = x
        y_shifted = y

    res = linregress(x_cropped, y_shifted)

    return res.rvalue**2





df_chunks = pd.DataFrame(columns=["exp name", "chunk number", "LP period", "LPPD delay", "slope", "R2"])

def add_to_df_chunks(exp, chunk, start, end):

    lp_period = np.array(df_data[exp]["intervals"]["LP_period"])
    std = lp_period.std()
    lp_period = (lp_period - lp_period.mean()) / std

    lppd_delay = np.array(df_data[exp]["intervals"]["LPPD1_delay"])
    std = lppd_delay.std()
    lppd_delay = (lppd_delay - lppd_delay.mean()) / std

    df_chunks.loc[len(df_chunks)] = {
        "exp name": exp,
        "chunk number": chunk,
        "LP period": df_data[exp]["intervals"]["LP_period"][start:end],
        "LPPD delay": df_data[exp]["intervals"]["LPPD1_delay"][start:end],
        "slope": filtered_slope(lp_period[start:end]),
        "R2": calculate_R2(df_data[exp]["intervals"]["LP_period"][start:end], df_data[exp]["intervals"]["LPPD1_delay"][start:end])
    }




df_chunks = pd.DataFrame(columns=["exp name", "chunk number", "LP period", "LPPD delay", "slope", "R2"])


def find_highest_lowest_slope_chunks(exp,
                           window_size,
                           filter_type='median',
                           slope_method='linregress',
                           standardize=None):

    lp_period = np.asarray(df_data[exp]["intervals"]["LP_period"])
    lp_period = (lp_period - lp_period.mean()) / lp_period.std()

    lppd_delay = np.asarray(df_data[exp]["intervals"]["LPPD1_delay"])
    lppd_delay = (lppd_delay - lppd_delay.mean()) / lppd_delay.std()

    results = []

    for start in range(len(lp_period) - window_size + 1):

        end = start + window_size

        lp_chunk = lp_period[start:end]
        delay_chunk = lppd_delay[start:end]

        slope = filtered_slope(
            lp_chunk,
            filter_type=filter_type,
            slope_method=slope_method,
            standardize=standardize
        )

        r2 = calculate_R2(lp_chunk, delay_chunk)

        results.append({
            "start": start,
            "end": end,
            "slope": slope,
            "R2": r2,
            "lp_chunk": lp_chunk,
            "delay_chunk": delay_chunk
        })

    #Get two chunks: highest slope and lowest slope
    highest_chunk = max(results, key=lambda x: x["slope"])
    lowest_chunk = min(results, key=lambda x: x["slope"])
    
    return {
        "highest": highest_chunk,
        "lowest": lowest_chunk
    }
    
"""
#Test with 1 experiment
res = find_highest_lowest_slope_chunks("1", window_size=40)

print("highest")
print(res["highest"]["start"], res["highest"]["end"])
print("slope =", res["highest"]["slope"])
print("R2    =", res["highest"]["R2"])

print("\nlowest")
print(res["lowest"]["start"], res["lowest"]["end"])
print("slope =", res["lowest"]["slope"])
print("R2    =", res["lowest"]["R2"])
"""

def get_selected_chunks(exp, window_size):
    res = find_highest_lowest_slope_chunks(exp, window_size)

    return [
        {"exp": exp, "start": res["lowest"]["start"], "end": res["lowest"]["end"], "chunk": 1},
        {"exp": exp, "start": res["highest"]["start"], "end": res["highest"]["end"], "chunk": 2},
        
    ]
    

all_chunks = []

for exp in df_data.keys():

    #For experiments with only modulation, save only the highest slope chunk
    if exp == "4" or exp == "17":
        res = find_highest_lowest_slope_chunks(exp, window_size=40)
        chunks = [{
            "exp": exp,
            "start": res["highest"]["start"],
            "end": res["highest"]["end"],
            "chunk": 2
        }]

    #For experiments with only non-modulation, save only the lowest slope
    elif exp == "3" or exp == "1":
        res = find_highest_lowest_slope_chunks(exp, window_size=40)
        chunks = [{
            "exp": exp,
            "start": res["lowest"]["start"],
            "end": res["lowest"]["end"],
            "chunk": 1
        }]

    else:
        chunks = get_selected_chunks(exp, window_size=40)

    all_chunks.extend(chunks)


    #all_chunks.extend(get_selected_chunks(exp, window_size=40))



for c in all_chunks:
    add_to_df_chunks(
        exp=c["exp"],
        chunk=c["chunk"],
        start=c["start"],
        end=c["end"]
    )


# Jus get the n_exps with highest R² shifted 1 cycle (100 ensures all exps so no filtering)
n_exps = 100
df_chunks = (
    df_chunks[df_chunks["chunk number"].isin([1, 2])]
    .sort_values("R2", ascending=False)# This sorts the segments ascedning by R²
    .groupby("chunk number", group_keys=False)
    .head(n_exps)#Then take only the head (highest)
    .reset_index(drop=True)
)




results = []

#Calculate R² per shift from 0 to max_shift (for each chunk in each exp)
max_shift = 11
for exp_name, df_exp in df_chunks.groupby("exp name"):

    for chunk in [1, 2]:

        df_sub = df_exp[df_exp["chunk number"] == chunk]

        if df_sub.empty:
            continue

        x = np.hstack(df_sub["LP period"])
        y = np.hstack(df_sub["LPPD delay"])

        for shift in range(max_shift):
            r2 = calculate_R2(x, y, shift)

            #claculate cv for shift 0
            if shift == 0:
                cv = np.std(x, ddof=1) / np.mean(x)
            
            # skip this chunk if CV is too small
            if cv is not None and cv < 0.06:
                break
            
            results.append({
                "exp name": exp_name,
                "chunk number": chunk,
                "shift": shift,
                "R2": r2
            })


df_r2 = pd.DataFrame(results)


#MATPLOTLIB BOXPLOTS

def r2_shift_boxplots(df_chunk, title=None):
    """
    Creates boxplots of R2 vs shift with mean overlay.
    Returns fig, ax.
    """

    shifts = np.sort(df_chunk["shift"].unique())

    # build boxplot structure
    boxplot_data = [
        df_chunk[df_chunk["shift"] == s]["R2"].values
        for s in shifts
    ]

    # convert safely for mean computation
    means = np.array([
        np.mean(vals) if len(vals) > 0 else np.nan
        for vals in boxplot_data
    ])

    x = np.arange(len(shifts))

    fig, ax = plt.subplots(figsize=(14, 6))

    # Boxplots
    ax.boxplot(boxplot_data, positions=x)

    # Mean line
    ax.plot(
        x,
        means,
        "ro--",
        linewidth=2,
        markersize=6,
        label="Mean"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(shifts)

    ax.set_xlabel("cycle lag")
    ax.set_ylabel("R² (LPPDdelay—LPperiod)")

    if title is not None:
        ax.set_title(title)

    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


df_chunk1 = df_r2[df_r2["chunk number"] == 1]
df_chunk2 = df_r2[df_r2["chunk number"] == 2]

fig1, ax1 = r2_shift_boxplots(
    df_chunk1,
    title="Segments without slow modulation"
)

plt.savefig("r2_decay_no_mod.svg")

fig2, ax2 = r2_shift_boxplots(
    df_chunk2,
    title="Segments with slow modulation"
)

plt.savefig("r2_decay_slow_mod.svg")


#Plot a few chunks of each type
import numpy as np
import matplotlib.pyplot as plt
import math

def plot_decay(df_r2, exps_to_plot, norm_y=True):
    n = len(exps_to_plot)

    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.array(axes).flatten()


    for i, exp_id in enumerate(exps_to_plot):
        ax = axes[i]

        df_exp = df_r2[df_r2["exp name"] == exp_id].sort_values("shift")
        chunk_ids = df_exp["chunk number"].unique()
        ax.set_title(f"Exp {exp_id} | chunks: {chunk_ids}")

        x = df_exp["shift"].values
        y = df_exp["R2"].values

        ax.plot(x, y)

        if norm_y:
            ax.set_ylim(0, 1)

        ax.set_xticks(range(x.min(), x.max() + 1, 2))
        ax.set_title(f"Exp {exp_id}")


        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    return fig










def plot_decay(df_r2, exps_to_plot, chunk_number, norm_y=True):

    n = len(exps_to_plot)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 5*nrows))
    axes = np.array(axes).flatten()

    for i, exp_id in enumerate(exps_to_plot):
        ax = axes[i]

        df_exp = df_r2[df_r2["exp name"] == exp_id]

        df_exp = df_exp[df_exp["chunk number"] == chunk_number].sort_values("shift")

        if df_exp.empty:
            ax.set_title(f"Exp {exp_id} (no data)")
            ax.axis("off")
            continue

        x = df_exp["shift"].values
        y = df_exp["R2"].values

        ax.plot(x, y, linewidth=1)

        if norm_y:
            ax.set_ylim(0, 1)

        ax.set_xticks(np.arange(int(x.min()), int(x.max()) + 1, 2))
        ax.set_title(f"Exp {exp_id}")

        ax.set_box_aspect(1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # remove unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    return fig



no_trend_chunks = ["1", "3", "7", "11"]
trend_chunks = ["13", "18p", "15", "17"]

fig = plot_decay(df_r2, no_trend_chunks, chunk_number = 1, norm_y=True)
plt.savefig("decay_no_trend.svg")

fig = plot_decay(df_r2, trend_chunks, chunk_number = 2, norm_y=True)
plt.savefig("decay_trend.svg")


