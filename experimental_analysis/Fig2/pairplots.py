"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

#GENERAL CONSTANTS
plt.rcParams.update({'font.size': 20})#Consistent fontsize for all figures

#Route to the analysed_data.pkl data file, by default in the previous folder to this script
path = "../intervals_data.pkl"
df_data = pd.read_pickle(path)


def custom_pairplots(dict, shift=0, cmap = plt.cm.Reds, diagonal_plot = "histogram", scatter_color = "#555555", scatter_size = 3, scatter_gradient = "False"):
    df = pd.DataFrame(dict)#ensure its always a dataframe but can also be given a dataframe already
    cols = df.columns
    n = len(cols)

    fig, axes = plt.subplots(n, n, figsize=(3.5*n, 3.5*n))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            ax.set_box_aspect(1)

            x = df[cols[j]]
            y = df[cols[i]]

            x_cropped = x[shift:]
            y_shifted = y[:-shift] if shift > 0 else y
            
            # Color gradient based on index
            colors = np.linspace(0, 1, len(x_cropped))
            
            if i > j:      

                if scatter_gradient == "True":
                    # Lower triangle: scatter with gradient color based on index
                    ax.scatter(x_cropped, y_shifted, s=scatter_size, c = colors, cmap=plt.cm.jet)
                else:
                    # Lower triangle: scatter
                    ax.scatter(x_cropped, y_shifted, s=scatter_size, color = scatter_color)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

            elif i == j:

                # Diagonal histogram
                if diagonal_plot == "histogram":
                    ax.hist(x_cropped, bins=20, color="#818181", edgecolor="#FFFFFF", linewidth = 0.3)
                # Diagonal scatter
                elif diagonal_plot == "scatter":
                    x_shifted = x[:-shift] if shift > 0 else x
                    ax.scatter(x_cropped, x_shifted, s=scatter_size, color = scatter_color)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3))

            else:
                # Upper triangle: R²
                r = np.corrcoef(x_cropped, y_shifted)[0, 1]
                r2 = r**2

                # background color
                ax.set_facecolor(cmap(r2))

                # Keep text readable (black or white depending on intensity)
                text_color = "white" if r2 > 0.5 else "black"

                ax.text(0.5, 0.5,
                        f"{r2:.2f}",
                        ha='center', va='center',
                        transform=ax.transAxes,
                        fontsize=30,
                        color = text_color)

                ax.set_xticks([])
                ax.set_yticks([])

                # Remove borders (spines)
                for spine in ax.spines.values():
                    spine.set_visible(False)
               

            # Labels only on outer edges
            if i == n - 1:
                ax.set_xlabel(cols[j])
            else:
                ax.set_xticklabels([])

            if j == 0:
                ax.set_ylabel(cols[i])
            else:
                ax.set_yticklabels([])


            

    return fig




#Exp 10 no shift
exp = "10"
shift = 0

plot_dict = {
    "LP period": np.array(df_data[exp]["intervals"]["LP_period"]),
    "LPPD delay": np.array(df_data[exp]["intervals"]["LPPD1_delay"]),
    "PDLP delay": np.array(df_data[exp]["intervals"]["PD1LP_delay"]),
    "LP burst": np.array(df_data[exp]["intervals"]["LP_burst"]),
    "PD burst": np.array(df_data[exp]["intervals"]["PD1_burst"])
}

df = pd.DataFrame(plot_dict)
figure = custom_pairplots(df, shift)
plt.savefig(f"pp_exp{exp}_shift{shift}.svg")


#Exp 10 shift 1 cycle
shift = 1



df = pd.DataFrame(plot_dict)
figure = custom_pairplots(df, shift, diagonal_plot="scatter")
plt.savefig(f"pp_exp{exp}_shift{shift}.svg")


#To plot all exps
"""for exp in df_data.keys():
    intervals_dict = df_data[exp]["intervals"]
    selected_intervals = ["LP_period", "LP_burst", "LP_hyperpolarization", "PD1_burst", "LPPD1_delay", "LPPD1_interval", "PD1LP_delay", "PD1LP_interval"]
    selected_dict = {k: intervals_dict[k] for k in selected_intervals if k in intervals_dict}
    
    shift = 1

    figure = custom_pairplots(selected_dict, shift)

    figure.savefig(f'PP_exp{exp}_shift_{shift}')"""
