import numpy as np
import pandas as pd




path = "./spikes_data.pkl"
df_data = pd.read_pickle(path)


def intervals_PD_reference(PD1_beg, PD1_end, PD2_beg, PD2_end, LP_beg, LP_end):
    #sanity check which one is before PD or LP (in this case it should start with PD)
    #If first PD burst is higher than first LP (PD after LP), we ignore first LP cycle
    if PD1_beg[0] > LP_beg[0]:
          LP_beg = LP_beg[1:]
          LP_end = LP_end[1:]
    
    LP_period = []
    LP_burst = []
    LP_hyperpol = []
    PD1_period = []
    PD1_burst = []
    PD1_hyperpol = []
    PD2_period = []
    PD2_burst = []
    PD2_hyperpol = []

    LPPD1_delay = []
    LPPD1_interval = []
    PD1LP_delay = []
    PD1LP_interval = []

    LPPD2_delay = []
    LPPD2_interval = []
    PD2LP_delay = []
    PD2LP_interval = []

    for i in range(len(PD1_beg) - 1):
                
        #Append value of intervals		
        LP_period.append(LP_beg[i+1] - LP_beg[i])
        LP_burst.append(LP_end[i] - LP_beg[i])
        LP_hyperpol.append(LP_beg[i+1] - LP_end[i])
        PD1_period.append(PD1_beg[i+1] - PD1_beg[i])
        PD1_burst.append(PD1_end[i] - PD1_beg[i])
        PD1_hyperpol.append(PD1_beg[i+1] - PD1_end[i])
        PD2_period.append(PD2_beg[i+1] - PD2_beg[i])
        PD2_burst.append(PD2_end[i] - PD2_beg[i])
        PD2_hyperpol.append(PD2_beg[i+1] - PD2_end[i])

        LPPD1_delay.append(PD1_beg[i+1] - LP_end[i])
        LPPD1_interval.append(PD1_beg[i+1] - LP_beg[i])
        PD1LP_delay.append(LP_beg[i] - PD1_end[i])
        PD1LP_interval.append(LP_beg[i] - PD1_beg[i])

        LPPD2_delay.append(PD2_beg[i+1] - LP_end[i])
        LPPD2_interval.append(PD2_beg[i+1] - LP_beg[i])
        PD2LP_delay.append(LP_beg[i] - PD2_end[i])
        PD2LP_interval.append(LP_beg[i]- PD2_beg[i])

    intervals_both_dict = {
        "LP_period" : LP_period,
        "LP_burst" : LP_burst,
		"LP_hyperpolarization" : LP_hyperpol,
		"PD1_period" : PD1_period,
		"PD1_burst" : PD1_burst,
		"PD1_hyperpolarization" : PD1_hyperpol,
		"PD2_period" : PD2_period,
		"PD2_burst" : PD2_burst,
		"PD2_hyperpolarization" : PD2_hyperpol,
		"LPPD1_delay" : LPPD1_delay,
		"LPPD1_interval" : LPPD1_interval,
		"PD1LP_delay" : PD1LP_delay,
		"PD1LP_interval" : PD1LP_interval,
		"LPPD2_delay" : LPPD2_delay,
		"LPPD2_interval" : LPPD2_interval,
		"PD2LP_delay" : PD2LP_delay,
		"PD2LP_interval" : PD2LP_interval}
		
    intervals_PD1_dict = {
        "LP_period" : LP_period,
        "LP_burst" : LP_burst,
        "LP_hyperpolarization" : LP_hyperpol,
        "PD1_period" : PD1_period,
        "PD1_burst" : PD1_burst,
        "PD1_hyperpolarization" : PD1_hyperpol,
        "LPPD1_delay" : LPPD1_delay,
        "LPPD1_interval" : LPPD1_interval,
        "PD1LP_delay" : PD1LP_delay,
        "PD1LP_interval" : PD1LP_interval}
            
    intervals_PD2_dict = {
        "LP_period" : LP_period,
        "LP_burst" : LP_burst,
        "LP_hyperpolarization" : LP_hyperpol,
        "PD2_period" : PD2_period,
        "PD2_burst" : PD2_burst,
        "PD2_hyperpolarization" : PD2_hyperpol,
        "LPPD2_delay" : LPPD2_delay,
        "LPPD2_interval" : LPPD2_interval,
        "PD2LP_delay" : PD2LP_delay,
        "PD2LP_interval" : PD2LP_interval}
    
    return intervals_both_dict, intervals_PD1_dict, intervals_PD2_dict



new_data = {}


for exp in df_data.keys():
    PD1_beg = df_data[exp]["PD1"]["beg"]
    PD1_end = df_data[exp]["PD1"]["end"]

    PD2_beg = df_data[exp]["PD2"]["beg"]
    PD2_end = df_data[exp]["PD2"]["end"]

    LP_beg = df_data[exp]["LP"]["beg"]
    LP_end = df_data[exp]["LP"]["end"]

    intervals_exp,_,_ = intervals_PD_reference(
        PD1_beg, PD1_end,
        PD2_beg, PD2_end,
        LP_beg, LP_end
    )

    new_data[exp] = intervals_exp   # store per experiment

new_df = pd.DataFrame([new_data])  # single-row dataframe
new_df.index = ["intervals"]




new_df.to_pickle("intervals_data.pkl")