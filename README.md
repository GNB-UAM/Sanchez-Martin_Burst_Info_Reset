# Sanchez-Martin_Burst_Info_Reset
Code and datasets generated in Sanchez-Martin et al. publication titled "Burst-to-burst information resetting in sequential rhythmic neural activity".
This code includes all data analysis and figure plotting.

If you use this code please cite: Sanchez-Martin, P., Elices, I., Garrido-Peña, A., Garcia-Saura, C., Levi, R., Rodriguez, F.B., Varona, P. (2026). *Burst-to-burst information resetting in sequential rhythmic neural activity* [Manuscript submitted for publication].

## How to use
Load and activate conda environment from environment.yml file with the following commands:

	conda env create -f environment.yml
	conda activate info_reset

### Experimental analysis
The detection of spikes and bursts (first and last spike of each neurons) is stored in spikes_data.pkl while intervals_data.pkl has the calculation of intervals considering those burst references (run "calculate_intervals.py" to recreate those intervals from the spikes). All the scripts use the intervals_data.pkl file in the main folder.

To recreate each figure run the corresponding script in the Figures folder like:

	python3 invariants.py
	

### Computational analysis

#### Installation

```bash
# 1. Clone the pyloric simulator
git clone https://github.com/mackelab/pyloric.git
cd pyloric

# Install pyximport requires gcc
sudo apt-get install gcc python3-dev   # Ubuntu/WSL
conda install cython                   # if using Anaconda

# 2. Install in editable mode (compiles the Cython solver)
pip install -e .

# 3. Replace interface.py with the patched version from this repository
cp path/to/this/repo/interface.py pyloric/interface.py
```

#### Usage

To reproduce the figures from the paper, follow these steps.

##### Repository structure

```
.
├── environment.yaml
├── README.md
├── experimental_analysis/
|	├── calculate_intervals.py
│   ├── intervals_data.pkl
│   ├── spikes_data.pkl
│   ├── Fig1/
│   │   └── intervals_two_cycles.csv
│   │   └── intervals.py
│   │   └── invariants.py
│   ├── Fig2/
│   │   └── pairplots.py
│   ├── Fig3/
│   │   └── trend.py
│   ├── Fig4/
│   │   └── exp_trends.py
│   └── Fig5/
│       └── auto_segment.py
└── pyloric_model/
 	├── interface.py
    ├── run_pyloric.py
    ├── pyloric_data.py
    ├── pyloric_iext.py
    ├── pyloric_currents.py
    ├── close_to_xo_circuit_parameters_min_burst_condition_078.pkl
    └── (generated .h5 / .pkl files)
    ├── Fig6/
    │    └── pyloric_plot_voltage.py
    ├── Fig7/
    │    └── pyloric_plot_pairplots.py
    ├── Fig8/
    │    └── pyloric_plot_r2_shift.py
    └── Fig9/
        └── pyloric_plot_currents.py
```

##### 1. Generate the voltage and current data (.h5 files)

Run `run_pyloric.py` from `pyloric_model/`, once per condition:

**Without modulation**:
```python
# --- No current ---
I = None
I_label = 'noIext'
```

**With modulation**:
```python
# --- Ramp on AB/PD: 100 -> -15 pA over 130 s ---
I = make_I_ext(t_max, dt, neuron='AB/PD', kind='ramp',
                v_start=0.00010, v_end=-0.000015, t_start=10000, t_end=140000)
I_label = 'ramp_ABPD_amp0.00010-desc-15_dur140s'
```

```bash
cd pyloric_model
python3 run_pyloric.py
```

##### 2. Detect intervals and generate the .pkl file

Run `pyloric_data.py` on each voltage file:

```bash
# Without modulation
python3 pyloric_data.py simulation_circuit0_noIext_voltages.h5 --t-start 10000 --t-end 140000

# With modulation
python3 pyloric_data.py simulation_circuit0_ramp_ABPD_amp0.00010-desc-15_dur140s_voltages.h5 --t-start 10000 --t-end 140000
```

##### 3. Generate the figures

Each figure script lives in its own `Fig*/` folder and reads the `.pkl` and `h5` file(s) from `pyloric_model/` (one level up):

```bash
# Fig 6
cd Fig6
python3 pyloric_plot_voltage.py ../pyloric_model/simulation_circuit0_ramp_ABPD_amp0.00010-desc-15_dur140s_data.pkl

# Fig 7
cd ../Fig7
python3 pyloric_plot_pairplots.py \
    --pkl-mod   ../pyloric_model/simulation_circuit0_ramp_ABPD_amp0.00010-desc-15_dur140s_data.pkl \
    --pkl-nomod ../pyloric_model/simulation_circuit0_noIext_data.pkl

# Fig 8
cd ../Fig8
python3 pyloric_plot_r2_shift.py \
    --pkl-mod   ../pyloric_model/simulation_circuit0_ramp_ABPD_amp0.00010-desc-15_dur140s_data.pkl \
    --pkl-nomod ../pyloric_model/simulation_circuit0_noIext_data.pkl

# Fig 9
cd ../Fig9
python3 pyloric_plot_currents.py
```

> `pyloric_plot_currents.py` defaults to reading
> `../pyloric_model/simulation_circuit0_ramp_ABPD_amp0.00010-desc-15_dur140s_data.pkl`
> (the modulated condition), so no argument is needed if the file is in that
> location. Pass a path explicitly to use a different `.pkl`.


## License
Source code (python files): Licensed under the GNU GPL v3.0. See the LICENSE file for details.
Generated data (Data file .pkl and images): Licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0). For the full legal text, see: https://creativecommons.org/licenses/by/4.0/deed.en
