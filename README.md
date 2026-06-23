# Sanchez-Martin_Burst_Info_Reset
Code and datasets generated in Sanchez-Martin et al. publication titled "Burst-to-burst information resetting in sequential rhythmic neural activity".
This code includes all data analysis and figure plotting.

If you use this code please cite: Sanchez-Martin, P., Elices, I., Garrido-Peña, A., Garcia-Saura, C., Levi, R., Rodriguez, F.B., Varona, P. (2026). *Burst-to-burst information resetting in sequential rhythmic neural activity* [Manuscript submitted for publication].

## How to use
Load and activate conda environment from environment.yml file with the following commands:

	conda env create -f environment.yml
	conda activate info_reset

### Experimental analysis
The detection of spikes and bursts is stored in spikes_data.pkl while intervals_data.pkl has the calculation of intervals considering those burst references (first and last spike of each neuron). All the scripts use the intervals_data.pkl file in the main folder.

To recreate each figure run the corresponding script in the Figures folder like:

	python3 invariants.py
	

### Computational analysis



## License
Source code (python files): Licensed under the GNU GPL v3.0. See the LICENSE file for details.
Generated data (Data file .pkl and images): Licensed under the Creative Commons Attribution 4.0 International License (CC BY-NC-SA 4.0). For the full legal text, see: https://creativecommons.org/licenses/by/4.0/deed.en
