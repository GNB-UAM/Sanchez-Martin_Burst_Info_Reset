"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""


import matplotlib.pyplot as plt
import numpy as np

file = "./intervals_two_cycles.csv"

data = np.loadtxt(file)


PD_trace = data[:, 0]
LP_trace = data[:, 1]

plt.plot(PD_trace, linewidth=1)
plt.plot(LP_trace, linewidth=1)
plt.show()


