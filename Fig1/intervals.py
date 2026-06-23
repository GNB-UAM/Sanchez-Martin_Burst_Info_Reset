"""Copyright (c) 2026 Pablo Sanchez-Martin. All Rights Reserved.
Use of this source code is govern by GPL-3.0 license that 
can be found in the LICENSE file"""


import matplotlib.pyplot as plt
import numpy as np

file = "./16h03m23s-18-Feb-2016.txt"#PD is 3, LP intra is 4, Extra is 2

data = np.loadtxt(file)


PD = data[:, 3]
LP = data[:, 4]

plt.plot(PD[4200:15000], linewidth=1)
plt.plot(LP[4200:15000], linewidth=1)
plt.show()


