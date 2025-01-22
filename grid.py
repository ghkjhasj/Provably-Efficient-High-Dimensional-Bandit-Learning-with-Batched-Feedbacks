import numpy as np
import math
import matplotlib.pyplot as plt


T_set = [1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000]
L_set = np.zeros(8)
for i in range(8):
    T = T_set[i]
    a = np.square(np.log(T))/5
    grid = [2]  # t_1
    for l in range(1, T + 1):
        val0 = grid[l - 1]
        val = (a / l / np.log(val0) + 1) * val0
        val = min(math.floor(val), T)
        grid.append(val)
        if val == T:
            break
    L_set[i] = l

plt.plot(T_set, L_set)
plt.show()
