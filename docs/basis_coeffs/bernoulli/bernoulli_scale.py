from src.FoKL import FoKLRoutines
import numpy as np
import matplotlib.pyplot as plt


res = int(500)
rows = 4
cols = 6

n = int(rows * cols)
x = np.linspace(0, 1, res)
y = np.zeros([res, n])
f = FoKLRoutines.FoKL(kernel=1)
for n in range(n):
    c = f.phis[n]
    for i in range(res):
        y[i, n] = f.evaluate_basis(c, x[i])

    # plt.cla()
    # plt.plot(x, y[:, n])
    # plt.show()

plt.cla()
plt.figure()
fig, ax = plt.subplots(rows, cols)
# fig.suptitle("Orthonormal Bernoulli Polynomials, Scaled by sqrt(eigenvalues) from 500x500 K")
fig.suptitle("Orthonormal Bernoulli Polynomials, unscaled")
ni = 0
for i in range(rows):
    for j in range(cols):
        ax[i, j].plot(x, y[:, ni])
        ni += 1
plt.show()


breakpoint()






