from src.FoKL import FoKLRoutines
import numpy as np


def main():

    # Dataset:
    res = int(1e2)
    x0 = np.exp(np.linspace(0, 1, res))
    x1 = (np.random.rand(res) - 0.5) * 0.2
    x2 = np.sin(res) * 2
    y = x0 * x1 + x2

    # Hyperparameters to sweep through:
    a = 2 ** np.linspace(1, 10, 10)  # = [2, 4, 8, ..., 1024]
    atau = [2, 4, 6]

    # Sweep:
    f = FoKLRoutines.FoKL(kernel='Bernoulli Polynomials')  # = ...(kernel=1)
    f.clean([x0, x1, x2], y, train=0.8)
    rmse = np.zeros(len(a), len(atau))
    for i in range(len(a)):
        for j in range(len(atau)):
            f.fit(f.traininputs, f.traindata, clean=False, a=a[i], atau=atau[j])
            _, _, rmse[i, j] = f.coverage3(inputs=f.traininputs, data=f.traindata)

    # Determine best results from sweep:
    ij = np.argmin(rmse)

    optimal_a = a[ij[0]]
    optimal_atau = atau[ij[1]]

    print("The optimal hyperparameters found in the sweep are:")
    print(f"    a    = {optimal_a}")
    print(f"    atau = {optimal_atau}")
    print()
    print("Complete!")


if __name__ == '__main__':
    main()

