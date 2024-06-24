"""
[ExAMPLE]: GP Integrate

This is an example using the 'GP_Integrate' function with a FoKL model.
"""
from FoKL import FoKLRoutines
from FoKL.GP_Integrate import GP_Integrate
import os
dir = os.path.abspath(os.path.dirname(__file__))  # directory of script
# # -----------------------------------------------------------------------
# # UNCOMMENT IF USING LOCAL FOKL PACKAGE:
# import sys
# sys.path.append(os.path.join(dir, '..', '..'))  # package directory
# from src.FoKL import FoKLRoutines
# from src.FoKL.GP_Integrate import GP_Integrate
# # -----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Known dataset:
    traininputs = np.loadtxt(os.path.join(dir, 'traininputs.txt'), dtype=float, delimiter=',')  # input variables for both datasets
    traindata = [np.loadtxt(os.path.join(dir, 'traindata1.txt'), dtype=float, delimiter=','),   # data for first and second datasets
                 np.loadtxt(os.path.join(dir, 'traindata2.txt'), dtype=float, delimiter=',')]
    y = np.loadtxt(os.path.join(dir, 'y.txt'), dtype=float, delimiter=',')
    utest = np.loadtxt(os.path.join(dir, 'utest.csv'), dtype=float, delimiter=',')

    # Initializing FoKL model with user-defined constant hyperparameters (to override default values) and turning off
    # user-warnings (i.e., warnings from FoKL) since working example requires no troubleshooting:
    model = FoKLRoutines.FoKL(relats_in=[1, 1, 1, 1, 1, 1], a=1000, b=1, draws=2000, way3=True, threshav=0,
                              threshstda=0, threshstdb=100, UserWarnings=False)

    # Iterating through datasets:
    btau = [0.6091, 1]  # user-defined variable hyperparameter to iterate through for different datasets
    betas = []
    betas_avg = []
    mtx = []
    for i in range(2):
        print(f"\nCurrently training model on dataset {int(i + 1)}...\n")

        # Updating model with current iteration of variable hyperparameters:
        model.btau = btau[i]

        # Running emulator routine for current dataset:
        betas_i, mtx_i, _ = model.fit(traininputs, traindata[i], clean=True)
        print("\nDone!")

        # Store coefficients and interaction matrix of model equation for post-processing (i.e., for 'GP_Integrate'):
        betas.append(betas_i[1000:])  # store only last 1000 such that draws minus 1000 serves as burn-in
        betas_avg.append(np.mean(betas_i, axis=0))
        mtx.append(mtx_i)

        # Clear all attributes (except for hypers) so previous results do not influence the next iteration:
        model.clear()

    # ------------------------------------------------------------------------------------------------------------------

    # Integrating with 'GP_Integrate()':

    n, m = np.shape(y)
    norms1 = [np.min(y[0, 0:int(m / 2)]), np.max(y[0, 0:int(m / 2)])]
    norms2 = [np.min(y[1, 0:int(m / 2)]), np.max(y[1, 0:int(m / 2)])]
    norms = np.transpose([norms1, norms2])

    start = 4
    stop = 3750 * 4
    stepsize = 4
    used_inputs = [[1, 1, 1], [1, 1, 1]]
    ic = y[:, int(m / 2) - 1]

    t, yt = GP_Integrate(betas_avg, [mtx[0], mtx[1]], utest, norms, model.phis, start, stop, ic, stepsize, used_inputs)

    plt.figure()
    plt.plot(t, yt[0], t, y[0][3750:7500])
    plt.plot(t, yt[1], t, y[1][3750:7500])
    plt.show()


if __name__ == '__main__':
    main()
    print("\nEnd of GP Integrate example.")

