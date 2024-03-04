from FoKL import FoKLRoutines
from FoKL.GP_Integrate import GP_Integrate
import numpy as np
import matplotlib.pyplot as plt


def main():

    # Inputs:
    traininputs = np.loadtxt('traininputs.txt', dtype=float, delimiter=',')
    traindata1 = np.loadtxt('traindata1.txt', dtype=float, delimiter=',')
    traindata2 = np.loadtxt('traindata2.txt', dtype=float, delimiter=',')
    traindata = [traindata1, traindata2]
    y = np.loadtxt('y.txt', dtype=float, delimiter=',')
    utest = np.loadtxt('utest.csv', dtype=float, delimiter=',')

    # User-defined hyperparameters (to override default values):
    relats_in = [1, 1, 1, 1, 1, 1]
    a = 1000
    b = 1
    draws = 2000
    way3 = True
    threshav = 0
    threshstda = 0
    threshstdb = 100

    # Initializing FoKL model with constant hypers:
    model = FoKLRoutines.FoKL(relats_in=relats_in, a=a, b=b, draws=draws, way3=way3, threshav=threshav,
                              threshstda=threshstda, threshstdb=threshstdb)

    # User-defined hyperparameters (to iterate through for different data or to sweep the same data):
    btau = [0.6091, 1]

    # Iterating through datasets:

    betas = []
    mtx = []
    for ii in range(2):

        print("\nCurrently fitting model to dataset", int(ii+1), "...")

        # Updating model with current iteration of variable hyperparameters:
        model.btau = btau[ii]

        # Running emulator routine for current model/data:
        betas_i, mtx_i, _ = model.fit(traininputs, traindata[ii])

        print("Done!")

        # Store values for post-processing (i.e., GP integration):
        betas.append(betas_i[1000:])
        mtx.append(mtx_i)

        # Clear all attributes (except for hypers) so previous results do not influence the next iteration:
        model.clear()

    # Integrating with FoKL.GP_Integrate():

    phis = model.phis # same for all models iterated through, so just grab value from most recent model

    n, m = np.shape(y)
    norms1 = [np.min(y[0, 0:int(m/2)]), np.max(y[0, 0:int(m/2)])]
    norms2 = [np.min(y[1, 0:int(m/2)]), np.max(y[1, 0:int(m/2)])]
    norms = np.transpose([norms1, norms2])

    start = 4
    stop = 3750*4
    stepsize = 4
    used_inputs = [[1, 1, 1], [1, 1, 1]]
    ic = y[:, int(m/2)-1]

    T, Y = GP_Integrate([np.mean(betas[0], axis=0), np.mean(betas[1], axis=0)], [mtx[0],mtx[1]], utest, norms, phis,
                        start, stop, ic, stepsize, used_inputs)

    plt.figure()
    plt.plot(T, Y[0], T, y[0][3750:7500])
    plt.plot(T, Y[1], T, y[1][3750:7500])
    plt.show()


if __name__ == '__main__':
    main()

