import numpy as np
import math
import matplotlib.pyplot as plt

def coverage3(betas, normputs, data, phis, mtx, draws, plots):
    """
        Inputs:
            Interprets outputs of emulator

            betas - betas emulator output

            normputs - normalized inputs

            phis - from spline convert

            mtx - interaction matrix from emulator

            draws - number of beta terms used

            plots - binary for plot output

        returns:
            Meen: Predicted values for each indexed input

            RSME: root mean squared deviation

            Bounds: confidence interval, dotted lines on plot, larger bounds means more uncertainty at location


       """
    m, mbets = np.shape(betas)  # Size of betas
    n, mputs = np.shape(normputs)  # Size of normalized inputs

    setnos_p = np.random.randint(m, size=(1, draws))  # Random draws  from integer distribution
    i = 1
    while i == 1:
        setnos = np.unique(setnos_p)

        if np.size(setnos) == np.size(setnos_p):
            i = 0
        else:
            setnos_p = np.append(setnos, np.random.randint(m, size=(1, draws - np.shape(setnos)[0])))

    X = np.zeros((n, mbets))
    normputs = np.asarray(normputs)
    for i in range(n):
        phind = []  # Rounded down point of input from 0-499
        for j in range(len(normputs[i])):
            phind.append(math.floor(normputs[i, j] * 498))
            # 499 changed to 498 for python indexing

        phind_logic = []
        for k in range(len(phind)):
            if phind[k] == 498:
                phind_logic.append(1)
            else:
                phind_logic.append(0)

        phind = np.subtract(phind, phind_logic)

        for j in range(1, mbets):
            phi = 1
            for k in range(mputs):
                num = mtx[j - 1, k]
                if num > 0:
                    xsm = 498 * normputs[i][k] - phind[k]
                    phi = phi * (phis[int(num) - 1][0][phind[k]] + phis[int(num) - 1][1][phind[k]] * xsm +
                                 phis[int(num) - 1][2][phind[k]] * xsm ** 2 + phis[int(num) - 1][3][
                                     phind[k]] * xsm ** 3)
            X[i, j] = phi

    X[:, 0] = np.ones((n,))
    modells = np.zeros((np.shape(data)[0], draws))
    for i in range(draws):
        modells[:, i] = np.matmul(X, betas[setnos[i], :])
    meen = np.mean(modells, 1)
    bounds = np.zeros((np.shape(data)[0], 2))
    cut = int(np.floor(draws * .025))
    for i in range(np.shape(data)[0]):
        drawset = np.sort(modells[i, :])
        bounds[i, 0] = drawset[cut]
        bounds[i, 1] = drawset[draws - cut]

    if plots:
        plt.plot(meen, 'b', linewidth=2)
        plt.plot(bounds[:, 0], 'k--')
        plt.plot(bounds[:, 1], 'k--')

        plt.plot(data, 'ro')

        plt.show()

    rmse = np.sqrt(np.mean(meen - data) ** 2)
    return meen, bounds, rmse
