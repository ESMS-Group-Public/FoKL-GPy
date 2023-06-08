import numpy as np
from scipy.linalg import eigh
import math
from numpy import linalg as LA

def gibbs(inputs, data, phis, Xin, discmtx, a, b, atau, btau,
          draws):
    """
    'sigsqd0' is the initial guess for the obs error variance

    'inputs' is the set of normalized inputs -- both parameters and model
    inputs -- with columns corresponding to inputs and rows the different
    experimental designs

    'data' are the experimental results: column vector, with entries
    corresponding to rows of 'inputs'

    'phis' are a data structure with the spline coefficients for the basis
    functions, built with 'BasisSpline.txt' and 'splineloader' or
    'splineconvert'

    'discmtx' is the interaction matrix for the bss-anova function -- rows
    are terms in the function and columns are inputs (cols should line up
    with cols in 'inputs'

    'a' and 'b' are the parameters of the ig distribution for the
    observation error variance of the data

    'atau' and 'btau' are the parameters of the ig distribution for the 'tau
    squared' parameter: the variance of the beta priors

    'draws' is the total number of draws
    """

    # building the matrix by calculating the corresponding basis function
    # outputs for each set of inputs
    minp, ninp = np.shape(inputs)
    phi_vec = []
    if np.shape(discmtx) == ():  # part of fix for single input model
        mmtx = 1
    else:
        mmtx, null = np.shape(discmtx)

    if Xin == []:
        Xin = np.ones((minp, 1))
        mxin, nxin = np.shape(Xin)
    else:
        # X = Xin
        mxin, nxin = np.shape(Xin)
    if mmtx - nxin < 0:
        X = Xin
    else:
        X = np.append(Xin, np.zeros((minp, mmtx - nxin)), axis=1)

    for i in range(minp):

        phind = []
        for j in range(len(inputs[i])):
            phind.append(math.ceil(inputs[i][j] * 498))
            # 499 changed to 498 for python indexing

        phind_logic = []
        for k in range(len(phind)):
            if phind[k] == 0:
                phind_logic.append(1)
            else:
                phind_logic.append(0)

        phind = np.add(phind, phind_logic)

        for j in range(nxin, mmtx + 1):
            null, nxin2 = np.shape(X)
            if j == nxin2:
                X = np.append(X, np.zeros((minp, 1)), axis=1)

            phi = 1

            for k in range(ninp):

                if np.shape(discmtx) == ():
                    num = discmtx
                else:
                    num = discmtx[j - 1][k]

                if num != 0:  # enter if loop if num is nonzero
                    xsm = 498 * inputs[i][k] - phind[k]
                    phi = phi * (phis[int(num) - 1][0][phind[k]] + phis[int(num) - 1][1][phind[k]] * xsm +
                                 phis[int(num) - 1][2][phind[k]] * xsm ** 2 + phis[int(num) - 1][3][phind[k]] *
                                 xsm ** 3)
                    ppp = 1

            X[i][j] = phi

    # initialize tausqd at the mode of its prior: the inverse of the mode of
    # sigma squared, such that the initial variance for the betas is 1
    sigsqd = b / (1 + a)
    tausqd = btau / (1 + atau)

    XtX = np.transpose(X).dot(X)

    Xty = np.transpose(X).dot(data)

    # See the link https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
    Lamb, Q = eigh(XtX)  # using scipy eigh function to avoid generation of imaginary values due to numerical errors
    # Lamb, Q = LA.eig(XtX)

    Lamb_inv = np.diag(1 / Lamb)

    betahat = Q.dot(Lamb_inv).dot(np.transpose(Q)).dot(Xty)
    squerr = LA.norm(data - X.dot(betahat)) ** 2

    astar = a + 1 + len(data) / 2 + (mmtx + 1) / 2
    atau_star = atau + mmtx / 2

    dtd = np.transpose(data).dot(data)

    # Gibbs iterations

    betas = np.zeros((draws, mmtx + 1))
    sigs = np.zeros((draws, 1))
    taus = np.zeros((draws, 1))

    lik = np.zeros((draws, 1))
    n = len(data)

    for k in range(draws):

        Lamb_tausqd = np.diag(Lamb) + (1 / tausqd) * np.identity(mmtx + 1)
        Lamb_tausqd_inv = np.diag(1 / np.diag(Lamb_tausqd))

        mun = Q.dot(Lamb_tausqd_inv).dot(np.transpose(Q)).dot(Xty)
        S = Q.dot(np.diag(np.diag(Lamb_tausqd_inv) ** (1 / 2)))

        vec = np.random.normal(loc=0, scale=1, size=(mmtx + 1, 1))  # drawing from normal distribution
        betas[k][:] = np.transpose(mun + sigsqd ** (1 / 2) * (S).dot(vec))

        vecc = mun - np.reshape(betas[k][:], (len(betas[k][:]), 1))


        bstar = b + 0.5 * (
                    betas[k][:].dot(XtX.dot(np.transpose([betas[k][:]]))) - 2 * betas[k][:].dot(Xty) + dtd + betas[
                                                                                                                 k][
                                                                                                             :].dot(
                np.transpose([betas[k][:]])) / tausqd)
        # bstar = b + comp1.dot(comp2) + 0.5 * dtd - comp3;

        # Returning a 'not a number' constant if bstar is negative, which would
        # cause np.random.gamma to return a ValueError
        if bstar < 0:
            sigsqd = math.nan
        else:
            sigsqd = 1 / np.random.gamma(astar, 1 / bstar)

        sigs[k] = sigsqd

        btau_star = (1 / (2 * sigsqd)) * (betas[k][:].dot(np.reshape(betas[k][:], (len(betas[k][:]), 1)))) + btau

        tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
        taus[k] = tausqd

        # Calculate the evidence
    siglik = np.var(data - np.matmul(X, betahat))

    lik = -(n / 2) * np.log(siglik) - (n - 1) / 2
    ev = (mmtx + 1) * np.log(n) - 2 * np.max(lik)

    X = X[:, 0:mmtx + 1]

    return betas, sigs, taus, betahat, X, ev
