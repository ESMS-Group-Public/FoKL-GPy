import numpy as np
import itertools

def emulator(inputs, data, phis, relats_in, a, b, atau, btau, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic):
    """
    this version uses 3 way interactions use routines.emulator_Xin for two way interactions

    this version uses the 'Xin' mode of the gibbs sampler

    builds a single-output bss-anova emulator for a stationary dataset in an
    automated fashion

    function inputs:
    'sigsqd0' is the initial guess for the obs error variance

    'inputs' is the set of inputs normalized on [0,1]: matrix or numpy array
    with columns corresponding to inputs and rows the different experimental designs

    'data' are the output dataset used to build the function: column vector,
    with entries corresponding to rows of 'inputs'

    'relats' is a boolean matrix indicating which terms should be excluded
    from the model building; for instance if a certain main effect should be
    excluded relats will include a row with a 1 in the column for that input
    and zeros elsewhere; if a certain two way interaction should be excluded
    there should be a row with ones in those columns and zeros elsewhere
    to exclude no terms 'relats = np.array([[0]])'. An example of excluding
    the first input main effect and its interaction with the third input for
    a case with three total inputs is:'relats = np.array([[1,0,0],[1,0,1]])'

    'phis' are a data structure with the spline coefficients for the basis
    functions, built with 'spline_coefficient.txt' and 'splineconvert' or
    'spline_coefficient_500.txt' and 'splineconvert500' (the former provides
    25 basis functions: enough for most things -- while the latter provides
    500: definitely enough for anything)

    'a' and 'b' are the shape and scale parameters of the ig distribution for
    the observation error variance of the data. the observation error model is
    white noise choose the mode of the ig distribution to match the noise in
    the output dataset and the mean to broaden it some

    'atau' and 'btau' are the parameters of the ig distribution for the 'tau
    squared' parameter: the variance of the beta priors is iid normal mean
    zero with variance equal to sigma squared times tau squared. tau squared
    must be scaled in the prior such that the product of tau squared and sigma
    squared scales with the output dataset

    'tolerance' controls how hard the function builder tries to find a better
    model once adding terms starts to show diminishing returns. a good
    default is 3 -- large datasets could benefit from higher values

    'draws' is the total number of draws from the posterior for each tested
    model

    'draws' is the total number of draws from the posterior for each tested

    'gimmie' is a boolean causing the routine to return the most complex
    model tried instead of the model with the optimum bic

    'aic' is a boolean specifying the use of the aikaike information
    criterion

    function outputs:

    'betas' are a draw from the posterior distribution of coefficients: matrix,
    with rows corresponding to draws and columns corresponding to terms in the
    GP

    'mtx' is the basis function interaction matrix from the best model:
    matrix, with rows corresponding to terms in the GP (and thus to the
    columns of 'betas' and columns corresponding to inputs). A given entry in
    the matrix gives the order of the basis function appearing in a given term
    in the GP.
    All basis functions indicated on a given row are multiplied together.
    a zero indicates no basis function from a given input is present in a
    given term.

    'ev' is a vector of BIC values from all of the models evaluated
    """

    def perms(x):
        """Python equivalent of MATLAB perms."""
        # from https://stackoverflow.com/questions/38130008/python-equivalent-for-matlabs-perms
        a = np.vstack(list(itertools.permutations(x)))[::-1]

        return a

    # 'n' is the number of datapoints whereas 'm' is the number of inputs
    n, m = np.shape(inputs)
    mrel = n
    damtx = np.array([])
    evs = np.array([])

    # Conversion of Lines 79-100 of emulator_Xin.m
    if np.logical_not(all([isinstance(index, int) for index in relats_in])):  # checks relats to see if it's an array
        if np.any(relats_in):
            relats = np.zeros((sum(np.logical_not(relats_in)), m))
            ind = 1
            for i in range(0, m):
                if np.logical_not(relats_in[i]):
                    relats[ind][i] = 1
                    ind = ind + 1
            ind_in = m + 1
            for i in range(0, m - 1):
                for j in range(i + 1, m):
                    if np.logical_not(relats_in[ind_in]):
                        relats[ind][i] = 1
                        relats[ind][j] = 1
                        ind = ind + 1
                ind_in = ind_in + 1
        mrel = sum(np.logical_not(relats_in)).all()
    else:
        mrel = sum(np.logical_not(relats_in))
    # End conversion

    # 'ind' is an integer which controls the development of new terms
    ind = 1
    greater = 0
    finished = 0
    X = []
    killset = []
    killtest = []
    if m == 1:
        sett = 1
    elif way3:
        sett = 3
    else:
        sett = 2

    while True:
        # first we have to come up with all combinations of 'm' integers that
        # sums up to ind
        indvec = np.zeros((m))
        summ = ind

        while summ:
            for j in range(0,sett):
                indvec[j] = indvec[j] + 1
                summ = summ - 1
                if summ == 0:
                    break

        while 1:
            vecs = np.unique(perms(indvec),axis=0)
            if ind > 1:
                mvec, nvec = np.shape(vecs)
            else:
                mvec = np.shape(vecs)[0]
                nvec = 1
            killvecs = []
            if mrel != 0:
                for j in range(1, mvec):
                    testvec = np.divide(vecs[j, :], vecs[j, :])
                    testvec[np.isnan(testvec)] = 0
                    for k in range(1, mrel):
                        if sum(testvec == relats[k, :]) == m:
                            killvecs.append(j)
                            break
                nuvecs = np.zeros(mvec - np.size(killvecs), m)
                vecind = 1
                for j in range(1, mvec):
                    if not (j == killvecs):
                        nuvecs[vecind, :] = vecs[j, :]
                        vecind = vecind + 1

                vecs = nuvecs
            if ind > 1:
                vm, vn = np.shape(vecs)
            else:
                vm = np.shape(vecs)[0]
                vn = 1
            if np.size(damtx) == 0:
                damtx = vecs
            else:
                damtx = np.append(damtx, vecs, axis=0)
            [dam,null] = np.shape(damtx)

            [beters, null, null, null, xers, ev] = FokL.gibbs(inputs, data, phis, X, damtx, a, b, atau, btau, draws)

            if aic:
                ev = ev + (2 - np.log(n)) * (dam + 1)

            betavs = np.abs(np.mean(beters[int(np.ceil((draws / 2)+1)):draws, (dam - vm + 1):dam+1], axis=0))
            betavs2 = np.divide(np.std(np.array(beters[int(np.ceil(draws/2)+1):draws, dam-vm+1:dam+1]), axis=0), np.abs(np.mean(beters[int(np.ceil(draws / 2)):draws, dam-vm+1:dam+2], axis=0))) # betavs2 error in std deviation formatting
            betavs3 = np.array(range(dam-vm+2, dam+2))
            betavs = np.transpose(np.array([betavs,betavs2, betavs3]))
            if np.shape(betavs)[1] > 0:
                sortInds = np.argsort(betavs[:, 0])
                betavs = betavs[sortInds]

            killset = []
            evmin = ev

            for i in range(0, vm):
                if betavs[i, 1] > threshstdb or betavs[i, 1] > threshstda and betavs[i, 0] < threshav * np.mean(np.abs(np.mean(beters[int(np.ceil(draws/2 +1)):draws, 0]))):

                    killtest = np.append(killset, (betavs[i, 2] - 1))
                    damtx_test = damtx
                    count = 1
                    for k in range(0, np.size(killtest)):
                        damtx_test = np.delete(damtx_test, (int(np.array(killtest[k]))-count), 0)
                        count = count + 1
                    damtest, null = np.shape(damtx_test)

                    [betertest, null, null, null, Xtest, evtest] = FokL.gibbs(inputs, data, phis, X, damtx_test, a, b, atau, btau, draws)
                    if aic:
                        evtest = evtest + (2 - np.log(n))*(damtest+1)
                    if evtest < evmin:
                        killset = killtest
                        evmin = evtest
                        xers = Xtest
                        beters = betertest
            count = 1
            for k in range(0, np.size(killset)):
                damtx = np.delete(damtx, (int(np.array(killset[k])) - count), 0)
                count = count + 1

            ev = evmin
            X = xers

            print([ind, ev])
            if np.size(evs) > 0:
                if ev < np.min(evs):

                    betas = beters
                    mtx = damtx
                    greater = 1
                    evs = np.append(evs, ev)

                elif greater < tolerance:
                    greater = greater + 1
                    evs = np.append(evs, ev)
                else:
                    finished = 1
                    evs = np.append(evs, ev)

                    break
            else:
                greater = greater + 1
                evs = np.append(evs, ev)
            if m == 1:
                break
            elif way3:
                if indvec[1] > indvec[2]:
                    indvec[0] = indvec[0] + 1
                    indvec[1] = indvec[1] - 1
                elif indvec[2]:
                    indvec[1] = indvec[1] + 1
                    indvec[2] = indvec[2] - 1
                    if indvec[1] > indvec[0]:
                        indvec[0] = indvec[0] + 1
                        indvec[1] = indvec[1] - 1
                else:
                    break
            elif indvec[1]:
                indvec[0] = indvec[0] + 1
                indvec[1] = indvec[1] - 1
            else:
                break

        if finished != 0:
            break

        ind = ind + 1

        if ind > len(phis):
            break

    # Implementation of 'gimme' feature
    if gimmie:
        betas = beters
        mtx = damtx

    return betas, mtx, evs
