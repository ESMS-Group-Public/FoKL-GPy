import warnings
import math
import itertools
import sys
import numpy as np
from scipy.linalg import eigh
from numpy import linalg as LA
from ..utils import _process_kwargs, _str_to_bool
from src.FoKL.samplers.set_sampler import samplers


class fitSampler:
    def __init__(self, fokl, config, dataFormat, functions):
        self.fokl = fokl
        self.config = config
        self.ConsoleOutput = self.config.DEFAULT['ConsoleOutput']
        self.dataFormat = dataFormat
        self.functions = functions
        self.sampler = samplers(self.fokl, self.config, self.functions)

    def fit(self, inputs=None, data=None, sampler = 'gibbs', **kwargs):
        """
        For fitting model to known inputs and data (i.e., training of model).
        Inputs:
            inputs == NxM matrix of independent (or non-linearly dependent) 'x' variables for fitting f(x1, ..., xM)
            data   == Nx1 vector of dependent variable to create model for predicting the value of f(x1, ..., xM)
        Keyword Inputs (for fit):
            clean         == boolean to perform automatic cleaning and formatting               == False (default)
            ConsoleOutput == boolean to print [ind, ev] to console during FoKL model generation == True (default)
        See 'clean' for additional keyword inputs, which may be entered here.
        Return Outputs:
            'betas' are a draw (after burn-in) from the posterior distribution of coefficients: matrix, with rows
            corresponding to draws and columns corresponding to terms in the GP.
            'mtx' is the basis function interaction matrix from the
            best model: matrix, with rows corresponding to terms in the GP (and thus to the
            columns of 'betas' and columns corresponding to inputs. a given entry in the
            matrix gives the order of the basis function appearing in a given term in the GP.
            all basis functions indicated on a given row are multiplied together.
            a zero indicates no basis function from a given input is present in a given term
            'ev' is a vector of BIC values from all of the models
            evaluated
        Added Attributes:
            - Various ... please see description of 'clean()'
        """
        # Check for unexpected keyword arguments:
        default_for_fit = {'ConsoleOutput': True}
        default_for_fit['ConsoleOutput'] = _str_to_bool(kwargs.get('ConsoleOutput', self.ConsoleOutput))
        default_for_fit['clean'] = _str_to_bool(kwargs.get('clean', False))
        default_for_clean = {'train': 1, 
                             # For '_format':
                             'AutoTranspose': True, 'SingleInstance': False, 'bit': 64,
                             # For '_normalize':
                             'normalize': True, 'minmax': None, 'pillow': None, 'pillow_type': 'percent'}
        expected = self.config.HYPERS + list(default_for_fit.keys()) + list(default_for_clean.keys()) + self.config.samplers
        kwargs = _process_kwargs(expected, kwargs)
        if default_for_fit['clean'] is False:
            if any(kwarg in default_for_clean.keys() for kwarg in kwargs.keys()):
                warnings.warn("Keywords for automatic cleaning were defined but clean=False.")
            default_for_clean = {}  # not needed for future since 'clean=False'
        # Process keyword arguments and update/define class attributes:
        kwargs_to_clean = {}
        for kwarg in kwargs.keys():
            if kwarg in self.config.HYPERS:  # for case of user sweeping through hyperparameters within 'fit' argument
                if kwarg in ['gimmie', 'way3', 'aic']:
                    setattr(self, kwarg, _str_to_bool(kwargs[kwarg]))
                else:
                    setattr(self, kwarg, kwargs[kwarg])
            elif kwarg in default_for_clean.keys():
                # if kwarg in ['']:
                #     kwargs_to_clean.update({kwarg: _str_to_bool(kwargs[kwarg])})
                # else:
                kwargs_to_clean.update({kwarg: kwargs[kwarg]})
        self.ConsoleOutput = default_for_fit['ConsoleOutput']
        # Perform automatic cleaning of 'inputs' and 'data' (unless user specified not to), and handle exceptions:
        error_clean_failed = False
       # if default_for_fit['clean'] is True:
        if default_for_fit['clean'] is True:
            try:
                if inputs is None:  # assume clean already called and len(data) same as train data if data not None
                    inputs, _ = self.dataFormat.trainset()
                if data is None:  # assume clean already called and len(inputs) same as train inputs if inputs not None
                    _, data = self.dataFormat.trainset()
            except Exception as exception:
                error_clean_failed = True
            inputs, data, minmax = self.dataFormat.clean(inputs, data, kwargs_from_other=kwargs_to_clean, _setattr=True)
        else:  # user input implies that they already called clean prior to calling fit
            try:
                if inputs is None:  # assume clean already called and len(data) same as train data if data not None
                    inputs, _ = self.dataFormat.trainset()
                if data is None:  # assume clean already called and len(inputs) same as train inputs if inputs not None
                    _, data = self.dataFormat.trainset()
            except Exception as exception:
                warnings.warn("Keyword 'clean' was set to False but is required prior to or during 'fit'. Assuming "
                              "'clean' is True.", category=UserWarning)
                if inputs is None or data is None:
                    error_clean_failed = True
                else:
                    default_for_fit['clean'] = True
                    inputs, data, minmax = self.dataFormat.clean(inputs, data, kwargs_from_other=kwargs_to_clean, _setattr=True)
        if error_clean_failed is True:
            raise ValueError("'inputs' and/or 'data' were not provided so 'clean' could not be performed.")
        # After cleaning and/or handling exceptions, define cleaned 'inputs' and 'data' as local variables:
        try:
            inputs, data = self.dataFormat.trainset()
        except Exception as exception:
            warnings.warn("If not calling 'clean' prior to 'fit' or within the argument of 'fit', then this is the "
                          "likely source of any subsequent errors. To troubleshoot, simply include 'clean=True' within "
                          "the argument of 'fit'.", category=UserWarning)
            
        
        # Define attributes as local variables:
        phis = self.config.DEFAULT['phis']
        relats_in = self.config.DEFAULT['relats_in']
        a = self.config.DEFAULT['a']
        b = self.config.DEFAULT['b']
        atau = self.config.DEFAULT['atau']
        btau = self.config.DEFAULT['btau']
        tolerance = self.config.DEFAULT['tolerance']
        draws = self.config.DEFAULT['draws'] + self.config.DEFAULT['burnin']  # after fitting, the 'burnin' draws will be discarded from 'betas'
        gimmie = self.config.DEFAULT['gimmie']
        way3 = self.config.DEFAULT['way3']
        threshav = self.config.DEFAULT['threshav']
        threshstda = self.config.DEFAULT['threshstda']
        threshstdb = self.config.DEFAULT['threshstdb']
        aic = self.config.DEFAULT['aic']
        self.inputs = inputs  # update class attribute to be the cleaned 'inputs' (if not already done)
        self.data = data



        # Update 'b' and/or 'btau' if set to default:
        if btau is None or b is None:  # then use 'data' to define (in combination with 'a' and/or 'atau')
            # Calculate variance and mean, both as 64-bit, but for large datasets (i.e., less than 64-bit) be careful
            # to avoid converting the entire 'data' to 64-bit:
            if data.dtype != np.float64:  # and sigmasq == math.inf  # then recalculate but as 64-bit
                n = data.shape[0]
                data_mean = 0
                for i in range(n):  # element-wise to avoid memory errors when entire 'data' is 64-bit
                    data_mean += np.array(data[i], dtype=np.float64)
                data_mean = data_mean / n
                for i in range(n):  # element-wise to avoid memory errors when entire 'data' is 64-bit
                    sigmasq += (np.array(data[i], dtype=np.float64) - data_mean) ** 2
                sigmasq = sigmasq / (n - 1)
            else:  # same as above but simplified syntax avoiding for loops since 'data.dtype=np.float64'
                sigmasq = np.var(data)
                data_mean = np.mean(data)
            if sigmasq == math.inf:
                warnings.warn("The dataset is too large such that 'sigmasq=inf' even as 64-bit. Consider training on a "
                              "smaller percentage of the dataset.", category=UserWarning)
            if b is None:
                b = sigmasq * (a + 1)
                self.b = b
            if btau is None:
                scale = np.abs(data_mean)
                btau = (scale / sigmasq) * (atau + 1)
                self.btau = btau
        def perms(x):
            """Python equivalent of MATLAB perms."""
            # from https://stackoverflow.com/questions/38130008/python-equivalent-for-matlabs-perms
            a = np.vstack(list(itertools.permutations(x)))[::-1]
            return a
        # Prepare phind and xsm if using cubic splines, else match variable names required for gibbs argument
        if self.config.KERNELS[0] == self.config.DEFAULT['kernel']:  # == 'Cubic Splines':
            _, phind, xsm = self.dataFormat._inputs_to_phind(inputs)  # ..., phis=self.phis, kernel=self.kernel) already true
        elif self.config.KERNELS[1] == self.config.DEFAULT['kernel']:  # == 'Bernoulli Polynomials':
            phind = None
            xsm = inputs
        # [BEGIN] initialization of constants (for use in gibbs to avoid repeat large calculations):
        if self.config.DEFAULT['update'] == True:
            self.betas, self.mtx, self.evs = self.fit(inputs, data)
            return self.betas, self.mtx, self.evs
        # initialize tausqd at the mode of its prior: the inverse of the mode of sigma squared, such that the
        # initial variance for the betas is 1
        sigsqd0 = b / (1 + a)
        tausqd0 = btau / (1 + atau)
        dtd = np.transpose(data).dot(data)
        # Check for large datasets, where 'dtd=inf' is common and causes bug 'betas=nan', by only converting one
        # point to 64-bit at a time since there is likely not enough memory to convert all of 'data' to 64-bit:
        if dtd[0][0] == math.inf and data.dtype != np.float64:
            # # If converting all of 'data' to 64-bit:
            # data64 = np.array(data, dtype=np.float64)  # assuming large dataset means using less than 64-bit
            # dtd = np.dot(data64.T, data64)  # same equation, but now 64-bit
            # Converting one point at a time to limit memory use:
            dtd = 0
            for i in range(data.shape[0]):
                data_i = np.array(data[i], dtype=np.float64)
                dtd += data_i ** 2  # manually calculated inner dot product
            dtd = np.array([dtd])  # to align with dimensions of 'np.transpose(data).dot(data)' such that dtd[0][0]
        if dtd[0][0] == math.inf:
            warnings.warn("The dataset is too large such that the inner product of the output 'data' vector is "
                          "Inf. This will likely cause values in 'betas' to be Nan.", category=UserWarning)
        # [END] initialization of constants

        # 'n' is the number of datapoints whereas 'm' is the number of inputs
        n, m = np.shape(inputs)
        mrel = n
        damtx = np.array([])
        evs = np.array([])
        # Conversion of Lines 79-100 of emulator_Xin.m
        if np.logical_not(all([isinstance(index, int) for index in relats_in])):  # checks if relats is an array
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
            while True:
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

                [dam, null] = np.shape(damtx)
                [beters, null, null, null, xers, ev] = self.sampler.run_sampler(sampler, inputs, data, phis, X, damtx, a, b, atau, btau, draws, phind, xsm, sigsqd0, tausqd0, dtd)
                if aic:
                    ev = ev + (2 - np.log(n)) * (dam + 1)
                betavs = np.abs(np.mean(beters[int(np.ceil((draws / 2)+1)):draws, (dam - vm + 1):dam+1], axis=0))
                betavs2 = np.divide(np.std(np.array(beters[int(np.ceil(draws/2)+1):draws, dam-vm+1:dam+1]), axis=0),
                    np.abs(np.mean(beters[int(np.ceil(draws / 2)):draws, dam-vm+1:dam+2], axis=0)))
                    # betavs2 error in std deviation formatting
                betavs3 = np.array(range(dam-vm+2, dam+2))
                betavs = np.transpose(np.array([betavs,betavs2, betavs3]))
                if np.shape(betavs)[1] > 0:
                    sortInds = np.argsort(betavs[:, 0])
                    betavs = betavs[sortInds]
                killset = []
                evmin = ev
                for i in range(0, vm):
                    if betavs[i, 1] > threshstdb or betavs[i, 1] > threshstda and betavs[i, 0] < threshav * \
                            np.mean(np.abs(np.mean(beters[int(np.ceil(draws/2)):draws, 0]))):  # index to 'beters'
                        killtest = np.append(killset, (betavs[i, 2] - 1))
                        if killtest.size > 1:
                            killtest[::-1].sort()  # max to min so damtx_test rows get deleted in order of end to start
                        damtx_test = damtx
                        for k in range(0, np.size(killtest)):
                            damtx_test = np.delete(damtx_test, int(np.array(killtest[k])-1), 0)
                        damtest, null = np.shape(damtx_test)
                        [betertest, null, null, null, Xtest, evtest] = self.sampler.run_sampler(sampler, inputs, data, phis, X, damtx_test, a, b,
                                                                             atau, btau, draws, phind, xsm, sigsqd0,
                                                                             tausqd0, dtd)
                        if aic:
                            evtest = evtest + (2 - np.log(n))*(damtest+1)
                        if evtest < evmin:
                            killset = killtest
                            evmin = evtest
                            xers = Xtest
                            beters = betertest
                for k in range(0, np.size(killset)):
                    damtx = np.delete(damtx, int(np.array(killset[k]) - 1), 0)
                ev = evmin
                X = xers

                if self.ConsoleOutput:
                    if data.dtype != np.float64:  # if large dataset, then 'Gibbs: 100.00%' printed from inside gibbs
                        sys.stdout.write('\r')  # place cursor at start of line to erase 'Gibbs: 100.00%'
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
                    betas = beters
                    mtx = damtx
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
        self.betas = betas[-self.config.DEFAULT['draws']::, :]  # discard 'burnin' draws by only keeping last 'draws' draws
        self.mtx = mtx
        self.evs = evs
        return inputs, data, betas[-self.config.DEFAULT['draws']::, :], minmax, mtx, evs  # discard 'burnin'
    
    def fitupdate(self, inputs, data):
        """
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

        'ev' is a vector of BIC values from all the models evaluated
        """
        phis = self.config.DEFAULT['phis']
        relats_in = self.config.DEFAULT['relats_in']
        a = self.config.DEFAULT['a']
        b = self.config.DEFAULT['b']
        atau = self.config.DEFAULT['atau']
        btau = self.config.DEFAULT['btau']
        tolerance = self.config.DEFAULT['tolerance']
        draws = self.config.DEFAULT['burnin'] + self.config.DEFAULT['draws']  # after fitting, the 'burnin' draws will be discarded from 'betas'
        gimmie = self.config.DEFAULT['gimmie']
        way3 = self.config.DEFAULT['way3']
        aic = self.config.DEFAULT['aic']
        burn = self.config.DEFAULT['burn'] # burn draws are disregarded prior to update fitting
        sigsqd0 = self.config.DEFAULT['sigsqd0']


        def modelBuilder():
            if self.config.DEFAULT['built']:
                model = True
                mu_old = np.asmatrix(np.mean(self.betas[self.burn:-1], axis=0))
                sigma_old = np.cov(self.betas[self.burn:-1].transpose())
            else:
                model = False
                mu_old = []
                sigma_old = []
            return model, mu_old, sigma_old


        def perms(x):
            """Python equivalent of MATLAB perms."""
            # from https://stackoverflow.com/questions/38130008/python-equivalent-for-matlabs-perms
            a = np.vstack(list(itertools.permutations(x)))[::-1]

            return a

        def gibbs_Xin_update(sigsqd0, inputs, data, phis, Xin, discmtx, a, b, atau, btau, phind, xsm, \
                             mu_old, Sigma_old, draws):
            """
            This version of the sampler increases efficiency by accepting an set of
            inputs 'Xin' (matrix of data evaluated at basis function combinations)
            derived from earlier iterations.

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

            'mu_old' and 'Sigma_old' are the means and covariance matrix respectively
            of the priors of the previous model being updated.

            'draws' is the total number of draws
            """

            # building the matrix by calculating the corresponding basis function outputs for each set of inputs
            minp, ninp = np.shape(inputs)
            phi_vec = []
            if np.shape(discmtx) == ():  # part of fix for single input model
                mmtx = 1
            else:
                mmtx, null = np.shape(discmtx)

            if np.size(Xin) == 0:
                Xin = np.ones((minp, 1))
                mxin, nxin = np.shape(Xin)
            else:
                # X = Xin
                mxin, nxin = np.shape(Xin)
            if mmtx - nxin < 0:
                X = Xin
            else:
                X = np.append(Xin, np.zeros((minp, mmtx - nxin)), axis=1)

            for i in range(minp):  # for datapoint in training datapoints

                # ------------------------------
                # [IN DEVELOPMENT] PRINT PERCENT COMPLETION TO CONSOLE (reported to cause significant delay):
                #
                # if self.ConsoleOutput and data.dtype != np.float64:  # if large dataset, show progress for sanity check
                #     percent = i / (minp - 1)
                #     sys.stdout.write(f"Gibbs: {round(100 * percent, 2):.2f}%")  # show percent of data looped through
                #     sys.stdout.write('\r')  # set cursor at beginning of console output line (such that next iteration
                #         # of Gibbs progress (or [ind, ev] if at end) overwrites current Gibbs progress)
                #     sys.stdout.flush()
                #
                # [END]
                # ----------------------------

                for j in range(nxin, mmtx + 1):
                    null, nxin2 = np.shape(X)
                    if j == nxin2:
                        X = np.append(X, np.zeros((minp, 1)), axis=1)

                    phi = 1

                    for k in range(ninp):  # for input var in input vars

                        if np.shape(discmtx) == ():
                            num = discmtx
                        else:
                            num = discmtx[j - 1][k]

                        if num != 0:  # enter if loop if num is nonzero
                            nid = int(num - 1)

                            # Evaluate basis function:
                            if self.config.KERNELS[0] == self.config.DEFAULT['kernel']:  # == 'Cubic Splines':
                                coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
                            elif self.config.KERNELS[1] == self.config.DEFAULT['kernel']:  # == 'Bernoulli Polynomials':
                                coeffs = phis[nid]  # coefficients for bernoulli
                            phi = phi * self.functions.evaluate_basis(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.

                    X[i][j] = phi

            # Once X is evaluated at the new data points, three different conditions are evaluated to
            # determine the appropriate methodology to pursue.  They are:
            # 1) No given mean or covariance strong prior (first time model)
            # 2) Re-evaluate model with strong prior, but same number of terms
            # 3) Create new terms for model given a strong prior

            # Case 1) New Model
            if np.size(mu_old) == 0:
                # initialize tausqd at the mode of its prior: the inverse of the mode of
                # sigma squared, such that the initial variance for the betas is 1
                # Start both at the mode of the prior


                tausqd = 1 / sigsqd0

                XtX = np.transpose(X).dot(X)

                Xty = np.transpose(X).dot(data)

                # See the link https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
                Lamb, Q = eigh(
                    XtX)  # using scipy eigh function to avoid generation of imaginary values due to numerical errors
                # Lamb, Q = LA.eig(XtX)

                Lamb_inv = np.diag(1 / Lamb)

                betahat = Q.dot(Lamb_inv).dot(np.transpose(Q)).dot(Xty)
                # This is sum squared error, not just squared error or L2 Norm
                squerr = LA.norm(data - X.dot(betahat)) ** 2

                astar = a + 1 + len(data) / 2 + (mmtx + 1) / 2
                atau_star = atau + mmtx / 2

                dtd = np.transpose(data).dot(data)

                # Gibbs iterations

                betas = np.zeros((draws, mmtx + 1))
                sigs = np.zeros((draws, 1))
                taus = np.zeros((draws, 1))
                sigsqd = sigsqd0

                lik = np.zeros((draws, 1))
                n = len(data)

                for k in range(draws):
                    # Step 1: Hold sigma squared and tau squared constant
                    # This is to get the inverse of the new covariance
                    Lamb_tausqd = np.diag(Lamb) + (1 / tausqd) * np.identity(mmtx + 1)
                    Lamb_tausqd_inv = np.diag(1 / np.diag(Lamb_tausqd))

                    # What does mun (mu new)
                    mun = Q.dot(Lamb_tausqd_inv).dot(np.transpose(Q)).dot(Xty)
                    S = Q.dot(np.diag(np.diag(Lamb_tausqd_inv) ** (1 / 2)))

                    vec = np.random.normal(loc=0, scale=1, size=(mmtx + 1, 1))  # drawing from normal distribution
                    betas[k][:] = np.transpose(mun + sigsqd ** (1 / 2) * (S).dot(vec))

                    # Components for the likelihood
                    comp1 = -(n / 2) * np.log(sigsqd)
                    comp2 = np.transpose(betahat) - betas[k][:]
                    comp3 = betahat - np.reshape(betas[k][:], (len(betas[k][:]), 1))
                    # forcing array into a column for comp3, np.transpose not effective
                    lik[k] = comp1 - (squerr + comp2.dot(XtX).dot(comp3)) / (2 * sigsqd)

                    # vecc is difference between mu new and betas calculated
                    vecc = mun - np.reshape(betas[k][:], (len(betas[k][:]), 1))

                    # Step 2: Hold Beta and tau squared constant, calculate sigma squared
                    comp1 = 0.5 * np.transpose(vecc)
                    comp2 = (XtX + (1 / tausqd) * np.identity(mmtx + 1)).dot(vecc)
                    comp3 = 0.5 * np.transpose(mun).dot(Xty)

                    bstar = b + comp1.dot(comp2) + 0.5 * dtd - comp3;

                    # Returning a 'not a number' constant if bstar is negative, which would
                    # cause np.random.gamma to return a ValueError
                    if bstar < 0:
                        sigsqd = math.nan
                    else:
                        sigsqd = 1 / np.random.gamma(astar, 1 / bstar)

                    sigs[k] = sigsqd

                    # Step 3: Hold Beta and sigma squared constant, calculate tau squared
                    btau_star = (1 / (2 * sigsqd)) * (
                        betas[k][:].dot(np.reshape(betas[k][:], (len(betas[k][:]), 1)))) + btau

                    tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
                    taus[k] = tausqd

                # Calculate the evidence (BIC)
                ev = (mmtx + 1) * np.log(n) - 2 * max(lik)

                X = X[:, 0:mmtx]

                return betas, sigs, taus, X, ev

            # Case 2) Given Strong Prior (Old Model), no new terms
            elif np.shape(mu_old)[1] == mmtx + 1:
                print('same')
                # seperate into a section with X_old and X_new based on the existing
                # parameters noted in Xin

                null, num_old_terms = np.shape(mu_old)

                length_old = num_old_terms

                X_old = X

                # initialize tausqd at the mode of its prior: the inverse of the mode of
                # sigma squared, such that the initial variance for the betas is 1
                # Start both at the mode of the prior
                tausqd = 1 / sigsqd0

                # Precompute Terms related to 'old' values
                XotXo = np.asmatrix(np.transpose(X_old).dot(X_old))

                Xoty = np.transpose(X_old).dot(data)

                Sigma_old_inverse = np.linalg.inv(Sigma_old)

                # Precompute terms for sigma squared and tau squared
                astar = a + len(data) / 2 + (mmtx + 1) / 2
                atau_star = atau + (mmtx + 1) / 2

                yty = np.transpose(data).dot(data)
                ytXo = np.transpose(data).dot(X_old)

                # Gibbs iterations

                betas_old = np.asmatrix(np.zeros((draws, num_old_terms)))
                sigs = np.zeros((draws, 1))
                taus = np.zeros((draws, 1))
                sigsqd = sigsqd0

                lik = np.zeros((draws, 1))
                n = len(data)

                mu_old = mu_old.transpose()

                for k in range(draws):
                    # Step 1: Hold betas_new, sigma squared, and tau squared constant. Get betas_old
                    Sigma_old_inverse_post = XotXo + (1 / tausqd) * Sigma_old_inverse
                    Sigma_old_post = np.linalg.inv(Sigma_old_inverse_post)

                    # See the link https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
                    Lamb_old, Q_old = eigh(XotXo + (
                                1 / tausqd) * Sigma_old_inverse)  # using scipy eigh function to avoid generation of imaginary values due to numerical errors
                    # Lamb_tausqd_inv_old = np.diag(np.linalg.inv((np.diag(Lamb_old))))
                    Lamb_tausqd_inv_old = 1/Lamb_old

                    # Mean of distribution
                    mu_old_first_part = (Xoty + (1 / tausqd * Sigma_old_inverse).dot(mu_old))
                    mu_old_post = Sigma_old_post.dot(mu_old_first_part)
                    # print('Sigma_old_inverse.dot(mu_old)', Sigma_old_inverse.dot(mu_old))
                    S_old = Q_old.dot((np.diag(Lamb_tausqd_inv_old) ** (1 / 2)))

                    # Random draw for sampling
                    vec_old = np.random.normal(loc=0, scale=1, size=(length_old, 1))  # drawing from normal distribution
                    betas_old[k][:] = np.transpose(mu_old_post + sigsqd ** (1 / 2) * (S_old).dot(vec_old))

                    # Step 2: Hold betas_old, betas_new, and tau squared constant, calculate sigma squared
                    comp1 = 0.5 * (yty - ytXo.dot(betas_old[k][:].transpose()))
                    comp2 = 0.5 * (-(betas_old[k][:]).dot(Xoty) + (betas_old[k][:]).dot(XotXo).dot(
                        betas_old[k][:].transpose()))
                    comp3 = 0.5 * (1 / tausqd) * (
                                (betas_old[k][:]).dot(Sigma_old_inverse).dot(betas_old[k][:].transpose()) - (
                        betas_old[k][:]).dot(Sigma_old_inverse).dot(mu_old))
                    comp4 = 0.5 * (1 / tausqd) * (-np.transpose(mu_old).dot(Sigma_old_inverse).dot(
                        betas_old[k][:].transpose()) + np.transpose(mu_old).dot(Sigma_old_inverse).dot(mu_old))

                    bstar = comp1 + comp2 + comp3 + comp4 + b
                    # Returning a 'not a number' constant if bstar is negative, which would
                    # cause np.random.gamma to return a ValueError
                    if bstar < 0:
                        sigsqd = math.nan
                    else:
                        sigsqd = 1 / np.random.gamma(astar, 1 / bstar)

                    sigs[k] = sigsqd

                    # Step 3: Hold betas_old, betas_new, and sigma squared constant, calculate tau squared
                    comp1 = 0.5 * (1 / sigsqd) * (
                                (betas_old[k][:]).dot(Sigma_old_inverse).dot(betas_old[k][:].transpose()) - (
                        betas_old[k][:]).dot(Sigma_old_inverse).dot(mu_old))
                    comp2 = 0.5 * (1 / sigsqd) * (-np.transpose(mu_old).dot(Sigma_old_inverse).dot(
                        betas_old[k][:].transpose()) + np.transpose(mu_old).dot(Sigma_old_inverse).dot(mu_old))

                    btau_star = comp1 + comp2 + btau

                    tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
                    taus[k] = tausqd

                    # Step 5: for BIC, calculate the Natural Log Likelihood
                    comp1 = -(n / 2) * np.log(sigsqd)
                    comp2 = yty - ytXo.dot(betas_old[k][:].transpose())
                    comp3 = -(betas_old[k][:]).dot(Xoty) + (betas_old[k][:]).dot(XotXo).dot(betas_old[k][:].transpose())

                    lik[k] = comp1 - 0.5 / sigsqd * (comp2 + comp3)

                # Calculate the evidence (BIC)
                ev = (mmtx + 1) * np.log(n) - 2 * max(lik)

                # Updated X matrix, not differentiated between old and new though
                X = X[:, 0:mmtx]

                betas = betas_old

                return betas, sigs, taus, X, ev

            # Case 3) Give Strong Prior (Old Model), creating new terms
            elif np.shape(mu_old)[1] < mmtx + 1:
                print('new')
                # seperate into a section with X_old and X_new based on the existing
                # parameters noted in Xin

                null, num_old_terms = np.shape(mu_old)

                length_old = num_old_terms
                length_new = mmtx - num_old_terms + 1

                X_old = X[:, 0:length_old]
                X_new = X[:, length_old: length_old + length_new]

                # initialize tausqd at the mode of its prior: the inverse of the mode of
                # sigma squared, such that the initial variance for the betas is 1
                # Start both at the mode of the prior
                tausqd = 1 / sigsqd0

                # Precompute Terms related to 'old' values
                XotXo = np.asmatrix(np.transpose(X_old).dot(X_old))

                Xoty = np.transpose(X_old).dot(data)

                Sigma_old_inverse = np.linalg.inv(Sigma_old)
                Sigma_old_inverse_post = XotXo + Sigma_old_inverse
                Sigma_old_post = np.linalg.inv(Sigma_old_inverse_post)

                # See the link https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
                Lamb_old, Q_old = eigh(
                    XotXo + Sigma_old_inverse)  # using scipy eigh function to avoid generation of imaginary values due to numerical errors
                # Lamb, Q = LA.eig(XtX)

                # Lamb_old_inv = np.diag(1/Lamb_old)

                # betahat_old = Q_old.dot(Lamb_old_inv).dot(np.transpose(Q_old)).dot(Xoty)
                # This is sum squared error, not just squared error or L2 Norm
                # squerr_old = LA.norm(data - X_old.dot(betahat_old)) ** 2

                # Precompute Terms related to 'new' values
                XntXn = np.transpose(X_new).dot(X_new)

                Xnty = np.transpose(X_new).dot(data)

                # See the link https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-covariance-matrix
                Lamb_new, Q_new = eigh(
                    XntXn)  # using scipy eigh function to avoid generation of imaginary values due to numerical errors
                # Lamb, Q = LA.eig(XtX)

                # Lamb_new_inv = np.diag(np.linalg.inv(Lamb_new))

                # betahat_new = Q_new.dot(Lamb_new_inv).dot(np.transpose(Q_new)).dot(Xnty)
                # This is sum squared error, not just squared error or L2 Norm
                # squerr_new = LA.norm(data - X_new.dot(betahat_new)) ** 2

                XotXn = np.transpose(X_old).dot(X_new)
                XntXo = np.transpose(X_new).dot(X_old)

                # Sigma_old_inverse = np.linalg.inv(Sigma_old)
                # Sigma_old_inverse_post = XotXo+Sigma_old_inverse
                # Sigma_old_post = np.linalg.inv(Sigma_old_inverse_post)
                # print('Sigma Old Post', Sigma_old_post)

                # Lamb_tausqd_old = np.diag(Lamb_old) + Sigma_old
                Lamb_tausqd_inv_old = np.diag(np.linalg.inv((np.diag(Lamb_old))))
                # print('Lamb_tausqd_old', np.shape(Lamb_tausqd_old))
                # print('Lamb_tausqd_inv_old', np.shape(Lamb_tausqd_inv_old))
                # print('Q_old', np.shape(Q_old))

                # Precompute terms for sigma squared and tau squared
                astar = a + len(data) / 2 + (mmtx + 1) / 2
                atau_star = atau + (length_new) / 2

                yty = np.transpose(data).dot(data)
                ytXo = np.transpose(data).dot(X_old)
                ytXn = np.transpose(data).dot(X_new)

                # Gibbs iterations

                betas_old = np.asmatrix(np.zeros((draws, num_old_terms)))
                betas_new = np.asmatrix(np.zeros((draws, mmtx - num_old_terms + 1)))
                sigs = np.zeros((draws, 1))
                taus = np.zeros((draws, 1))
                sigsqd = sigsqd0

                lik = np.zeros((draws, 1))
                n = len(data)

                mu_old = mu_old.transpose()

                for k in range(draws):
                    # Step 1: Hold betas_new, sigma squared, and tau squared constant. Get betas_old

                    # Mean of distribution
                    mu_old_first_part = Xoty - XotXn.dot(betas_new[k - 1].transpose()) + Sigma_old_inverse.dot(mu_old)
                    mu_old_post = Sigma_old_post.dot(mu_old_first_part)
                    S_old = Q_old.dot((np.diag(Lamb_tausqd_inv_old) ** (1 / 2)))

                    vec_old = np.random.normal(loc=0, scale=1, size=(length_old, 1))  # drawing from normal distribution
                    betas_old[k][:] = np.transpose(mu_old_post + sigsqd ** (1 / 2) * (S_old).dot(vec_old))

                    # Step 2: Hold betas_new, sigma squared, and tau squared constant. Get betas_old
                    # This is to get the inverse of the new covariance
                    Lamb_tausqd_new = np.diag(Lamb_new) + (1 / tausqd) * np.identity(length_new)
                    Lamb_tausqd_inv_new = np.diag(np.linalg.inv(Lamb_tausqd_new))

                    # Mean of distribution
                    mu_new_first_part = Xnty - XntXo.dot(betas_old[k].transpose())
                    Sigma_new_inverse_post = XntXn + (1 / tausqd) * np.identity(length_new)
                    mu_new_post = (np.linalg.inv(Sigma_new_inverse_post)).dot(mu_new_first_part)
                    S_new = Q_new.dot((np.diag(Lamb_tausqd_inv_new) ** (1 / 2)))

                    vec_new = np.random.normal(loc=0, scale=1, size=(length_new, 1))  # drawing from normal distribution
                    betas_new[k][:] = np.transpose(mu_new_post + sigsqd ** (1 / 2) * (S_new).dot(vec_new))

                    # Step 3: Hold betas_old, betas_new, and tau squared constant, calculate sigma squared
                    comp1 = 0.5 * (yty - ytXo.dot(betas_old[k][:].transpose()) - ytXn.dot(betas_new[k][:].transpose()))
                    comp2 = 0.5 * (-(betas_old[k][:]).dot(Xoty) + (betas_old[k][:]).dot(XotXo).dot(
                        betas_old[k][:].transpose()) + (betas_old[k][:]).dot(XotXn).dot(betas_new[k][:].transpose()))
                    comp3 = 0.5 * (-(betas_new[k][:]).dot(Xnty) + (betas_new[k][:]).dot(XntXo).dot(
                        betas_old[k][:].transpose()) + (betas_new[k][:]).dot(XntXn).dot(betas_new[k][:].transpose()))
                    comp4 = 0.5 / tausqd * ((betas_new[k][:]).dot(betas_new[k][:].transpose()))
                    comp5 = 0.5 * ((betas_old[k][:]).dot(Sigma_old_inverse).dot(betas_old[k][:].transpose()) - (
                    betas_old[k][:]).dot(Sigma_old_inverse).dot(mu_old))
                    comp6 = 0.5 * (-np.transpose(mu_old).dot(Sigma_old_inverse).dot(
                        betas_old[k][:].transpose()) + np.transpose(mu_old).dot(Sigma_old_inverse).dot(mu_old))

                    bstar = comp1 + comp2 + comp3 + comp4 + comp5 + comp6 + b
                    # Returning a 'not a number' constant if bstar is negative, which would
                    # cause np.random.gamma to return a ValueError
                    if bstar < 0:
                        sigsqd = math.nan
                    else:
                        sigsqd = 1 / np.random.gamma(astar, 1 / bstar)

                    sigs[k] = sigsqd

                    # Step 4: Hold betas_old, betas_new, and sigma squared constant, calculate tau squared
                    btau_star = (1 / (2 * sigsqd)) * (betas_new[k][:].dot(betas_new[k][:].transpose())) + btau

                    tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
                    taus[k] = tausqd

                    # Step 5: for BIC, calculate the Natural Log Likelihood
                    comp1 = -(n / 2) * np.log(sigsqd)
                    comp2 = yty - ytXo.dot(betas_old[k][:].transpose()) - ytXn.dot(betas_new[k][:].transpose())
                    comp3 = -(betas_old[k][:]).dot(Xoty) + (betas_old[k][:]).dot(XotXo).dot(
                        betas_old[k][:].transpose()) + (betas_old[k][:]).dot(XotXn).dot(betas_new[k][:].transpose())
                    comp4 = -(betas_new[k][:]).dot(Xnty) + (betas_new[k][:]).dot(XntXo).dot(
                        betas_old[k][:].transpose()) + (betas_new[k][:]).dot(XntXn).dot(betas_new[k][:].transpose())

                    lik[k] = comp1 - 0.5 / sigsqd * (comp2 + comp3 + comp4)

                # Calculate the evidence (BIC)
                ev = (mmtx + 1) * np.log(n) - 2 * max(lik)

                # Updated X matrix, not differentiated between old and new though
                X = X[:, 0:mmtx]

                betas = np.concatenate((betas_old, betas_new), axis=1)

                return betas, sigs, taus, X, ev


            # Case 4) Something unexpected happens
            else:
                print('Error: No appropriate cases for evaluation found.')
                return

        # Check if initial model of updated
        [model, mu_old, sigma_old] = modelBuilder()

        # Prepare phind and xsm if using cubic splines, else match variable names required for gibbs argument
        if self.config.KERNELS[0] == self.config.DEFAULT['kernel']:  # == 'Cubic Splines':
            _, phind, xsm = self.dataFormat._inputs_to_phind(inputs)  # ..., phis=self.phis, kernel=self.kernel) already true
        elif self.config.KERNELS[1] == self.config.DEFAULT['kernel']:  # == 'Bernoulli Polynomials':
            phind = None
            xsm = inputs


        # 'n' is the number of datapoints whereas 'm' is the number of inputs
        n, m = np.shape(inputs)
        mrel = n
        damtx = []
        evs = []

        # Conversion of Lines 79-100 of emulator_Xin.m
        if np.logical_not(
                all([isinstance(index, int) for index in relats_in])):  # checks relats to see if it's an array
            if sum(np.logical_not(relats_in)).all():
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

        # Define the number of terms the interaction matrix needs
        if np.size(mu_old) == 0:
            num_old_terms = 0
        else:
            null, num_old_terms = np.shape(mu_old)

        # End conversion

        # 'ind' is an integer which controls the development of new terms
        ind = 1
        greater = 0
        finished = 0
        X = []

        while True:
            # first we have to come up with all combinations of 'm' integers that
            # sums up to ind (by twos since we only do two-way interactions)
            if ind == 1:
                i_list = [0]
            else:
                i_list = np.arange(0, math.floor(ind / 2) + 0.1, 1)
                i_list = i_list[::-1]
                # adding 0.1 to correct index list generation using floor function

            for i in i_list:

                if m > 1:
                    vecs = np.zeros(m)
                    vecs[0] = ind - i
                    vecs[1] = i
                    vecs = np.unique(perms(vecs), axis=0)

                    killrow = []
                    for t in range(mrel):
                        for iter in range(vecs.shape[0]):
                            if np.array_equal(relats_in[t, :].ravel().nonzero(), vecs[iter, :].ravel().nonzero()):
                                killrow.append(iter)
                    vecs = np.delete(vecs, killrow, 0)

                else:
                    vecs = ind

                if np.size(damtx) == 0:
                    damtx = vecs
                else:
                    if np.shape(damtx) == () or np.shape(vecs) == ():  # part of fix for single input model
                        if np.shape(damtx) == ():
                            damtx = np.array([damtx, vecs])
                            damtx = np.reshape(damtx, [len(damtx), 1])
                        else:
                            damtx = np.append(damtx, vecs)
                            damtx = np.reshape(damtx, [len(damtx), 1])
                    else:
                        damtx = np.concatenate((damtx, vecs), axis=0)

                interaction_matrix_length, null = np.shape(damtx)

                if num_old_terms - 1 <= interaction_matrix_length:  # Make sure number of terms is appropriate

                    betas, null, null, X, ev \
                        = gibbs_Xin_update(sigsqd0, inputs, data, phis, X, damtx, a, b, atau \
                                           , btau, phind, xsm, mu_old, sigma_old, draws=draws)

                    # Boolean implementation of the AIC if passed as 'True'
                    if aic:
                        if np.shape(damtx) == ():  # for single input models
                            dam = 1
                        else:
                            dam, null = np.shape(damtx)

                        ev = ev + (2 - np.log(n)) * dam

                    print(ind, float(ev))

                    # Keep running list of the evidence values for the sampling
                    if np.size(evs) == 0:
                        evs = ev
                    else:
                        evs = np.concatenate((evs, ev))

                    # ev (evidence) is the BIC and the smaller the better

                    if ev == min(evs):
                        betas_best = betas
                        mtx = damtx
                        greater = 1

                    elif greater <= tolerance:
                        greater = greater + 1

                    else:
                        finished = 1
                        self.config.DEFAULT['built'] = True
                        break

                    if m == 1:
                        break
            if finished != 0:
                break

            ind = ind + 1

            if ind > len(phis):
                break

        # Implementation of 'gimme' feature
        if gimmie:
            betas_best = betas
            mtx = damtx

        return betas_best, mtx, evs
    # return inputs, data, betas[-self.config.DEFAULT['draws']::, :], minmax, mtx, ev



