from FoKL import getKernels
import pandas as pd
import warnings
import itertools
import math
import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh
import matplotlib.pyplot as plt


class FoKL:
    def __init__(self, **kwargs):
        """
        Initialization Inputs (i.e., hyperparameters and their descriptions):

            - 'phis' is a data structure with the spline coefficients for the basis
            functions, built with 'spline_coefficient.txt' and 'splineconvert' or
            'spline_coefficient_500.txt' and 'splineconvert500' (the former provides
            25 basis functions: enough for most things -- while the latter provides
            500: definitely enough for anything)

            - 'relats_in' is a boolean matrix indicating which terms should be excluded
            from the model building; for instance if a certain main effect should be
            excluded relats will include a row with a 1 in the column for that input
            and zeros elsewhere; if a certain two way interaction should be excluded
            there should be a row with ones in those columns and zeros elsewhere.
            to exclude no terms 'relats = np.array([[0]])'. an example of excluding
            the first input main effect and its interaction with the third input for
            a case with three total inputs is: 'relats = np.array([[1,0,0],[1,0,1]])'

            - 'a' and 'b' are the shape and scale parameters of the ig distribution for
            the observation error variance of the data. the observation error model is
            white noise. choose the mode of the ig distribution to match the noise in
            the output dataset and the mean to broaden it some

            - 'atau' and 'btau' are the parameters of the ig distribution for the 'tau
            squared' parameter: the variance of the beta priors is iid normal mean
            zero with variance equal to sigma squared times tau squared. tau squared
            must be scaled in the prior such that the product of tau squared and sigma
            squared scales with the output dataset

            - 'tolerance' controls how hard the function builder tries to find a better
            model once adding terms starts to show diminishing returns. a good
            default is 3 -- large datasets could benefit from higher values

            - 'draws' is the total number of draws from the posterior for each tested
            model

            - 'gimmie' is a boolean causing the routine to return the most complex
            model tried instead of the model with the optimum bic

            - 'way3' is a boolean specifying the calculation of three-way interactions

            - 'threshav' and 'threshstda' form a threshold for the elimination of terms
                - 'threshav' is a threshold for proposing terms for elimination based on
                their mean values (larger thresholds lead to more elimination)
                - 'threshstda' is a threshold standard deviation -- expressed as a fraction
                relative to the mean
                - terms with coefficients that are lower than 'threshav' and higher than
                'threshstda' will be proposed for elimination (elimination will happen or not
                based on relative BIC values)

            - 'threshstdb' is a threshold standard deviation that is independent of the
            mean value of the coefficient -- all with a standard deviation (fraction
            relative to mean) exceeding this value will be proposed for elimination

            - 'aic' is a boolean specifying the use of the aikaike information
            criterion

        Default Values:

            - phis       = getKernels.sp500()
            - relats_in  = []
            - a          = 4
            - b          = f(a, data)
            - atau       = 4
            - btau       = f(atau, data)
            - tolerance  = 3
            - draws      = 1000
            - gimmie     = False
            - way3       = False
            - threshav   = 0.05
            - threshstda = 0.5
            - threshstdb = 2
            - aic        = False

        Other Optional Inputs:

            - UserWarnings == boolean to output user-warnings to the command terminal == 0 (default)
        """

        # Define default keywords:
        hypers = {'phis': getKernels.sp500(), 'relats_in': [], 'a': 4, 'b': 'default', 'atau': 4, 'btau': 'default',
                  'tolerance': 3, 'draws': 1000, 'gimmie': False, 'way3': False, 'threshav': 0.05, 'threshstda': 0.5,
                  'threshstdb': 2, 'aic': False}
        other_kwargs = {'UserWarnings': 0}

        # Update keywords based on user-input:
        kwargs_expected = set(hypers.keys()).union(other_kwargs.keys())
        for kwarg in kwargs.keys():
            if kwarg not in kwargs_expected:
                raise ValueError(f"Unexpected keyword argument: {kwarg}")
            else:
                if kwarg in other_kwargs.keys():
                    other_kwargs[kwarg] = kwargs.get(kwarg, other_kwargs.get(kwarg))
                else:
                    hypers[kwarg] = kwargs.get(kwarg, hypers.get(kwarg))

        # Define each hyperparameter as an attribute of 'self':
        for hyperKey, hyperValue in hypers.items():
            setattr(self, hyperKey, hyperValue)

        # Turn off/on user-warnings:
        UserWarnings = other_kwargs.get('UserWarnings')
        if isinstance(UserWarnings, str):
            UserWarnings = UserWarnings.lower()
            if UserWarnings in ['yes', 'y', 'on', 'all', 'true']:
                UserWarnings = 1
            elif UserWarnings in ['no', 'n', 'off', 'none', 'n/a', 'false']:
                UserWarnings = 0
        if UserWarnings:
            warnings.filterwarnings("default", category=UserWarning)
        else:
            warnings.filterwarnings("ignore", category=UserWarning)


    def splineconvert500(self,A):
        """
        Same as splineconvert, but for a larger basis of 500
        """

        coef = np.loadtxt(A)

        phi = []
        for i in range(500):
            a = coef[i * 499:(i + 1) * 499, 0]
            b = coef[i * 499:(i + 1) * 499, 1]
            c = coef[i * 499:(i + 1) * 499, 2]
            d = coef[i * 499:(i + 1) * 499, 3]

            phi.append([a, b, c, d])

        return phi


    def inputs_to_phind(self, inputs, phis):
        """
        Twice normalize the inputs to index the spline coefficients.

        Inputs:
            - inputs == normalized inputs as numpy array (i.e., self.inputs.np)
            - phis   == spline coefficients

        Output (and appended class attributes):
            - phind == index to spline coefficients
            - xsm   ==
        """

        L_phis = len(phis[0][0])  # = 499, length of coeff. in basis funtions
        phind = np.array(np.ceil(inputs * L_phis), dtype=int)  # 0-1 normalization to 0-499 normalization

        if phind.ndim == 1:  # if phind.shape == (number,) != (number,1), then add new axis to match indexing format
            phind = phind[:, np.newaxis]

        set = (phind == 0)  # set = 1 if phind = 0, otherwise set = 0
        phind = phind + set  # makes sense assuming L_phis > M

        r = 1 / L_phis  # interval of when basis function changes (i.e., when next cubic function defines spline)
        xmin = (phind - 1) * r
        X = (inputs - xmin) / r  # twice normalized inputs (0-1 first then to size of phis second)

        self.phind = phind - 1  # adjust MATLAB indexing to Python indexing after twice normalization
        self.xsm = L_phis * inputs - phind

        return self.phind, self.xsm


    def bss_derivatives(self, **kwargs):
        """
        For returning gradient of modeled function with respect to each, or specified, input variable.
        If user overrides default settings, then 1st and 2nd partial derivatives can be returned for any variables.

        Keyword Inputs:
            inputs == NxM matrix of 'x' input variables for fitting f(x1, ..., xM)    == self.inputs_np (default)
            d1     == index of input variable(s) to use for first partial derivative  == 'all' (default)
            d2     == index of input variable(s) to use for second partial derivative == 'none' (default)
            draws  == number of beta terms used                                       == self.draws (default)
            betas  == draw from the posterior distribution of coefficients            == self.betas (default)
            phis   == spline coefficients for the basis functions                     == self.phis (default)
            mtx    == basis function interaction matrix from the best model           == self.mtx (default)
            span   == list of [min, max]'s of input data used in the normalization    == self.normalize (default)
            IndividualDraws == boolean for returning derivative(s) at each draw       == 0 (default)
            ReturnFullArray == boolean for returning NxMx2 array instead of Nx2M      == 0 (default)

        Return Outputs:
            dState == derivative of input variables (i.e., states)
        """

        # Default keywords:
        kwargs_all = {'inputs': 'default', 'd1': 'default', 'd2': 'default', 'draws': 'default', 'betas': 'default',
                      'phis': 'default', 'mtx': 'default', 'span': 'default', 'IndividualDraws': 0,
                      'ReturnFullArray': 0, 'ReturnBasis': 0}

        # Update keywords based on user-input:
        for kwarg in kwargs.keys():
            if kwarg not in kwargs_all.keys():
                raise ValueError(f"Unexpected keyword argument: {kwarg}")
            else:
                kwargs_all[kwarg] = kwargs.get(kwarg, kwargs_all.get(kwarg)) # update

        # Define local variables from keywords:
        inputs = kwargs_all.get('inputs')
        d1 = kwargs_all.get('d1')
        d2 = kwargs_all.get('d2')
        draws = kwargs_all.get('draws')
        betas = kwargs_all.get('betas')
        phis = kwargs_all.get('phis')
        mtx = kwargs_all.get('mtx')
        span = kwargs_all.get('span')
        IndividualDraws = kwargs_all.get('IndividualDraws')
        ReturnFullArray = kwargs_all.get('ReturnFullArray')
        ReturnBasis = kwargs_all.get('ReturnBasis')

        # Further processing of keyword arguments:
        if isinstance(inputs, str): # handle case of user using bss_derivatives without first calling model.fit
            if inputs.lower() == 'default':
                inputs = self.inputs_np
        if isinstance(draws, str): # handle case of user using bss_derivatives without first calling model.fit
            if draws.lower() == 'default':
                draws = self.draws
        if isinstance(betas, str): # handle case of user using bss_derivatives without first calling model.fit
            if betas.lower() == 'default':
                betas = self.betas
        if isinstance(phis, str): # handle case of user using bss_derivatives without first calling model.fit
            if phis.lower() == 'default':
                phis = self.phis
        if isinstance(mtx, str): # handle case of user using bss_derivatives without first calling model.fit
            if mtx.lower() == 'default':
                mtx = self.mtx
        if isinstance(span, str): # handle case of user using bss_derivatives without first calling model.fit
            if span.lower() == 'default':
                span = self.normalize
        if isinstance(ReturnFullArray, str): # handle exception of user-defined string (not boolean)
            ReturnFullArray = ReturnFullArray.lower()
            if ReturnFullArray in ['yes', 'y', 'on', 'all', 'true']:
                ReturnFullArray = 1
            elif ReturnFullArray in ['no', 'n', 'off', 'none', 'n/a', 'false']:
                ReturnFullArray = 0

        inputs = np.array(inputs)
        if inputs.ndim == 1:
            inputs = inputs[:, np.newaxis]
        if isinstance(betas, list): # then most likely user-input, e.g., [0,1]
            betas = np.array(betas)
            if betas.ndim == 1:
                betas = betas[:, np.newaxis]
        if isinstance(mtx, int): # then most likely user-input, e.g., 1
            mtx = np.array(mtx)
            mtx = mtx[np.newaxis, np.newaxis]
        else:
            mtx = np.array(mtx)
            if mtx.ndim == 1:
                mtx = mtx[:, np.newaxis]
        if len(span) == 2: # if span=[0,1], span=[[0,1],[0,1]], or span=[array([0,1]),array([0,1])]
            if not (isinstance(span[0], list) or isinstance(span[0], np.ndarray)):
                span = [span] # make list of list to handle indexing errors for one input variable case, i.e., [0,1]

        if np.max(np.max(inputs)) > 1 or np.min(np.min(inputs)) < 0:
            warnings.warn("Input 'inputs' should be normalized (0-1). Auto-normalization is in-development.",
                          category=UserWarning)

        N = np.shape(inputs)[0]  # number of datapoints (i.e., experiments/timepoints)
        B,M = np.shape(mtx) # B == beta terms in function (not including betas0), M == number of input variables

        if B != np.shape(betas)[1]-1: # second axis is beta terms (first is draws)
            betas = np.transpose(betas)
            if B != np.shape(betas)[1]-1:
                raise ValueError(
                    "The shape of 'betas' does not align with the shape of 'mtx'. Transposing did not fix this.")

        derv = []
        i = 0
        for di in [d1, d2]:
            i = i + 1
            error_di = 1
            if isinstance(di, str):
                di = di.lower()
                if di == 'default' and i == 1:
                    di = np.ones(M) # default is all first derivatives (i.e., gradient)
                elif di == 'default' and i == 2:
                    di = np.zeros(M) # default is no second derivatives (i.e., gradient)
                elif di in ['on', 'yes', 'y', 'all', 'true']:
                    di = np.ones(M)
                elif di in ['off', 'no', 'n', 'none', 'n/a', 'false']:
                    di = np.zeros(M)
                else:
                    raise ValueError(
                        "Keyword input 'd1' and/or 'd2', if entered as a string, is limited to 'all' or 'none'.")
                error_di = 0
            elif isinstance(di, list): # e.g., d1 = [0, 0, 1, 0] for M = 4
                if len(di) == 1: # e.g., d1 = [3] instead of d1 = 3
                    di = di[0] # list to integer, --> error_di = 0 later
                elif len(di) == M:
                    error_di = 0
                else:
                    raise ValueError("Keyword input 'd1' and/or 'd2', if entered as a list, must be of equal length to "
                                     "the number of input variables.")
            if isinstance(di, int): # not elif because d1 might have gone list to integer in above elif
                if di != 0:
                    di_id = di - 1 # input var index to Python index
                    di = np.zeros(M)
                    di[di_id] = 1
                else: # treat as di = 'none'
                    di = np.zeros(M)
                error_di = 0
            if error_di:
                raise ValueError(
                    "Keyword input 'd1' and/or 'd2' is limited to an integer indexing an input variable, or to a list "
                    "of booleans corresponding to the input variables.")
            derv.append(di) # --> derv = [d1, d2], after properly formatted

        phind, _ = self.inputs_to_phind(inputs, phis)

        # Determine if only one or both derivatives should be run through in for loop:
        d1_log = any(derv[0])
        d2_log = any(derv[1])
        if d1_log and d2_log:
            d1d2_log = [0,1] # index for both first and second derivatives
        elif d1_log:
            d1d2_log = [0]  # index for only first derivative
        elif d2_log:
            d1d2_log = [1]  # index for only second derivative
        else:
            warnings.warn("Function 'bss_derivatives' was called but no derivatives were requested.",
                          category=UserWarning)
            return

        span_m = []
        for m in range(M):
            span_mi = span[m][1] - span[m][0]  # max minus min, = range of normalization per input variable
            span_m.append(span_mi)

        # Initialization before loops:
        dState = np.zeros([draws,N,M,2])
        phis_func_if_d0_for_nm = np.zeros([N,M,B]) # constant term for (n,md,b) that avoids repeat calculations
        phi = np.zeros([N,M,2])

        # Cycle through each timepoint, input variable, and perform 'bss_derivatives' like in MATLAB:
        if ReturnBasis:  # mostly for development to confirm basis function for mtx=1 and betas=[0,1]
            basis = np.zeros(N)
        for n in range(N): # loop through experiments (i.e., each timepoint/datapoint)
            for m in range(M): # loop through input variables (i.e., the current one to differentiate wrt if requested)
                for di in d1d2_log: # for first through second derivatives (or, only first/second depending on d1d2_log)
                    if derv[di][m]: # if integrating, then do so once or twice (depending on di) wrt to xm ONLY
                        span_L = span_m[m] / L_phis  # used multiple times in calculations below
                        derv_nm = np.zeros(M)
                        derv_nm[m] = di + 1 # if d2 = [1, 1, 1, 1], then for m = 2, derv_nm = [0, 0, 2, 0] like MATLAB

                        # The following is like the MATLAB function, with indexing for looping through n and m:
                        for b in range(B):  # loop through betas
                            phi[n,m,di] = 1 # reset after looping through non-wrt input variables (i.e., md)
                            for md in range(M):  # for input variable PER single differentiation, m of d(xm)
                                num = int(mtx[b,md])
                                if num: # if not 0
                                    derp = derv_nm[md]
                                    num = int(num - 1) # MATLAB to Python indexing
                                    phind_md = int(phind[n,md]) # make integer for indexing syntax
                                    if derp == 0: # if not wrt x_md
                                        if not phis_func_if_d0_for_nm[n,md,b]: # constant per (n,b,md)
                                            phis_func_if_d0_for_nm[n,md,b] = phis[num][0][phind_md] + \
                                                phis[num][1][phind_md]*X[n,md] + \
                                                phis[num][2][phind_md]*math.pow(X[n,md],2) + \
                                                phis[num][3][phind_md]*math.pow(X[n,md],3)
                                        phi[n,m,di] = phi[n,m,di] * phis_func_if_d0_for_nm[n,md,b]
                                    elif derp == 1: # if wrt x_md and first derivative
                                        phi[n,m,di] = phi[n,m,di]*(phis[num][1][phind_md] + \
                                            2*phis[num][2][phind_md]*X[n,md] + \
                                            3*phis[num][3][phind_md]*math.pow(X[n,md],2))/span_L
                                    elif derp == 2: # if wrt x_md and second derivative
                                        phi[n,m,di] = phi[n,m,di]*(2*phis[num][2][phind_md] + \
                                            6*phis[num][3][phind_md]*X[n,md])/math.pow(span_L,2)
                                    if ReturnBasis:  # mostly for development
                                        basis[n] = phis[num][0][phind_md] + phis[num][1][phind_md] * X[n,md] + \
                                            phis[num][2][phind_md] * math.pow(X[n,md],2) + \
                                            phis[num][3][phind_md] * math.pow(X[n,md],3)
                                elif derv_nm[md]:  # for terms that do not contain the variable being differentiated by
                                    phi[n,m,di] = 0
                                    break

                            dState[:,n,m,di] = dState[:,n,m,di] + betas[-draws:,b+1]*phi[n,m,di] # update after md loop

        dState = np.transpose(dState, (1, 2, 3, 0)) # move draws dimension to back so that dState is like (N,M,di,draws)

        if not IndividualDraws and draws > 1:  # then average draws
            dState = np.mean(dState, axis=3) # note 3rd axis index (i.e., 4th) automatically removed (i.e., 'squeezed')
            dState = dState[:, :, :, np.newaxis] # add new axis to avoid error in concatenate below

        if not ReturnFullArray: # then return only columns with values and stack d1 and d2 next to each other
            dState = np.concatenate([dState[:, :, 0, :], dState[:, :, 1, :]], axis=1)  # (N,M,2,draws) to (N,2M,draws)
            dState = dState[:, ~np.all(dState == 0, axis=0)] # remove columns ('N') with all zeros
        dState = np.squeeze(dState) # remove unnecessary axes

        if ReturnBasis: # mostly for development
            return dState, basis
        else: # standard
            return dState


    def evaluate(self, inputs, **kwargs):
        """
        Evaluate the inputs and output the predicted values of corresponding data. Optionally, calculate bounds.

        Input:
            inputs == matrix of independent (or non-linearly dependent) 'x' variables for evaluating f(x1, ..., xM)

        Keyword Inputs:
            draws        == number of beta terms used                              == self.draws (default)
            nform        == logical to automatically normalize and format 'inputs' == 1 (default)
            ReturnBounds == logical to return confidence bounds as second output   == 0 (default)
        """

        # Default keywords:
        kwargs_all = {'draws': self.draws, 'nform': 1, 'ReturnBounds': 0}

        # Update keywords based on user-input:
        for kwarg in kwargs.keys():
            if kwarg not in kwargs_all.keys():
                raise ValueError(f"Unexpected keyword argument: {kwarg}")
            else:
                kwargs_all[kwarg] = kwargs.get(kwarg, kwargs_all.get(kwarg))

        # Define local variables:
        # for kwarg in kwargs_all.keys():
        #     locals()[kwarg] = kwargs_all.get(kwarg) # defines each keyword (including defaults) as a local variable
        draws = kwargs_all.get('draws')
        nform = kwargs_all.get('nform')
        ReturnBounds = kwargs_all.get('ReturnBounds')

        # Process nform:
        if isinstance(nform, str):
            if nform.lower() in ['yes','y','on','auto','default','true']:
                nform = 1
            elif nform.lower() in ['no','n','off','false']:
                nform = 0
        else:
            if nform not in [0,1]:
                raise ValueError("Keyword argument 'nform' must a logical 1 (default) or 0.")

        # Automatically normalize and format inputs:
        def auto_nform(inputs):

            # Convert 'inputs' to numpy if pandas:
            if any(isinstance(inputs, type) for type in (pd.DataFrame, pd.Series)):
                inputs = inputs.to_numpy()
                warnings.warn("'inputs' was auto-converted to numpy. Convert manually for assured accuracy.", UserWarning)

            # Normalize 'inputs' and convert to proper format for FoKL:
            inputs = np.array(inputs) # attempts to handle lists or any other format (i.e., not pandas)
            # . . . inputs = {ndarray: (N, M)} = {ndarray: (datapoints, input variables)} =
            # . . . . . . array([[x1(t1),x2(t1),...,xM(t1)],[x1(t2),x2(t2),...,xM(t2)],...,[x1(tN),x2(tN),...,xM(tN)]])
            inputs = np.squeeze(inputs) # removes axes with 1D for cases like (N x 1 x M) --> (N x M)
            if inputs.ndim == 1:  # if inputs.shape == (number,) != (number,1), then add new axis to match FoKL format
                inputs = inputs[:, np.newaxis]
            N = inputs.shape[0]
            M = inputs.shape[1]
            if M > N: # if more "input variables" than "datapoints", assume user is using transpose of proper format above
                inputs = inputs.transpose()
                warnings.warn("'inputs' was transposed. Ignore if more datapoints than input variables.", category=UserWarning)
                N_old = N
                N = M # number of datapoints (i.e., timestamps)
                M = N_old # number of input variables
            minmax = self.normalize
            inputs_min = np.array([minmax[ii][0] for ii in range(len(minmax))])
            inputs_max = np.array([minmax[ii][1] for ii in range(len(minmax))])
            inputs = (inputs - inputs_min) / (inputs_max - inputs_min)

            nformputs = inputs.tolist() # convert to list, which is proper format for FoKL, like:
            # . . . {list: N} = [[x1(t1),x2(t1),...,xM(t1)],[x1(t2),x2(t2),...,xM(t2)],...,[x1(tN),x2(tN),...,xM(tN)]]

            return nformputs

        if nform:
            normputs = auto_nform(inputs)
        else: # assume provided inputs are already normalized and formatted
            normputs = inputs

        betas = self.betas
        mtx = self.mtx
        phis = self.phis

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

        # # ================================================================
        # # Outdated as of v3.1.0 because of new 'inputs_to_phind' function:
        #
        # for i in range(n):
        #     phind = []  # Rounded down point of input from 0-499
        #     for j in range(len(normputs[i])):
        #         phind.append(math.floor(normputs[i, j] * 498))
        #         # 499 changed to 498 for python indexing
        #
        #     phind_logic = []
        #     for k in range(len(phind)):
        #         if phind[k] == 498:
        #             phind_logic.append(1)
        #         else:
        #             phind_logic.append(0)
        #
        #     phind = np.subtract(phind, phind_logic)
        #
        #     for j in range(1, mbets):
        #         phi = 1
        #         for k in range(mputs):
        #             num = mtx[j - 1, k]
        #             if num > 0:
        #                 xsm = 498 * normputs[i][k] - phind[k]
        #                 phi = phi * (phis[int(num) - 1][0][phind[k]] + phis[int(num) - 1][1][phind[k]] * xsm +
        #                              phis[int(num) - 1][2][phind[k]] * xsm ** 2 + phis[int(num) - 1][3][
        #                                  phind[k]] * xsm ** 3)
        #
        # # End
        # # ================================================================
        # v3.1.0 update:

        phind, xsm = self.inputs_to_phind(normputs, phis)
        for i in range(n):
            for j in range(1, mbets):
                phi = 1
                for k in range(mputs):
                    num = mtx[j - 1, k]
                    if num > 0:
                        nid = int(num - 1)
                        phi = phi * (phis[nid][0][phind[i, k]] + phis[nid][1][phind[i, k]] * xsm[i, k] + \
                            phis[nid][2][phind[i, k]] * xsm[i, k] ** 2 + phis[nid][3][phind[i, k]] * xsm[i, k] ** 3)

        # End # v3.1.0 update
        # ==================================================================

                X[i, j] = phi

        X[:, 0] = np.ones((n,))
        modells = np.zeros((n, draws))  # note n == np.shape(data)[0] if data != 'ignore'
        for i in range(draws):
            modells[:, i] = np.matmul(X, betas[setnos[i], :])
        meen = np.mean(modells, 1)

        if ReturnBounds:
            bounds = np.zeros((n, 2))  # note n == np.shape(data)[0] if data != 'ignore'
            cut = int(np.floor(draws * .025))
            for i in range(n):  # note n == np.shape(data)[0] if data != 'ignore'
                drawset = np.sort(modells[i, :])
                bounds[i, 0] = drawset[cut]
                bounds[i, 1] = drawset[draws - cut]
            return meen, bounds
        else:
            return meen


    def coverage3(self, **kwargs):
        """
        For validation testing of FoKL model.

        Keyword Inputs:
            inputs == normalized and properly formatted inputs to evaluate              == self.inputs (default)
            data   == properly formatted data outputs to use for validating predictions == self.data (default)
            draws  == number of beta terms used                                         == self.draws (default)

        Keyword Inputs for Plotting:
            plot   == binary for generating plot, or 'sorted' for plot of ordered data == 0 (default)
            bounds == binary for plotting bounds                                       == 0 (default)
            xaxis  == vector to plot on x-axis                                         == indices (default)
            labels == binary for adding labels to plot                                 == 1 (default)
            xlabel == string for x-axis label                                          == 'Index' (default)
            ylabel == string for y-axis label                                          == 'Data' (default)
            title  == string for plot title                                            == 'FoKL' (default)
            legend == binary for adding legend to plot                                 == 0 (default)

        Additional Keyword Inputs for Plotting:
            PlotTypeFoKL      == string for FoKL's color and line type  == 'b' (default)
            PlotSizeFoKL      == scalar for FoKL's line size            == 2 (default)
            PlotTypeBounds    == string for Bounds' color and line type == 'k--' (default)
            PlotSizeBounds    == scalar for Bounds' line size           == 2 (default)
            PlotTypeData      == string for Data's color and line type  == 'ro' (default)
            PlotSizeData      == scalar for Data's line size            == 2 (default)
            LegendLabelFoKL   == string for FoKL's label in legend      == 'FoKL' (default)
            LegendLabelData   == string for Data's label in legend      == 'Data' (default)
            LegendLabelBounds == string for Bounds's label in legend    == 'Bounds' (default)

        Return Outputs:
            meen   == predicted output values for each indexed input
            rmse   == root mean squared deviation (RMSE) of prediction versus known data
            bounds == confidence interval for each predicted output value
        """

        def process_kwargs(kwargs):
            # Default keywords:
            kwargs_upd = {'inputs': 'default', 'data': 'default', 'draws': self.draws, 'plot': 0, 'bounds': 0, 'xaxis': 0, 'labels': 1, 'xlabel': 'Index', 'ylabel': 'Data', 'title': 'FoKL', 'legend': 0, 'PlotTypeFoKL': 'b', 'PlotSizeFoKL': 2, 'PlotTypeBounds': 'k--', 'PlotSizeBounds': 2, 'PlotTypeData': 'ro', 'PlotSizeData': 2, 'LegendLabelFoKL': 'FoKL', 'LegendLabelData': 'Data', 'LegendLabelBounds': 'Bounds'}

            # Update keywords based on user-input:
            kwargs_expected = kwargs_upd.keys()
            for kwarg in kwargs.keys():
                if kwarg not in kwargs_expected:
                    raise ValueError(f"Unexpected keyword argument: {kwarg}")
                else:
                    kwargs_upd[kwarg] = kwargs.get(kwarg, kwargs_upd.get(kwarg))

            # Define default values if no user-input, or if only 'inputs' is user-defined:
            if isinstance(kwargs_upd.get('data'), str): # if 'data' is default
                if not isinstance(kwargs_upd.get('inputs'), str): # if 'inputs' is NOT default
                    kwargs_upd['data'] = 'ignore' # do not plot 'data'
                    warnings.warn("Keyword argument 'data' must be defined for user-defined 'inputs' if intending to plot 'data'.")
                elif kwargs_upd.get('data').lower() == 'default': # confirm 'data' is default
                    kwargs_upd['data'] = self.data # default ... train+test data, i.e., validation
                    if not isinstance(kwargs_upd.get('inputs'), str): # if 'inputs' is not default
                        warnings.warn("Keyword argument 'data' should be defined for user-defined 'inputs'. Assuming default value for 'data' which may error if 'inputs' is not also default.")
                else:
                    raise ValueError("Keyword argument 'data' was improperly defined.")
            else: # if 'data' is user-defined
                if isinstance(kwargs_upd.get('inputs'), str): # if 'inputs' is default
                    if kwargs_upd.get('inputs').lower() == 'default': # confirm 'inputs' is default
                        raise ValueError("Keyword argument 'inputs' must be defined for user-defined 'data'.")
            if isinstance(kwargs_upd.get('inputs'), str): # if 'inputs' is default
                if kwargs_upd.get('inputs').lower() == 'default': # confirm 'inputs' is default
                    kwargs_upd['inputs'] = self.inputs # default ... train+test inputs, i.e., validation

            plots = kwargs_upd.get('plot') # 0 by default
            raise_error = 0
            if isinstance(plots, str):
                plots = plots.lower()
                if plots in ['yes', 'on', 'y','true']:
                    kwargs_upd['plot'] = 1
                elif plots in ['no', 'none', 'off', 'n','false']:
                    kwargs_upd['plot'] = 0
                elif plots in ['sort', 'sorted']:
                    kwargs_upd['plot'] = 'sorted'
                    if kwargs_upd['xlabel'] == 'Index': # if default value, i.e., no user-defined xlabel
                        kwargs_upd['xlabel'] = 'Index (Sorted)'
                elif plots in ['bounds', 'bound', 'bounded']:
                    kwargs_upd['plot'] = 1
                    kwargs_upd['bounds'] = 1
                else:
                    raise_error = 1
            elif plots not in (0, 1):
                raise_error = 1
            if raise_error:
                raise ValueError(f"Optional keyword argument 'plot' is limited to 0, 1, or 'sorted'.")

            xaxis = kwargs_upd['xaxis']
            if isinstance(xaxis, int): # then default
                if xaxis != 0: # if not default, i.e., user-defined
                    if isinstance(xaxis, str):
                        xaxis = xaxis.lower()
                        if xaxis in ['true', 'actual', 'denorm', 'denormed', 'denormalized']:
                            kwargs_upd['xaxis'] = 1 # assume first input variable is xaxis (e.g., time)
                        elif xaxis in ['indices', 'none', 'norm', 'normed', 'normalized']:
                            kwargs_upd['xaxis'] = 0
                    elif not isinstance(xaxis, int):
                        xaxis = np.squeeze(np.array(xaxis))
                        if xaxis.ndim == 1: # then not a vector
                            raise ValueError("Keyword argument 'xaxis' is limited to an integer corresponding to the input variable to plot along the xaxis (e.g., 1, 2, 3, etc.) or to a vector corresponding to the user-provided data. Leave blank or =0 to plot indices along the xaxis.")
                        else:
                            kwargs_upd['xaxis'] = xaxis # update as a numpy vector

            plt_labels = kwargs_upd['labels']
            if plt_labels != 1: # if not default, i.e., user-defined
                if isinstance(plt_labels, str):
                    plt_labels = plt_labels.lower()
                    if plt_labels in ['no', 'off', 'hide', 'none', 'false']:
                        kwargs_upd['labels'] = 0
                    elif plt_labels in ['yes', 'on', 'show', 'all', 'plot', 'true' 'y']:
                        kwargs_upd['labels'] = 1
                elif plt_labels != 0:
                    raise ValueError("Keyword argument 'labels' is limited to 0 or 1.")

            plt_legend = kwargs_upd['legend']
            if plt_legend != 0:  # if not default, i.e., user-defined
                if isinstance(plt_legend, str):
                    plt_legend = plt_legend.lower()
                    if plt_legend in ['y', 'yes', 'on', 'show', 'all', 'plot', 'true']:
                        kwargs_upd['legend'] = 1
                    elif plt_legend in ['n', 'no', 'off', 'hide', 'none', 'false']:
                        kwargs_upd['legend'] = 0
                elif plt_legend != 1:
                    raise ValueError("Keyword argument 'legend' is limited to 0 or 1.")

            if not all(isinstance(kwargs_upd[key], str) for key in ['LegendLabelFoKL', 'LegendLabelData', 'LegendLabelBounds']):
                raise ValueError("Keyword arguments 'LegendLabelFoKL', 'LegendLabelData',and 'LegendLabelBounds' are limited to strings.")

            return kwargs_upd
        kwargs_upd = process_kwargs(kwargs)
        normputs = kwargs_upd.get('inputs')
        data = kwargs_upd.get('data')
        draws = kwargs_upd.get('draws')

        meen, bounds = self.evaluate(normputs, draws=draws, nform=0, ReturnBounds=1)
        n, mputs = np.shape(normputs)  # Size of normalized inputs ... calculated in 'evaluate' but not returned

        if kwargs_upd.get('plot') != 0: # if user requested a plot
            plt_x = kwargs_upd.get('xaxis') # 0, integer indexing input variable to plot, or user-defined vector
            linspace_needed_if_sorted = 1
            if isinstance(plt_x, int): # if not user-defined vector
                if plt_x != 0: # if user specified x-axis
                    if isinstance(plt_x, int): # if user-specified an input variable (i.e., not a vector)
                        warnings.warn("Using default inputs for defining x-axis. Set keyword argument 'xaxis' equal to a vector if using custom inputs.", category=UserWarning)
                        minmax = self.normalize
                        min = minmax[plt_x - 1][0]
                        max = minmax[plt_x - 1][1]
                        inputs_np = self.inputs_np
                        plt_x = inputs_np[:, plt_x-1] * (max - min) + min # vector to plot on x-axis
                else: # if plt_x == 0
                    plt_x = np.linspace(0, n - 1, n)
                    linspace_needed_if_sorted = 0

            if kwargs_upd.get('plot') == 'sorted': # if user requested a sorted plot
                if isinstance(kwargs_upd.get('data'), str):
                    if kwargs_upd.get('data').lower() == 'ignore':
                        raise ValueError("Keyword argument 'data' must be defined if using a sorted plot with user-defined 'inputs'.")
                sort_id = np.squeeze(np.argsort(data, axis=0))
                plt_meen = meen[sort_id]
                plt_bounds = bounds[sort_id]
                plt_data = data[sort_id]
                if linspace_needed_if_sorted:
                    plt_x = np.linspace(0, len(data) - 1, len(data))
                    warnings.warn("Keyword argument 'xaxis' was overwritten. For a sorted plot, the x-axis is of arbitrary indices.", category=UserWarning)
            else:
                plt_meen = meen
                plt_data = data
                plt_bounds = bounds

            plt.figure()
            plt.plot(plt_x, plt_meen, kwargs_upd.get('PlotTypeFoKL'), linewidth=kwargs_upd.get('PlotSizeFoKL'), label=kwargs_upd.get('LegendLabelFoKL'))
            if not isinstance(kwargs_upd.get('data'), str): # else kwargs_upd.get('data').lower() == 'ignore':
                plt.plot(plt_x, plt_data, kwargs_upd.get('PlotTypeData'), markersize=kwargs_upd.get('PlotSizeData'), label=kwargs_upd.get('LegendLabelData'))
            if kwargs_upd.get('bounds'):
                plt.plot(plt_x, plt_bounds[:, 0], kwargs_upd.get('PlotTypeBounds'), linewidth=kwargs_upd.get('PlotSizeBounds'), label=kwargs_upd.get('LegendLabelBounds'))
                plt.plot(plt_x, plt_bounds[:, 1], kwargs_upd.get('PlotTypeBounds'), linewidth=kwargs_upd.get('PlotSizeBounds'))
            if kwargs_upd.get('labels'):
                if not all(isinstance(kwargs_upd.get(key), str) for key in ['xlabel', 'ylabel', 'title']):
                    if any(kwargs_upd.get(key) != 0 for key in ['xlabel', 'ylabel', 'title']):
                        raise ValueError("Keyword arguments 'xlabel', 'ylabel', and 'title' are limited to strings.")
                if kwargs_upd.get('xlabel') != 0:
                    plt.xlabel(kwargs_upd.get('xlabel'))
                if kwargs_upd.get('ylabel') != 0:
                    plt.ylabel(kwargs_upd.get('ylabel'))
                if kwargs_upd.get('title') != 0:
                    plt.title(kwargs_upd.get('title'))

            if kwargs_upd.get('legend'):
                plt.legend()

            plt.show()

        if isinstance(kwargs_upd.get('data'), str): # then assume kwargs_upd.get('data').lower() == 'ignore':
            warnings.warn("Keyword argument 'data' must be defined for user-defined 'inputs' in order to calculate RMSE.")
            rmse = []
        else:
            rmse = np.sqrt(np.mean(meen - data) ** 2)

        return meen, bounds, rmse


    def fit(self, inputs, data, **kwargs):
        """
        For fitting model to known inputs and data (i.e., training of model).

        Inputs:
            inputs == NxM matrix of independent (or non-linearly dependent) 'x' variables for fitting f(x1, ..., xM)
            data   == Nx1 vector of dependent variable to create model for predicting the value of f(x1, ..., xM)

        Keyword Inputs:
            train                == percentage (0-1) of N datapoints to use for training  == 1 (default)
            TrainMethod          == method for splitting test/train set for train < 1     == 'random' (default)
            CatchOutliers        == logical for removing outliers from inputs and/or data == 0 (default)
            OutliersMethod       == string defining the method to use for removing outliers (e.g., 'Z-Score)
            OutliersMethodParams == parameters to modify OutliersMethod (format varies per method)

        Return Outputs:
            'betas' are a draw from the posterior distribution of coefficients: matrix, with
            rows corresponding to draws and columns corresponding to terms in the GP

            'mtx' is the basis function interaction matrix from the
            best model: matrix, with rows corresponding to terms in the GP (and thus to the
            columns of 'betas' and columns corresponding to inputs. a given entry in the
            matrix gives the order of the basis function appearing in a given term in the GP.
            all basis functions indicated on a given row are multiplied together.
            a zero indicates no basis function from a given input is present in a given term

            'ev' is a vector of BIC values from all of the models
            evaluated

        Added Attributes:
            > 'inputs' and 'data' get automatically formatted, cleaned, reduced to a train set, and stored as:
                > model.inputs         == all normalized inputs w/o outliers (i.e., model.traininputs plus
                                          model.testinputs)
                > model.data           == all data w/o outliers (i.e., model.traindata plus model.testdata)
            > other useful info related to 'inputs' and 'data' get stored as:
                > model.rawinputs      == all normalized inputs w/ outliers == user's 'inputs' but normalized and
                                                                               formatted
                > model.rawdata        == all data w/ outliers              == user's 'data' but formatted
                > model.traininputs    == train set of model.inputs
                > model.traindata      == train set of model.data
                > model.testinputs     == test set of model.inputs
                > model.testdata       == test set of model.data
                > model.normalize      == [min, max] factors used to normalize user's 'inputs' to 0-1 scale of
                                          model.rawinputs
                > model.outliers       == indices removed from model.rawinputs and model.rawdata as outliers
                > model.trainlog       == indices of model.inputs used for model.traininputs
                > model.testlog        == indices of model.data used for model.traindata
            > to access numpy versions of the above attributes related to 'inputs', use:
                > model.inputs_np      == model.inputs as a numpy array of timestamps x input variables
                > model.rawinputs_np   == model.rawinputs as a numpy array of timestamps x input variables
                > model.traininputs_np == model.traininputs as a numpy array of timestamps x input variables
                > model.testinputs_np  == model.testinputs as a numpy array of timestamps x input variables
        """

        # Check all keywords and update hypers if re-defined by user:
        kwargs_hypers = ['phis', 'relats_in', 'a', 'b', 'atau', 'btau', 'tolerance', 'draws', 'gimmie', 'way3',
                         'threshav', 'threshstda', 'threshstdb', 'aic']
        kwargs_for_fit = ['train', 'TrainMethod', 'CatchOutliers', 'OutliersMethod', 'OutliersMethodParams']
        kwargs_expected = kwargs_hypers + kwargs_for_fit
        for kwarg in kwargs.keys():
            if kwarg not in kwargs_expected:
                raise ValueError(f"Unexpected keyword argument: {kwarg}")
            elif kwarg in kwargs_hypers: # then update hyper as attribute
                setattr(self, kwarg, kwargs.get(kwarg))

        # Process keywords specific to fit():
        p_train = 1 # default
        p_train_method = 'random' # default
        if 'train' in kwargs:
            p_train = kwargs.get('train')
            if 'TrainMethod' in kwargs:
                p_train_method = kwargs.get('TrainMethod')
        CatchOutliers = 0  # default
        OutliersMethod = []  # default
        OutliersMethodParams = [] # default
        if 'CatchOutliers' in kwargs:
            CatchOutliers = kwargs.get('CatchOutliers')
            if 'OutliersMethod' in kwargs:
                OutliersMethod = kwargs.get('OutliersMethod')
                if 'OutliersMethodParams' in kwargs:
                    OutliersMethodParams = kwargs.get('OutliersMethodParams')
            if isinstance(CatchOutliers, str): # convert user input to logical for auto_cleanData to interpret
                if CatchOutliers.lower() in ('all', 'both', 'yes', 'y', 'true'):
                    CatchOutliers = 1
                elif CatchOutliers.lower() in ('none','no', 'n', 'false'):
                    CatchOutliers = 0
                elif CatchOutliers.lower() in ('inputs', 'inputsonly', 'input', 'inputonly'):
                    CatchOutliers = [1, 0] # note 1 will need to be replicated later to match number of input variables
                elif CatchOutliers.lower() in ('data', 'dataonly', 'outputs', 'outputsonly', 'output', 'outputonly'):
                    CatchOutliers = [0, 1] # note 0 will need to be replicated later to match number of input variables
            elif isinstance(CatchOutliers, np.ndarray): # assume 1D list if not, which is the goal
                if CatchOutliers.ndim == 1:
                    CatchOutliers = CatchOutliers.to_list() # should return 1D list
                else:
                    CatchOutliers = np.squeeze(CatchOutliers)
                    if CatchOutliers.ndim != 1:
                        raise ValueError(
                            "CatchOutliers, if being applied to a user-selected inputs+data combo, must be a logical "
                            "1D list (e.g., [0,1,...,1,0]) corresponding to [input1, input2, ..., inputM, data].")
                    else:
                        CatchOutliers = CatchOutliers.to_list()  # should return 1D list
            elif not isinstance(CatchOutliers, list):
                raise ValueError("CatchOutliers must be defined as 'Inputs', 'Data', 'All', or a logical 1D list "
                                 "(e.g., [0,1,...,1,0]) corresponding to [input1, input2, ..., inputM, data].")
        # note at this point CatchOutliers might be 0, 1, [1, 0], [0, 1, 0, 0], etc.

        # Automatically handle some data formatting exceptions:
        def auto_cleanData(inputs, data, p_train, CatchOutliers, OutliersMethod, OutliersMethodParams):

            # Convert 'inputs' and 'datas' to numpy if pandas:
            if any(isinstance(inputs, type) for type in (pd.DataFrame, pd.Series)):
                inputs = inputs.to_numpy()
                warnings.warn(
                    "'inputs' was auto-converted to numpy. Convert manually for assured accuracy.", UserWarning)
            if any(isinstance(data, type) for type in (pd.DataFrame, pd.Series)):
                data = data.to_numpy()
                warnings.warn("'data' was auto-converted to numpy. Convert manually for assured accuracy.", UserWarning)

            # Normalize 'inputs' and convert to proper format for FoKL:
            inputs = np.array(inputs) # attempts to handle lists or any other format (i.e., not pandas)
            # . . . inputs = {ndarray: (N, M)} = {ndarray: (datapoints, input variables)} =
            # . . . . . . array([[x1(t1),x2(t1),...,xM(t1)],[x1(t2),x2(t2),...,xM(t2)],...,[x1(tN),x2(tN),...,xM(tN)]])
            inputs = np.squeeze(inputs) # removes axes with 1D for cases like (N x 1 x M) --> (N x M)
            if inputs.ndim == 1:  # if inputs.shape == (number,) != (number,1), then add new axis to match FoKL format
                inputs = inputs[:, np.newaxis]
            N = inputs.shape[0]
            M = inputs.shape[1]
            if M > N: # if more "input variables" than "datapoints", assume user is using transpose of proper format
                inputs = inputs.transpose()
                warnings.warn(
                    "'inputs' was transposed. Ignore if more datapoints than input variables.", category=UserWarning)
                N_old = N
                N = M # number of datapoints (i.e., timestamps)
                M = N_old # number of input variables
            inputs_max = np.max(inputs, axis=0) # max of each input variable
            inputs_scale = []
            for ii in range(len(inputs_max)):
                inputs_min = np.min(inputs[:, ii])
                if inputs_max[ii] != 1 or inputs_min != 0:
                    if inputs_min == inputs_max[ii]:
                        inputs[:,ii] = np.ones(len(inputs[:,ii]))
                        warnings.warn("'inputs' contains a column of constants which will not improve the model's fit."
                                      , category=UserWarning)
                    else: # normalize
                        inputs[:,ii] = (inputs[:,ii] - inputs_min) / (inputs_max[ii] - inputs_min)
                inputs_scale.append(np.array([inputs_min, inputs_max[ii]]))  # store for post-processing convenience
            inputs = inputs.tolist() # convert to list, which is proper format for FoKL, like:
            # . . . {list: N} = [[x1(t1),x2(t1),...,xM(t1)],[x1(t2),x2(t2),...,xM(t2)],...,[x1(tN),x2(tN),...,xM(tN)]]

            # Transpose 'data' if needed:
            data = np.array(data)  # attempts to handle lists or any other format (i.e., not pandas)
            if data.ndim == 1:  # if data.shape == (number,) != (number,1), then add new axis to match FoKL format
                data = data[:, np.newaxis]
                warnings.warn("'data' was made into (n,1) column vector from single list (n,) to match FoKL formatting."
                    , category=UserWarning)
            else: # check user provided only one output column/row, then transpose if needed
                N_data = data.shape[0]
                M_data = data.shape[1]
                if (M_data != 1 and N_data != 1) or (M_data == 1 and N_data == 1):
                    raise ValueError("Error: 'data' must be a vector.")
                elif M_data != 1 and N_data == 1:
                    data = data.transpose()
                    warnings.warn("'data' was transposed to match FoKL formatting.",category=UserWarning)

            # Store properly formatted data and normalized inputs BEFORE removing outliers and BEFORE splitting train
            rawinputs = inputs
            rawdata = data

            # Catch and remove outliers:
            if CatchOutliers == [1,0]: # i.e., inputs only
                CatchOutliers = list(np.ones(M))+[0] # [1,1,...,1,0] as a list
            elif CatchOutliers == [0,1]: # i.e., data only
                CatchOutliers = list(np.zeros(M))+[1] # [0,0,...,0,1] as a list
            elif CatchOutliers == 1: # i.e., all
                CatchOutliers = list(np.ones(M+1)) # [1,1,...,1,1] as a list
            elif CatchOutliers == 0: # i.e., none
                CatchOutliers = list(np.zeros(M+1)) # [0,0,...,0,0] as a list
            elif len(CatchOutliers) != M+1:
                raise ValueError(
                    "CatchOutliers must be defined as 'Inputs', 'Data', 'All', or a logical 1D list (e.g., "
                    "'[0,1,...,1,0]) corresponding to [input1, input2, ..., inputM, data].")
            def catch_outliers(inputs, data, CatchOutliers, OutliersMethod, OutliersMethodParams):
                inputs_wo_outliers = inputs
                data_wo_outliers = data
                outliers_indices = [] # if logical true/false, then use np.zeros(len(data))
                if OutliersMethod != []:
                    CatchOutliers_np = np.array(CatchOutliers)
                    CatchOutliers_id = np.where(CatchOutliers_np == 1)[0]
                    inputs_data = np.hstack((inputs, data))
                    inputs_data_rel = inputs_data[:, CatchOutliers_id]

                    from scipy import stats

                    if OutliersMethod == 'Z-Score':
                        z_scores = np.abs(stats.zscore(inputs_data_rel))
                        if OutliersMethodParams != []: # if user-defined
                            threshold = OutliersMethodParams
                        else:
                            threshold = 3 # default value if threshold of z-score is not user-specified
                        outliers_indices = np.any(np.where(z_scores > threshold, True, False), axis=1)

                    elif OutliersMethod == 'other': # maybe ... Interquartile Range
                        outliers_indices = np.ones_like(inputs_data).astype(bool) # stand-in until future development

                    elif OutliersMethod == 'other': # maybe ... Kernel Density Estimation (KDE) ... can be multivariate
                        outliers_indices = np.ones_like(inputs_data).astype(bool) # stand-in until future development

                    elif OutliersMethod == 'other': # maybe ... Mahalanobis Distance ... can be multivariate
                        outliers_indices = np.ones_like(inputs_data).astype(bool) # stand-in until future development

                    elif OutliersMethod == 'other': # maybe ... Local Outlier Factor (LOF)
                        outliers_indices = np.ones_like(inputs_data).astype(bool) # stand-in until future development

                    elif OutliersMethod != []:
                        raise ValueError("Keyword argument 'OutliersMethod' is limited to 'Z-Score'. Other methods are "
                                         "in development.")

                    inputs_data_wo_outliers = inputs_data[~outliers_indices, :]
                    inputs_wo_outliers = inputs_data_wo_outliers[:, :-1]
                    data_wo_outliers = inputs_data_wo_outliers[:, -1]

                return inputs_wo_outliers, data_wo_outliers, outliers_indices
            inputs, data, outliers_indices = \
                catch_outliers(inputs, data, CatchOutliers, OutliersMethod, OutliersMethodParams)

            # Spit [inputs,data] into train and test sets (depending on TrainMethod):
            if p_train < 1: # split inputs+data into training and testing sets for validation of model
                def random_train(p_train, inputs, data): # random split, if TrainMethod = 'random'
                    Ldata = len(data)
                    train_log = np.random.rand(Ldata) < p_train # indices to use as training data
                    test_log = ~train_log

                    inputs_train = [inputs[ii] for ii, ele in enumerate(train_log) if ele]
                    data_train = data[train_log]
                    inputs_test = [inputs[ii] for ii, ele in enumerate(test_log) if ele]
                    data_test = data[test_log]

                    return inputs_train, data_train, inputs_test, data_test, train_log, test_log

                def other_train(p_train, inputs, data): # IN DEVELOPMENT ... other split, if TrainMethod = 'other'
                    # WRITE CODE HERE FOR NEW METHOD OF SPLITTING TEST/TRAIN SETS
                    inputs_train = inputs
                    data_train = data
                    inputs_test = []
                    data_test = []
                    train_log = np.linspace(0, len(inputs[:, 0]) - 1, len(inputs[:, 0]))
                    test_log = []

                    return inputs_train, data_train, inputs_test, data_test, train_log, test_log

                def otherN_train(p_train, inputs, data): # IN DEVELOPMENT ... otherN split, if TrainMethod = 'otherN'
                    # WRITE CODE HERE FOR NEW METHOD OF SPLITTING TEST/TRAIN SETS
                    inputs_train = inputs
                    data_train = data
                    inputs_test = []
                    data_test = []
                    train_log = np.linspace(0, len(inputs[:,0]) - 1, len(inputs[:,0]))
                    test_log = []

                    return inputs_train, data_train, inputs_test, data_test, train_log, test_log

                function_mapping = {'random': random_train,'other': other_train,'otherN': otherN_train}
                if p_train_method in function_mapping:
                    inputs_train, data_train, inputs_test, data_test, train_log, test_log = \
                        function_mapping[p_train_method](p_train, inputs, data)
                else:
                    raise ValueError("Keyword argument 'TrainMethod' is limited to 'random' as of now. Additional "
                        "methods of splitting are in development.")

            else:
                inputs_train = inputs
                data_train = data
                inputs_test = []
                data_test = []
                train_log = []
                test_log = []

            return inputs, data, rawinputs, rawdata, inputs_train, data_train, inputs_test, data_test, inputs_scale, \
                   outliers_indices, train_log, test_log

        inputs, data, rawinputs, rawdata, traininputs, traindata, testinputs, testdata, inputs_scale, \
            outliers_indices, train_log, test_log = auto_cleanData(inputs, data, p_train, CatchOutliers,
                                                                   OutliersMethod, OutliersMethodParams)

        def inputslist_to_np(inputslist, do_transpose):
            was_auto_transposed = 0
            if np.any(inputslist): # if inputslist is not empty (i.e., != [] )
                inputslist_np = np.array(inputslist) # should be N datapoints x M inputs
                NM = np.shape(inputslist_np)
                if NM[0] < NM[1] and do_transpose == 'auto':
                    inputslist_np = np.transpose(inputslist_np)
                    was_auto_transposed = 1
                elif do_transpose == 1:
                    inputslist_np = np.transpose(inputslist_np)
            else:
                inputslist_np = np.array([])
            return inputslist_np, was_auto_transposed

        # Define/update attributes with cleaned data and other relevant variables:
        setattr(self, 'inputs', inputs)
        setattr(self, 'data', data)
        setattr(self, 'rawinputs', rawinputs)
        setattr(self, 'traininputs', traininputs)
        setattr(self, 'traindata', traindata)
        setattr(self, 'testinputs', testinputs)
        setattr(self, 'testdata', testdata)
        setattr(self, 'normalize', inputs_scale) # [min,max] of each input before normalization
        setattr(self, 'outliers', outliers_indices) # indices removed from raw
        setattr(self, 'trainlog', train_log) # indices used for training AFTER OUTLIERS WERE REMOVED from raw
        setattr(self, 'testlog', test_log) # indices used for validation testing AFTER OUTLIERS WERE REMOVED from raw
        inputs_np, do_transpose = inputslist_to_np(self.inputs, 'auto')
        setattr(self, 'inputs_np', inputs_np)
        setattr(self, 'rawinputs_np', inputslist_to_np(self.rawinputs, do_transpose)[0])
        setattr(self, 'traininputs_np', inputslist_to_np(self.traininputs, do_transpose)[0])
        setattr(self, 'testinputs_np', inputslist_to_np(self.testinputs, do_transpose)[0])

        # Initializations:
        phis = self.phis
        relats_in = self.relats_in
        a = self.a
        b = self.b
        atau = self.atau
        btau = self.btau
        tolerance = self.tolerance
        draws = self.draws
        gimmie = self.gimmie
        way3 = self.way3
        threshav = self.threshav
        threshstda = self.threshstda
        threshstdb = self.threshstdb
        aic = self.aic

        # Update 'b' and/or 'btau' if set to default:
        if btau == 'default' or b == 'default':  # if not user-defined then use 'data' and ('a' and/or 'atau') to define
            sigmasq = np.var(data)
            if b == 'default':
                b = sigmasq * (a + 1)
                self.b = b
            if btau == 'default':
                scale = np.abs(np.mean(data))
                btau = (scale / sigmasq) * (atau + 1)
                self.btau = btau

        def perms(x):
            """Python equivalent of MATLAB perms."""
            # from https://stackoverflow.com/questions/38130008/python-equivalent-for-matlabs-perms
            a = np.vstack(list(itertools.permutations(x)))[::-1]
            return a

        def gibbs(inputs, data, phis, Xin, discmtx, a, b, atau, btau, draws):
            """
            'inputs' is the set of normalized inputs -- both parameters and model
            inputs -- with columns corresponding to inputs and rows the different
            experimental designs. (numpy array)

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

            phind, xsm = self.inputs_to_phind(inputs=inputs, phis=phis)  # v3.1.0 update

            for i in range(minp):

                # # =================================================================
                # # Outdated as of v3.1.0 because results differ from MATLAB results:
                #
                # phind = []
                # for j in range(len(inputs[i])):
                #     phind.append(math.ceil(inputs[i][j] * 498))
                #     # 499 changed to 498 for python indexing
                #
                # phind_logic = []
                # for k in range(len(phind)):
                #     if phind[k] == 0:
                #         phind_logic.append(1)
                #     else:
                #         phind_logic.append(0)
                #
                # phind = np.add(phind, phind_logic)
                #
                # # End
                # # =================================================================

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
                            nid = int(num - 1)
                            phi = phi * (phis[nid][0][phind[i, k]] + phis[nid][1][phind[i, k]] * xsm[i, k] + \
                                phis[nid][2][phind[i, k]] * xsm[i, k] ** 2 + phis[nid][3][phind[i, k]] * xsm[i, k] ** 3)

                    X[i][j] = phi

            # initialize tausqd at the mode of its prior: the inverse of the mode of sigma squared, such that the
            # initial variance for the betas is 1
            sigsqd = b / (1 + a)
            tausqd = btau / (1 + atau)

            XtX = np.transpose(X).dot(X)

            Xty = np.transpose(X).dot(data)

            # See the link:
            #     - "https://stackoverflow.com/questions/8765310/scipy-linalg-eig-return-complex-eigenvalues-for-
            #        covariance-matrix"
            Lamb, Q = eigh(XtX)  # using scipy eigh function to avoid imaginary values due to numerical errors
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


                bstar = b + 0.5 * (betas[k][:].dot(XtX.dot(np.transpose([betas[k][:]]))) - 2 * betas[k][:].dot(Xty) + \
                                   dtd + betas[k][:].dot(np.transpose([betas[k][:]])) / tausqd)
                # bstar = b + comp1.dot(comp2) + 0.5 * dtd - comp3;

                # Returning a 'not a number' constant if bstar is negative, which would
                # cause np.random.gamma to return a ValueError
                if bstar < 0:
                    sigsqd = math.nan
                else:
                    sigsqd = 1 / np.random.gamma(astar, 1 / bstar)

                sigs[k] = sigsqd

                btau_star = (1/(2*sigsqd)) * (betas[k][:].dot(np.reshape(betas[k][:], (len(betas[k][:]), 1)))) + btau

                tausqd = 1 / np.random.gamma(atau_star, 1 / btau_star)
                taus[k] = tausqd

            # Calculate the evidence
            siglik = np.var(data - np.matmul(X, betahat))

            lik = -(n / 2) * np.log(siglik) - (n - 1) / 2
            ev = (mmtx + 1) * np.log(n) - 2 * np.max(lik)

            X = X[:, 0:mmtx + 1]

            return betas, sigs, taus, betahat, X, ev



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

                [beters, null, null, null, xers, ev] = gibbs(inputs_np, data, phis, X, damtx, a, b, atau, btau, draws)

                if aic:
                    ev = ev + (2 - np.log(n)) * (dam + 1)

                betavs = np.abs(np.mean(beters[int(np.ceil((draws / 2)+1)):draws, (dam - vm + 1):dam+1], axis=0))
                betavs2 = np.divide(np.std(np.array(beters[int(np.ceil(draws/2)+1):draws, dam-vm+1:dam+1]), axis=0), \
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


                    # if betavs[i, 1] > threshstdb or betavs[i, 1] > threshstda and betavs[i, 0] < threshav * \
                    #     np.mean(np.abs(np.mean(beters[int(np.ceil(draws/2 +1)):draws, 0]))):
                    if betavs[i, 1] > threshstdb or betavs[i, 1] > threshstda and betavs[i, 0] < threshav * \
                        np.mean(np.abs(np.mean(beters[int(np.ceil(draws/2)):draws, 0]))):  # index to 'beters' \
                        # adjusted for matlab to python [JPK DEV v3.1.0 20240129]

                        killtest = np.append(killset, (betavs[i, 2] - 1))
                        if killtest.size > 1:
                            killtest[::-1].sort()  # max to min so damtx_test rows get deleted in order of end to start
                        damtx_test = damtx
                        for k in range(0, np.size(killtest)):
                            damtx_test = np.delete(damtx_test, int(np.array(killtest[k])-1), 0)
                        damtest, null = np.shape(damtx_test)

                        [betertest, null, null, null, Xtest, evtest] = gibbs(inputs_np, data, phis, X, damtx_test, a, b,
                                                                             atau, btau, draws)
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

        self.betas = betas
        self.mtx = mtx
        self.evs = evs

        return betas, mtx, evs


    def clear(self, **kwargs):
        """
        Delete all attributes from the FoKL class except for hyperparameters and any specified by the user.

        Keyword Inputs:
            Assign any value to a keyword representing the attribute that should be kept.

            For example, to keep only the hyperparameters, self.inputs_np, and self.normalize, use:
            > self.clear(inputs_np=1, normalize=1)
        """
        attrs_to_keep = ['phis', 'relats_in', 'a', 'b', 'atau', 'btau', 'tolerance', 'draws', 'gimmie', 'way3',
                         'threshav', 'threshstda', 'threshstdb', 'aic']
        for kwarg in kwargs.keys():
            if isinstance(kwarg, str):
                attrs_to_keep = attrs_to_keep + [kwarg] # append user-specified attribute to list of what not to delete
            else:
                warnings.warn(f"The user-specified attribute, {kwarg}, must be a string.",category=UserWarning)
        attrs = list(vars(self).keys()) # list of all currently defined attributes
        for attr in attrs:
            if attr not in attrs_to_keep:
                try:
                    delattr(self, attr)
                except:
                    warnings.warn(
                        "The requested attribute, {attr}, is not defined and so was ignored when attempting to delete.",
                        category=UserWarning)

