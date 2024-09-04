from FoKL import getKernels
import pandas as pd
import warnings
import itertools
import math
# import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad

import jax.numpy as jnp
from jax import grad
from jax import random
from jax import jit
from jax.lax import fori_loop, cond, while_loop
from jax import ops as jops
from jax import lax
import jax
from jax.scipy.special import gamma

def inverse_gamma_pdf(y, alpha, beta):
    """
    Calculate the PDF of the inverse gamma distribution.

    Parameters:
    y (float or array-like): The value(s) at which to evaluate the PDF.
    alpha (float): Shape parameter of the inverse gamma distribution.
    beta (float): Scale parameter of the inverse gamma distribution.

    Returns:
    float or array-like: The PDF value(s) at y.
    """
    # Ensure y is a JAX array for compatibility
    y = jnp.array(y)
    
    # Calculate the PDF of the inverse gamma distribution
    pdf = (beta ** alpha / gamma(alpha)) * y ** (-alpha - 1) * jnp.exp(-beta / y)
    return pdf

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
        
        def hmc(inputs, data, phis, Xin, discmtx, a, b, atau, btau, draws):
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

            def inputs_to_phind(inputs, phis):
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

                # self.phind = phind - 1  # adjust MATLAB indexing to Python indexing after twice normalization
                # self.xsm = L_phis * inputs - phind

                return phind - 1, L_phis * inputs - phind

            def neg_log_likelihood(betahat):
                Xin = []
                
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

                phind, xsm = inputs_to_phind(inputs=inputs, phis=phis)  # v3.1.0 update

                for i in range(minp):

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

                n = len(data)

                prediction = np.matmul(X, betahat)

                error = data.T - prediction

                variance = 0.1

                lik = np.sum(0.5 * np.log(2 * math.pi * variance) + (error**2 / (2 * variance)))

                X = X[:, 0:mmtx + 1]

                return lik

            d_neg_log_likelihood = grad(neg_log_likelihood)

            def HMC(U, grad_U, epsilon, L, current_q, M, Cov_Matrix):
                                
                # U and grad_U are functions
                # current_q[0] = 9.0
                # current_q[1] = 8.0
                q = current_q
                mean = np.zeros(len(M))
                p = np.random.multivariate_normal(mean, M, 1)[0]
                current_p = p

                # Limitation on variance
                # q[2] = max(0.0001, q[2])

                # Make half step for momentum at the beginning
                p = p - epsilon * grad_U(q) / 2
                
                # Alternate full steps for position and momentum
                for i in range(L):
                    # Position Full Step for Position
                    q = q + epsilon * (Cov_Matrix @ p.reshape(-1, 1)).flatten()
                    # q = q + epsilon * p
                    # print((Cov_Matrix @ p.reshape(-1, 1)).flatten())
                    
                    # Momentum full step, except at the end of trajectory
                    if i != (L-1):
                        # Limitation on variance again
                        # q[2] = max(0.0001, q[2])

                        p = p - epsilon * grad_U(q)
                        # print(p)
                
                # Limitation on variance last time
                # q[2] = max(0.0001, q[2])
                # Make half step for momentum at the end
                p = p - epsilon * grad_U(q) / 2
                # Negate momentum at end of trajectory to make the proposal symmetric
                p = -p

                # Evaluate potential and kinetic energies at start and end of trajectory
                current_U = U(current_q)
                # print(current_U)
                current_K = sum(current_p @ Cov_Matrix @ current_p.reshape(-1, 1)) / 2 # No Covariance Matrix
                proposed_U = U(q)
                proposed_K = sum(p @ Cov_Matrix @ p.reshape(-1, 1)) / 2

                # print('Q',q)
                # print('Current U', current_U)
                # print('Proposed U', proposed_U)
                # print('proposed K', proposed_K)
                # print('Current K', current_K)
                # print('Proposed K', proposed_K)
                # Accept or reject the state at end of trajectory, returning either
                # the position at the end of the trajectory or the initial position
                # Based on total energy
                if (np.random.random() < np.exp(current_U-proposed_U+current_K-proposed_K)):
                    final = q
                    accept = True
                else:
                    final = current_q
                    accept = False
                # if proposed_U > current_U:
                #     print('Better!')
                return final, accept

            Cov_Matrix = np.eye(mmtx + 1)

            M = np.linalg.inv(Cov_Matrix)
            
            for k in range(draws):

                # Lamb_tausqd = np.diag(Lamb) + (1 / tausqd) * np.identity(mmtx + 1)
                # Lamb_tausqd_inv = np.diag(1 / np.diag(Lamb_tausqd))

                # mun = Q.dot(Lamb_tausqd_inv).dot(np.transpose(Q)).dot(Xty)
                # S = Q.dot(np.diag(np.diag(Lamb_tausqd_inv) ** (1 / 2)))

                # vec = np.random.normal(loc=0, scale=1, size=(mmtx + 1, 1))  # drawing from normal distribution
                # betas[k][:] = np.transpose(mun + sigsqd ** (1 / 2) * (S).dot(vec))

                # vecc = mun - np.reshape(betas[k][:], (len(betas[k][:]), 1))

                # Single sample of HMC

                betas[k][:], _ = HMC(neg_log_likelihood, 
                                     d_neg_log_likelihood, 
                                     epsilon=0.01, 
                                     L = 10, 
                                     current_q = betas[k-1][:], # Initial point always 0 array
                                     M = M,
                                     Cov_Matrix = Cov_Matrix)

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
                    
class GP:
    def __init__(self):
        """
        Initializes an instance of the GP (Gaussian Process) class. This constructor sets up
        initial configuration or state for the class instance.

        Attributes:
            init (str): A simple initialization attribute set to 'test' as a placeholder.

        Example:
            # Creating an instance of the GP class
            gp_instance = GP()
        """
        self.init = 'test'

    def inputs_to_phind(self, inputs, phis):
        """
        Processes and normalizes inputs to compute indices for accessing spline coefficients
        and additional transformations needed for spline evaluation.

        This method normalizes the input values twice: first, to an index corresponding to the
        spline coefficients and second, to calculate the offset within the specific segment of the spline.

        Args:
            inputs (jax.numpy.ndarray): Normalized input values (expected to be in the range [0, 1]) that represent
                                        positions within the spline domain.
            phis (jax.numpy.ndarray): Spline coefficients organized in an array where each sub-array
                                    contains coefficients for a different segment of the spline.

        Returns:
            tuple: Contains:
                - phind (jax.numpy.ndarray): Array of indices pointing to the specific segments of the spline
                                            coefficients adjusted for zero-based indexing.
                - xsm (jax.numpy.ndarray): Normalized distance of the inputs from the beginning of their respective
                                        spline segment, adjusted for further computations.

        Notes:
            - The method assumes that `inputs` are scaled between 0 and 1. It multiplies `inputs` by the length
            of the coefficients to map these inputs to an index in the coefficients array.
            - `xsm` is computed to represent the distance from the exact point within the spline segment that `inputs`
            corresponds to, facilitating precise spline evaluations.
        """

        L_phis = len(phis[0][0])  # length of coeff. in basis funtions
        phind = jnp.array(jnp.ceil(inputs * L_phis), dtype=int)  # 0-1 normalization to 0-499 normalization

        phind = jnp.expand_dims(phind, axis=-1) if phind.ndim == 1 else phind

        set = (phind == 0)  # set = 1 if phind = 0, otherwise set = 0
        phind = phind + set

        xsm = L_phis * inputs - phind
        phind = phind - 1

        return phind, xsm

    def GP_eval(self, inputs, discmtx, phis, betas):
        """
        Evaluates a Gaussian Process (GP) model using the provided inputs, interaction matrix,
        spline coefficients, and beta coefficients. The function computes the GP prediction by constructing
        a feature matrix from inputs and applying transformations based on the spline coefficients.

        Args:
            inputs (jax.numpy.ndarray): The input data matrix where each row represents a different input instance.
                                        Assumed normalized from 0-1.
            discmtx (jax.numpy.ndarray): Discriminative matrix that indicates which basis functions are active
                                        for each feature. Can be a scalar if the model has only one input.
            phis (jax.numpy.ndarray): Array containing spline coefficients. Each set of coefficients corresponds
                                    to a different basis function.
            betas (jax.numpy.ndarray): Coefficient vector for the linear combination of basis functions to form
                                    the final prediction.

        Returns:
            jax.numpy.ndarray: The predicted values computed as a linear combination of transformed basis functions,
                            shaped according to the input matrix.

        Notes:
            This function constructs a feature matrix from the `inputs` by normalizing them and mapping them
            to their corresponding spline coefficients in `phis`. It then computes a transformation for each
            input feature, accumulating contributions from each basis function specified in `discmtx`. The
            result is a linear combination of these features weighted by the `betas`.

            The function handles different dimensions of input and discriminative matrices gracefully, padding
            with zeros where necessary to align dimensions.

        Example:
            # Example usage assuming predefined matrices `inputs`, `discmtx`, `phis`, and `betas`
            gp_model = GP()
            prediction = gp_model.GP_eval(inputs, discmtx, phis, betas)
            print("GP Predictions:", prediction)
        """

        # building the matrix by calculating the corresponding basis function outputs for each set of inputs
        minp, ninp = jnp.shape(inputs)

        if jnp.shape(discmtx) == ():  # part of fix for single input model
            mmtx = 1
        else:
            mmtx, null = jnp.shape(discmtx)

        # if jnp.size(Xin) == 0:
        Xin = jnp.ones((minp, 1))
        mxin, nxin = jnp.shape(Xin)
        
        if mmtx - nxin < 0:
            X = Xin
        else:
            X = jnp.append(Xin, jnp.zeros((minp, mmtx - nxin)), axis=1)

        phind, xsm = self.inputs_to_phind(inputs=inputs, phis=phis)

        null, nxin2 = jnp.shape(X)
        additional_cols = max(0, mmtx + 1 - nxin2)
        X = jnp.concatenate((X, jnp.zeros((minp, additional_cols))), axis=1)

        def body_fun_k(k, carry):
            i, j, phi_j = carry
            num = discmtx[j - 1, k] if discmtx.ndim > 1 else discmtx
            # Define the operation to perform if num is not zero
            def true_fun(_):
                nid = num - 1
                term = (phis[nid, 0, phind[i, k]] +
                        phis[nid, 1, phind[i, k]] * xsm[i, k] +
                        phis[nid, 2, phind[i, k]] * xsm[i, k] ** 2 +
                        phis[nid, 3, phind[i, k]] * xsm[i, k] ** 3)
                return phi_j * term

            # Define the operation to perform if num is zero
            def false_fun(_):
                return phi_j

            # Using lax.cond to conditionally execute true_fun or false_fun
            phi_j = lax.cond(num != 0, true_fun, false_fun, None)
            
            return (i, j, phi_j)

        def body_fun_j(j, carry):
            i, X = carry
            phi_j_initial = 1.0
            # Carry includes `j` to be accessible inside body_fun_k
            carry_initial = (i, j, phi_j_initial)
            carry = lax.fori_loop(0, ninp, body_fun_k, carry_initial)
            _, _, phi_j = carry  # Unpack the final carry to get phi_j
            X = X.at[i, j].set(phi_j)
            return (i, X)
        
        def body_fun_i(i, X):
            carry_initial = (i, X)
            _, X = lax.fori_loop(nxin, mmtx + 1, body_fun_j, carry_initial)
        
            return X

        X = lax.fori_loop(0, minp, body_fun_i, X)
        prediction = jnp.matmul(X, betas)
        return prediction   
       
class Embedded_GP_Model:
    """
    Manages multiple Gaussian Process (GP) models, allowing for physcial models
    that involve multiple GP models simultaneously.

    Attributes:
        GP (tuple of GaussianProcess): Stores the tuple of Gaussian Processes.
        key (jax.random.PRNGKey): Pseudo-random number generator key.
        discmtx (jax.numpy.ndarray): Interaction Matrix for model proposals.
        betas (jax.numpy.ndarray): Coefficients of model terms. Single vector containing
            all beta values for all GP models that is split later on.

    Args:
        *GP (GaussianProcess): A variable number of Gaussian Process objects.
    """
    
    def __init__(self, *GP):
        """
        Initializes a Multiple_GP_Model instance with several Gaussian Process models.

        Parameters:
            *GP (tuple of GaussianProcess): A tuple of Gaussian Process objects,
            each of which should be an instance of a Gaussian Process model with 
            the custom GP class previously created.
        """
        # Define critical parameters
        self.GP = GP
        self.key = random.PRNGKey(0)
        # Define placeholder values needed for GP_Processing to run when the 
        # set_equation function is ran.
        self.discmtx = jnp.array([[1]])
        self.betas = jnp.ones(len(GP)*(len(self.discmtx)+1) + 1) # Add one for sigma sampling
        # self.sigma_alpha = 2
        # self.sigma_beta = 0.01

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

    def GP_Processing(self):
        """
        Processes multiple Gaussian Processes (GPs) using a shared beta vector that is split among the processes.
        Each GP is evaluated with a subset of the beta parameters and additional inputs specific to each GP.
        The easiest way to think of this is given a set of betas, an interaction matrix, and the inputs, it 
        computes the results of the model and stores them in a two dimensional vector that is of
        size [# of GPs, # of inputs] needed to calculate the negative log likelihood.

        This method updates the `Processed_GPs` attribute of the class with the results of the GP evaluations.

        Side Effects:
            Modifies the `Processed_GPs` attribute by storing the results of the evaluations.

        Assumptions:
            - The class must have an attribute `betas` which is a JAX numpy array of beta coefficients.
            - The class must have an attribute `GP`, a tuple of GaussianProcess instances.
            - Each GaussianProcess instance in `GP` must have a method `GP_eval` (from the custom class) which
            expects the inputs, interaction matrix (`discmtx`), phis, and relevant segment of betas.
            - `inputs` and `phis` attributes must be set in the class manually by user, used as parameters 
            in the GP evaluations.
        """
        # Separate betas by GP into matrix of size [# of GPs, # of betas]
        num_functions = len(self.GP)
        num_betas = int((len(self.betas) - 1)/num_functions)
        betas_list = self.betas[:-1].reshape(num_functions,num_betas)

        # Evaluate GPs and save results into matrix of size [# of GPs, # of inputs]
        GP_results = jnp.empty((0,len(self.inputs)))
        for idx, _ in enumerate (self.GP):
              result = self.GP[idx].GP_eval(self.inputs, self.discmtx, jnp.array(self.phis), betas_list[idx])
              GP_results = jnp.append(GP_results, result.reshape(1, -1), axis=0)

        self.Processed_GPs = GP_results
    
    def set_equation(self, equation_func):
        """
        Sets the mathematical equation that contains multiple GPs and stores it in the `equation` attribute. 
        It first processes the GPs by invoking `GP_Processing` because that is what the user needs to define
        in their model as a placeholder for the GP model (probably a better way to do that and recommended
        area for improvement).

        Args:
            equation_func (callable): A function that defines the equation to be used with the processed
                                    Gaussian Processes results. This is the proposed physical model.

        Side Effects:
            - Calls `GP_Processing` which processes all Gaussian Processes as defined in the `GP` attribute
            and updates the `Processed_GPs` attribute with the results.
            - Sets the `equation` attribute to the passed `equation_func`, which can be used later
            in conjunction with the processed GP results.

        Example:
            # Define an equation function
            def my_equation():
                # Example of CSTR Reaction Kinetics
                r_co2 = -(jnp.exp(-(multi_gp_model.Processed_GPs[0]))*C_CO2*C_Sites - jnp.exp(-(multi_gp_model.Processed_GPs[1]))*C_CO2_ADS)
                return r_co2

            # Assuming `set_equation` is a method of the `Multiple_GP_Model` class and an instance `multi_gp_model` has been created:
            multi_gp_model.set_equation(my_equation)  # This will process the GPs and set the new equation function.

        Note:
            Ensure that `equation_func` is compatible with the output format of `Processed_GPs` as generated
            by `GP_Processing` to avoid runtime errors.  This means using JAX numpy.
        """
        self.GP_Processing()
        self.equation = equation_func
    
    def neg_log_likelihood(self, betas):
        """
        Calculates the overall negative log likelihood for the model results using a single specified
        set of beta coefficients. This method updates the betas, processes the Gaussian Processes, and
        applies the set equation to calculates the negative log likelihood based on the difference
        between observed data and model results.

        Args:
            betas (jax.numpy.ndarray): An array of beta coefficients to be used in the Gaussian Processes.
                                    These coefficients are set as the new value of the `betas` attribute.

        Returns:
            float: The sum of the negative log likelihood for the model results across all data points.

        Side Effects:
            - Updates the `betas` attribute with the new beta coefficients.
            - Processes the Gaussian Processes by invoking `GP_Processing`.
            - Uses the equation set in the `equation` attribute to compute results.
            - Updates internal state based on computations.

        Note:
            - This method assumes that `self.data` and `self.sig_sqd` (variance of the data) are properly set
            within the class by the user to compute the likelihood.
        """
        # Set up of method
        self.betas = betas
        self.GP_Processing()
        results = self.equation()

        # Calculate neg log likelihood
        error = self.data - results
        ln_variance = betas[-1]
        log_pdf = 0.5 * jnp.log(2 * jnp.pi * jnp.exp(ln_variance)) + (error ** 2 / (2 * jnp.exp(ln_variance)))

        # Calculate varaince prior (Will be used in future Calculations)
        # neg_log_ln_p_variance = -jnp.log(inverse_gamma_pdf(jnp.exp(ln_variance), alpha = self.sigma_alpha, beta = self.sigma_beta))
        return jnp.sum(log_pdf)
    
    def d_neg_log_likelihood_create(self):
        """
        Creates and stores the gradient function of the negative log likelihood with respect to the
        beta coefficients in the `d_neg_log_likelihood` attribute. This method uses the JAX `grad`
        function to automatically differentiate `neg_log_likelihood`.  The reason for this is for 
        more readable code later on as well as testing flexibility, though this could be done before 
        sampling explicitly.

        Side Effects:
            - Sets the `d_neg_log_likelihood` attribute to the gradient (derivative) function of
            `neg_log_likelihood`, allowing it to be called later to compute gradient values.

        Note:
            - This method must be called before using `d_neg_log_likelihood` to compute gradients.
            - The `neg_log_likelihood` method must be correctly defined and compatible with JAX's automatic
            differentiation, which includes ensuring that all operations within `neg_log_likelihood` are
            differentiable and supported by JAX.
        """
        self.d_neg_log_likelihood = grad(self.neg_log_likelihood)
            
    def HMC(self, epsilon, L, current_q, M, Cov_Matrix, key):
        """
        Performs one iteration of the Hamiltonian Monte Carlo (HMC) algorithm to sample from
        a probability distribution proportional to the exponential of the negative log likelihood
        of the model. This method updates positions and momenta using Hamiltonian dynamics.

        Args:
            epsilon (float): Step size for the leapfrog integrator.
            L (int): Number of leapfrog steps to perform in each iteration.
            current_q (jax.numpy.ndarray): Current position (parameter vector representing betas of all GPs).
            M (jax.numpy.ndarray): Mass matrix, typically set to the identity matrix for inital sampling.
            Cov_Matrix (jax.numpy.ndarray): Covariance matrix (inverse of M) used to scale the kinetic energy.
            key (jax.random.PRNGKey): Pseudo-random number generator key.

        Returns:
            tuple: Contains the following elements:
                - new_q (jax.numpy.ndarray): The new position (parameters) after one HMC iteration. Will be current_q if not accepted.
                - accept (bool): Boolean indicating whether the new state was accepted based on the Metropolis-Hastings algorithm.
                - new_neg_log_likelihood (float): The negative log likelihood evaluated at the new position, providing a measure of the fit or suitability of the new parameters.
                - updated_key (jax.random.PRNGKey): The updated PRNG key after random operations, necessary for subsequent random operations to maintain randomness properties.


        Side Effects:
            - Updates the pseudo-random number generator key by splitting it for use in stochastic steps.

        Note:
            - The `grad_U` function refers to the gradient of the `neg_log_likelihood` method and must be
            created and stored in `d_neg_log_likelihood` before calling this method.
            - This method assumes that all necessary mathematical operations within are supported by JAX
            and that `M` and `Cov_Matrix` are appropriately defined for the problem at hand.
        """
        # Reassign for brevity in coding
        U = self.neg_log_likelihood
        grad_U = self.d_neg_log_likelihood

        # Random Momentum Sampling
        key, subkey = random.split(key)
        mean = jnp.zeros(len(M))
        p = random.multivariate_normal(subkey, mean, self.M)
        current_p = p

        ### Begin Leapfrog Integration
        # Make half step for momentum at the beginning
        p = p - epsilon * grad_U(current_q) / 2

        def loop_body(i, val):
            q, p = val
            q = q + epsilon * (Cov_Matrix @ p.reshape(-1, 1)).flatten()
            p_update = epsilon * grad_U(q)
            last_iter_factor = 1 - (i == L - 1)
            p = p - last_iter_factor * p_update
            return (q, p)

        q, p = fori_loop(0, L, loop_body, (current_q, p))

        # Make half step for momentum at the end
        p = p - epsilon * grad_U(q) / 2
        ### End Leapfrog Integration

        # Metropolis Hastings Criteria Evaluation
        # Negate momentum for detail balance
        p = -p

        current_U = U(current_q)
        current_K = sum(current_p @ Cov_Matrix @ current_p.reshape(-1, 1)) / 2
        proposed_U = U(q)
        proposed_K = sum(p @ Cov_Matrix @ p.reshape(-1, 1)) / 2

        accept_prob = jnp.exp(current_U - proposed_U + current_K - proposed_K)

        # If statement of Metropolis Hastings Criteria in JAX for optimized performance
        def true_branch(_):
            return q, True

        def false_branch(_):
            return current_q, False

        final, accept = cond(random.uniform(subkey) < accept_prob, true_branch, false_branch, None)

        return final, accept, U(final), key
    
    def create_jit_HMC(self):
        """
        Compiles the Hamiltonian Monte Carlo (HMC) method using JAX's Just-In-Time (JIT) compilation.
        This process optimizes the HMC method for faster execution by compiling it to machine code
        tailored to the specific hardware it will run on. The compiled function is stored in the
        `jit_HMC` attribute of the class.  The reason it is done in this fashion is for a) code
        brevity later on and b) the fact that JIT needs to occur post the user giving their model
        so this automates this process.

        Side Effects:
            - Sets the `jit_HMC` attribute to a JIT-compiled version of the `HMC` method. This allows
            the HMC method to execute more efficiently by reducing Python's overhead and optimizing
            execution at the hardware level.

        Usage:
            After invoking this method, `jit_HMC` can be used in place of `HMC` to perform Hamiltonian
            Monte Carlo sampling with significantly improved performance, especially beneficial
            in scenarios involving large datasets/complex models and where `HMC` is called a significant
            number of times (which is most models).

         Note:
            - The `create_jit_HMC` method should be called before using `jit_HMC` for the first time to ensure
            that the JIT compilation is completed.
        """
        self.jit_HMC = jit(self.HMC)
    
    def leapfrog(self, theta, r, grad, epsilon, f, Cov_Matrix):
        """ Perfom a leapfrog jump in the Hamiltonian space
        INPUTS
        ------
        theta: ndarray[float, ndim=1]
            initial parameter position

        r: ndarray[float, ndim=1]
            initial momentum

        grad: float
            initial gradient value

        epsilon: float
            step size

        f: callable
            it should return the log probability and gradient evaluated at theta
            logp, grad = f(theta)

        OUTPUTS
        -------
        thetaprime: ndarray[float, ndim=1]
            new parameter position
        rprime: ndarray[float, ndim=1]
            new momentum
        gradprime: float
            new gradient
        logpprime: float
            new lnp
        """
        # make half step in r
        rprime = r + 0.5 * epsilon * grad
        # make new step in theta
        # Something is wrong here I think?? Theta prime is very large. Should M_inv be just M?
        thetaprime = theta + epsilon * (Cov_Matrix @ rprime.reshape(-1, 1)).flatten()
        #compute new gradient
        # Limitation on variance
        logpprime, gradprime = f(thetaprime)
        # make half step in r again
        rprime = rprime + 0.5 * epsilon * gradprime
        return thetaprime, rprime, gradprime, logpprime

    def find_reasonable_epsilon(self, theta0, key):
        """ 
        Heuristic for choosing an initial value of epsilon.
        Algorithm 4 from original paper 
        """
        def f(theta):
            return self.neg_log_likelihood(theta)*-1, self.d_neg_log_likelihood(theta)*-1
        
        logp0, grad0 = f(theta0)
        epsilon = 1.
        # Initial Momentum
        mean = jnp.zeros(len(self.M))
        key, subkey = random.split(key)
        r0 = random.multivariate_normal(key, mean, self.M)

        # Figure out what direction we should be moving epsilon.
        _, rprime, gradprime, logpprime = self.leapfrog(theta0, r0, grad0, epsilon, f, self.Cov_Matrix)
        # brutal! This trick make sure the step is not huge leading to infinite
        # values of the likelihood. This could also help to make sure theta stays
        # within the prior domain (if any)
        def cond_fun(k):
            _, _, gradprime, logpprime = self.leapfrog(theta0, r0, grad0, epsilon * k, f, self.Cov_Matrix)
            is_inf = jnp.isinf(logpprime) | jnp.isinf(gradprime).any()
            return is_inf

        def body_fun(k):
            k *= 0.5
            _, _, gradprime, logpprime = self.leapfrog(theta0, r0, grad0, epsilon * k, f, self.Cov_Matrix)
            is_inf = jnp.isinf(logpprime) | jnp.isinf(gradprime).any()
            return k # lax.select(is_inf, k * 0.5, k) # cond(is_inf, lambda _: k * 0.5, lambda _: k, None)

        k = 1.
        k = while_loop(cond_fun, body_fun, k)

        epsilon = 0.5 * k * epsilon

        # The goal is to find the current acceptance probability and then move
        # epsilon in a direction until it crosses the 50% acceptance threshold
        # via doubling of epsilon
        logacceptprob = logpprime-logp0-0.5*((rprime @ rprime)-(r0 @ r0))
        a = lax.select(logacceptprob > jnp.log(0.5), 1., -1.)
        # Keep moving epsilon in that direction until acceptprob crosses 0.5.

        def cond_fun(carry):
            epsilon, logacceptprob = carry
            # jax.debug.print("Log Acceptance Probability: {}", logacceptprob )
            return a * logacceptprob > -a * jnp.log(2.)

        def body_fun(carry):
            epsilon, logacceptprob = carry
            epsilon = epsilon * (2. ** a)
            _, rprime, _, logpprime = self.leapfrog(theta0, r0, grad0, epsilon, f, self.Cov_Matrix)
            logacceptprob = logpprime - logp0 - 0.5 * ((rprime @ rprime) - (r0 @ r0))
            return epsilon, logacceptprob

        # epsilon = 1.
        epsilon, logacceptprob = lax.while_loop(cond_fun, body_fun, (epsilon, logacceptprob))

        return epsilon
    
    def create_jit_find_reasonable_epsilon(self):
        """
        Compiles the `find_reasonable_epsilon` function using JAX's Just-In-Time (JIT) compilation to
        enhance its performance. This method optimizes the function for faster execution by compiling it
        to machine-specific code, significantly reducing runtime. The reason it is done in this fashion 
        is for a) code brevity later on and b) the fact that JIT needs to occur post the user giving their 
        model so this automates this process.

        Side Effects:
            - Sets the `jit_find_reasonable_epsilon` attribute to a JIT-compiled version of the
            `find_reasonable_epsilon` method. This allows the method to execute more efficiently by
            reducing Python overhead and leveraging optimized low-level operations.

        Usage:
            This method should be called before any intensive sampling procedures where `find_reasonable_epsilon`
            is expected to be called, to minimize computational overhead and improve overall
            performance of the sampling process.

        Note:
            - JIT compilation happens the first time the JIT-compiled function is called, not when
            `create_jit_find_reasonable_epsilon` is executed.
        """
        self.jit_find_reasonable_epsilon = jit(self.find_reasonable_epsilon)
    
    def full_sample(self, draws):
        """
        Conducts a full HMC sampling, creating multiple draws from the posterior distribution
        of the model parameters. This function initializes and updates sampling parameters, executes
        Hamiltonian Monte Carlo using a JIT-compiled version of the sampling routine, and
        dynamically adjusts the step size based on acceptance rates.

        Args:
            draws (int): Number of samples to draw from the posterior distribution.

        Returns:
            tuple: A tuple containing:
                - samples (jax.numpy.ndarray): An array of sampled parameter vectors.
                - acceptance_array (jax.numpy.ndarray): An array indicating whether each sample was accepted.
                - neg_log_likelihood_array (jax.numpy.ndarray): An array of negative log likelihood values for each sample.

        Procedure:
            1. Initialize the covariance and mass matrices.
            2. Create a JIT-compiled Hamiltonian Monte Carlo (HMC) sampler.
            3. Iteratively sample using HMC, adjusting the leapfrog step size (`epsilon`) based on acceptance rates.
            4. Adjust the mass matrix based on warm up.

        Notes:
            - This method assumes `find_reasonable_epsilon` and `create_jit_HMC` are available to set reasonable
            values for `epsilon` and to compile the HMC sampling method, respectively.
            - The dynamic adjustment of `epsilon` aims to optimize the sampling efficiency by tuning the
            acceptance rate to a desirable range.
            - The mass matrix (`M`) and the covariance matrix (`Cov_Matrix`) are recalibrated during the sampling
            based on the properties of the collected samples to enhance sampling accuracy and efficiency.
            - The function also monitors for stagnation in parameter space and makes significant adjustments to 
            `epsilon` and recalibrates `M` and `Cov_Matrix` as needed.

        Example Usage:
            # Assuming an instance of the model `model_instance` has been created:
            samples, accepts, nlls = model_instance.full_sample(1000)
            print("Sampled Parameters:", samples)
            print("Acceptance Rates:", accepts)
            print("Negative Log Likelihoods:", nlls)
        """
        # Initialize parameters for new interaction matrix
        self.Cov_Matrix = jnp.eye(len(self.GP)*(len(self.discmtx)+1) +1)
        self.M = jnp.linalg.inv(self.Cov_Matrix)
        neg_log_likelihood_array = jnp.zeros(draws+1, dtype=float)
        acceptance_array = jnp.zeros(draws+1, dtype=bool)
        samples = jnp.ones((draws+1, len(self.GP)*(len(self.discmtx)+1)+1)) # Starting point always all betas = 1

        # Create relevant functions
        self.d_neg_log_likelihood_create()
        self.create_jit_find_reasonable_epsilon()
        self.create_jit_HMC()

        # Create Initial Epsilon Estimate
        self.epsilon = self.jit_find_reasonable_epsilon(samples[0], self.key)

        # Loop for HMC Sampling        
        for i in range(draws):
            # Print iteration in loop
            print(i)
            # Actual HMC Sampling
            sample, accept, neg_log_likelihood_sample, self.key = self.jit_HMC(epsilon = self.epsilon,
                                                                               L = 20, 
                                                                               current_q = samples[i],
                                                                               M = self.M,
                                                                               Cov_Matrix = self.Cov_Matrix, 
                                                                               key = self.key)
            
            # Save HMC sampling results
            samples = samples.at[i+1].set(sample)
            acceptance_array = acceptance_array.at[i+1].set(accept)
            neg_log_likelihood_array = neg_log_likelihood_array.at[i+1].set(neg_log_likelihood_sample)  

            # To make epsilon adaptive, modify based on acceptance rate (ideal 65% per paper)
            if (i+1) % 50 == 0:
                if sum(acceptance_array[i-50:i]) < 20:
                    self.epsilon = self.epsilon*0.5
                    print('Massive Decrease to Epsilon')
                if sum(acceptance_array[i-50:i]) < 40 and sum(acceptance_array[i-50:i]) >= 20:
                    self.epsilon = self.epsilon*0.8
                    print('Decreased Epsilon')
                if sum(acceptance_array[i-50:i]) > 40 and sum(acceptance_array[i-50:i]) <= 48:
                    self.epsilon = self.epsilon*1.2
                    print('Increased Epsilon')
                if sum(acceptance_array[i-50:i]) > 48:
                    self.epsilon = self.epsilon*1.5
                    print('Massive Increase to Epsilon')

            # Update Mass Matrix after warmup (NOTE: breaks detail balance)
            if (i+1) in [500] and len(jnp.unique(samples[i-100:i],axis=0)) >= 5:
                print('M Update')
                # Take the last 100 values of the vector and create Covariance and Mass Matrixes
                last_100_values = jnp.unique(samples[i-100:i],axis=0)
                cov_matrix = jnp.cov(last_100_values, rowvar=False)
                self.Cov_Matrix = cov_matrix.diagonal()*jnp.identity(len(cov_matrix))
                self.M = jnp.linalg.inv(cov_matrix)*jnp.identity(len(cov_matrix))
                print(self.M)

                # Update epsilon
                theta = samples[i]
                self.epsilon = self.jit_find_reasonable_epsilon(theta, self.key)
        
        return samples, acceptance_array, neg_log_likelihood_array
    
    def full_routine(self, draws, tolerance, way3 = 0):
        """
        Creates the interaction matrixes and compares the models against eachother.  Taken from methodology
        with a singular GP.
        """
        # relats_in = jnp.array([])
        
        def perms(x):
            """Python equivalent of MATLAB perms."""
            # from https://stackoverflow.com/questions/38130008/python-equivalent-for-matlabs-perms
            a = jnp.array(jnp.vstack(list(itertools.permutations(x)))[::-1])
            return a

        # 'n' is the number of datapoints whereas 'm' is the number of inputs
        n, m = jnp.shape(self.inputs)
        mrel = n
        damtx = jnp.array([])
        evs = jnp.array([])

        # Conversion of Lines 79-100 of emulator_Xin.m
        # if jnp.logical_not(all([isinstance(index, int) for index in relats_in])):  # checks if relats is an array
        #     if jnp.any(relats_in):
        #         relats = jnp.zeros((sum(jnp.logical_not(relats_in)), m))
        #         ind = 1
        #         for i in range(0, m):
        #             if jnp.logical_not(relats_in[i]):
        #                 relats[ind][i] = 1
        #                 ind = ind + 1
        #         ind_in = m + 1
        #         for i in range(0, m - 1):
        #             for j in range(i + 1, m):
        #                 if jnp.logical_not(relats_in[ind_in]):
        #                     relats[ind][i] = 1
        #                     relats[ind][j] = 1
        #                     ind = ind + 1
        #             ind_in = ind_in + 1
        #     mrel = sum(np.logical_not(relats_in)).all()
        # else:
        #     mrel = sum(np.logical_not(relats_in))
        mrel = 0
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
                vecs = jnp.unique(perms(indvec),axis=0)
                if ind > 1:
                    mvec, nvec = np.shape(vecs)
                else:
                    mvec = jnp.shape(vecs)[0]
                    nvec = 1
                killvecs = []
                if mrel != 0:
                    for j in range(1, mvec):
                        testvec = jnp.divide(vecs[j, :], vecs[j, :])
                        testvec[jitnp.isnan(testvec)] = 0
                        for k in range(1, mrel):
                            if sum(testvec == relats[k, :]) == m:
                                killvecs.append(j)
                                break
                    nuvecs = jnp.zeros(mvec - jnp.size(killvecs), m)
                    vecind = 1
                    for j in range(1, mvec):
                        if not (j == killvecs):
                            nuvecs[vecind, :] = vecs[j, :]
                            vecind = vecind + 1

                    vecs = nuvecs
                if ind > 1:
                    vm, vn = jnp.shape(vecs)
                else:
                    vm = jnp.shape(vecs)[0]
                    vn = 1
                if jnp.size(damtx) == 0:
                    damtx = vecs
                else:
                    damtx = jnp.append(damtx, vecs, axis=0)
                [dam,null] = jnp.shape(damtx)
                self.discmtx = damtx.astype(int)
                print(damtx)

                beters, null, neg_log_likelihood = self.full_sample(draws)

                ev = (2*len(self.discmtx) + 1) * jnp.log(n) - 2 * jnp.max(neg_log_likelihood*-1)

                # if aic:
                #     ev = ev + (2 - np.log(n)) * (dam + 1)

                # This is the means and bounds fo the model just sampled
                # betavs = np.abs(np.mean(beters[int(np.ceil((draws / 2)+1)):draws, (dam - vm + 1):dam+1], axis=0))
                # betavs2 = np.divide(np.std(np.array(beters[int(np.ceil(draws/2)+1):draws, dam-vm+1:dam+1]), axis=0), \
                #     np.abs(np.mean(beters[int(np.ceil(draws / 2)):draws, dam-vm+1:dam+2], axis=0)))
                #     # betavs2 error in std deviation formatting
                # betavs3 = np.array(range(dam-vm+2, dam+2))
                # betavs = np.transpose(np.array([betavs,betavs2, betavs3]))
                # if np.shape(betavs)[1] > 0:
                #     sortInds = np.argsort(betavs[:, 0])
                #     betavs = betavs[sortInds]

                # killset = []
                evmin = ev


                # This is for deletion of terms
                # for i in range(0, vm):


                #     # if betavs[i, 1] > threshstdb or betavs[i, 1] > threshstda and betavs[i, 0] < threshav * \
                #     #     np.mean(np.abs(np.mean(beters[int(np.ceil(draws/2 +1)):draws, 0]))):
                #     if betavs[i, 1] > threshstdb or betavs[i, 1] > threshstda and betavs[i, 0] < threshav * \
                #         np.mean(np.abs(np.mean(beters[int(np.ceil(draws/2)):draws, 0]))):  # index to 'beters' \
                #         # adjusted for matlab to python [JPK DEV v3.1.0 20240129]

                #         killtest = np.append(killset, (betavs[i, 2] - 1))
                #         if killtest.size > 1:
                #             killtest[::-1].sort()  # max to min so damtx_test rows get deleted in order of end to start
                #         damtx_test = damtx
                #         for k in range(0, np.size(killtest)):
                #             damtx_test = np.delete(damtx_test, int(np.array(killtest[k])-1), 0)
                #         damtest, null = np.shape(damtx_test)

                #         [betertest, null, null, null, Xtest, evtest] = hmc(inputs_np, data, phis, X, damtx_test, a, b,
                #                                                              atau, btau, draws)
                #         if aic:
                #             evtest = evtest + (2 - np.log(n))*(damtest+1)
                #         if evtest < evmin:
                #             killset = killtest
                #             evmin = evtest
                #             xers = Xtest
                #             beters = betertest
                # for k in range(0, np.size(killset)):
                #     damtx = np.delete(damtx, int(np.array(killset[k]) - 1), 0)

                ev = jnp.min(evmin)
                # X = xers

                # print(ev)
                # print(evmin)
                print([ind, ev])
                if jnp.size(evs) > 0:
                    if ev < jnp.min(evs):

                        betas = beters
                        mtx = damtx
                        greater = 1
                        evs = jnp.append(evs, ev)

                    elif greater < tolerance:
                        greater = greater + 1
                        evs = jnp.append(evs, ev)
                    else:
                        finished = 1
                        evs = jnp.append(evs, ev)

                        break
                else:
                    greater = greater + 1
                    betas = beters
                    mtx = damtx
                    evs = jnp.append(evs, ev)
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

            if ind > len(self.phis):
                break

        # # Implementation of 'gimme' feature
        # if gimmie:
        #     betas = beters
        #     mtx = damtx

        self.betas = betas
        self.mtx = mtx
        self.evs = evs

        return betas, mtx, evs
    
    # def evaluate(self):
        
    #     results = self.equation()

    #     return results
    
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
    
    def evaluate(self, inputs, GP_number, **kwargs):
        """
        Evaluate the inputs and output the predicted values of corresponding data. Optionally, calculate bounds.

        Input:
            inputs == matrix of independent (or non-linearly dependent) 'x' variables for evaluating f(x1, ..., xM)
            GP_number == the GP you would like to evaluate from the training

        Keyword Inputs:
            draws        == number of beta terms used                              == 100 (default)
            nform        == logical to automatically normalize and format 'inputs' == 1 (default)
            ReturnBounds == logical to return confidence bounds as second output   == 0 (default)
        """

        # Default keywords:
        kwargs_all = {'draws': 100, 'ReturnBounds': 0}

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
        # nform = kwargs_all.get('nform')
        ReturnBounds = kwargs_all.get('ReturnBounds')

        # # Process nform:
        # if isinstance(nform, str):
        #     if nform.lower() in ['yes','y','on','auto','default','true']:
        #         nform = 1
        #     elif nform.lower() in ['no','n','off','false']:
        #         nform = 0
        # else:
        #     if nform not in [0,1]:
        #         raise ValueError("Keyword argument 'nform' must a logical 1 (default) or 0.")

        # Automatically normalize and format inputs:
        # def auto_nform(inputs):

        #     # Convert 'inputs' to numpy if pandas:
        #     if any(isinstance(inputs, type) for type in (pd.DataFrame, pd.Series)):
        #         inputs = inputs.to_numpy()
        #         warnings.warn("'inputs' was auto-converted to numpy. Convert manually for assured accuracy.", UserWarning)

        #     # Normalize 'inputs' and convert to proper format for FoKL:
        #     inputs = np.array(inputs) # attempts to handle lists or any other format (i.e., not pandas)
        #     # . . . inputs = {ndarray: (N, M)} = {ndarray: (datapoints, input variables)} =
        #     # . . . . . . array([[x1(t1),x2(t1),...,xM(t1)],[x1(t2),x2(t2),...,xM(t2)],...,[x1(tN),x2(tN),...,xM(tN)]])
        #     inputs = np.squeeze(inputs) # removes axes with 1D for cases like (N x 1 x M) --> (N x M)
        #     if inputs.ndim == 1:  # if inputs.shape == (number,) != (number,1), then add new axis to match FoKL format
        #         inputs = inputs[:, np.newaxis]
        #     N = inputs.shape[0]
        #     M = inputs.shape[1]
        #     if M > N: # if more "input variables" than "datapoints", assume user is using transpose of proper format above
        #         inputs = inputs.transpose()
        #         warnings.warn("'inputs' was transposed. Ignore if more datapoints than input variables.", category=UserWarning)
        #         N_old = N
        #         N = M # number of datapoints (i.e., timestamps)
        #         M = N_old # number of input variables
        #     minmax = self.normalize
        #     inputs_min = np.array([minmax[ii][0] for ii in range(len(minmax))])
        #     inputs_max = np.array([minmax[ii][1] for ii in range(len(minmax))])
        #     inputs = (inputs - inputs_min) / (inputs_max - inputs_min)

        #     nformputs = inputs.tolist() # convert to list, which is proper format for FoKL, like:
        #     # . . . {list: N} = [[x1(t1),x2(t1),...,xM(t1)],[x1(t2),x2(t2),...,xM(t2)],...,[x1(tN),x2(tN),...,xM(tN)]]

        #     return nformputs

        # if nform:
        #     normputs = auto_nform(inputs)
        # else: # assume provided inputs are already normalized and formatted
        normputs = inputs

        betas = self.betas[-draws:, 0:-1] # Get all the betas but the sigmas
        num_functions = len(self.GP)
        num_betas = int(len(betas[0])/num_functions)
        betas_list = betas[:, GP_number*num_betas:(GP_number+1)*num_betas]

        betas = betas_list
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
