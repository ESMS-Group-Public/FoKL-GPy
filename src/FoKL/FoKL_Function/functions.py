import warnings
import time
import os
import pickle
import numpy as np
from ..utils import _str_to_bool, _process_kwargs

class functions():
    def __init__(self, fokl, config, dataFormat):
        self.fokl = fokl
        self.config = config
        self.dataFormat = dataFormat
    
    def evaluate_basis(self, c, x, kernel=None, d=0):
        """
        Evaluate a basis function at a single point by providing coefficients, x value(s), and (optionally) the kernel.

        Inputs:
            > c == coefficients of a single basis functions
            > x == value of independent variable at which to evaluate the basis function

        Optional Input:
            > kernel == 'Cubic Splines' or 'Bernoulli Polynomials' == self.kernel (default)
            > d      == integer representing order of derivative   == 0 (default)

        Output (in Python syntax, for d=0):
            > if kernel == 'Cubic Splines':
                > basis = c[0] + c[1]*x + c[2]*(x**2) + c[3]*(x**3)
            > if kernel == 'Bernoulli Polynomials':
                > basis = sum(c[k]*(x**k) for k in range(len(c)))
        """
        if kernel is None:
            kernel = self.config.DEFAULT['kernel']
        elif isinstance(kernel, int):
            kernel = self.config.KERNELS[kernel]

        if kernel not in self.config.KERNELS:  # check user's provided kernel is supported
            raise ValueError(f"The kernel {kernel} is not currently supported. Please select from the following: "
                             f"{self.config.KERNELS}.")

        if kernel == self.config.KERNELS[0]:  # == 'Cubic Splines':
            if d == 0:  # basis function
                basis = c[0] + c[1] * x + c[2] * (x ** 2) + c[3] * (x ** 3)
            elif d == 1:  # first derivative
                basis = c[1] + 2 * c[2] * x + 3 * c[3] * (x ** 2)
            elif d == 2:  # second derivative
                basis = 2 * c[2] + 6 * c[3] * x
        elif kernel == self.config.KERNELS[1]:  # == 'Bernoulli Polynomials':
            if d == 0:  # basis function
                basis = c[0] + sum(c[k] * (x ** k) for k in range(1, len(c)))
            elif d == 1:  # first derivative
                basis = c[1] + sum(k * c[k] * (x ** (k - 1)) for k in range(2, len(c)))
            elif d == 2:  # second derivative
                basis = sum((k - 1) * k * c[k] * (x ** (k - 2)) for k in range(2, len(c)))

        return basis
    
    def bss_derivatives(self, **kwargs):
        """
        For returning gradient of modeled function with respect to each, or specified, input variable.
        If user overrides default settings, then 1st and 2nd partial derivatives can be returned for any variables.

        Keyword Inputs:
            inputs == NxM matrix of 'x' input variables for fitting f(x1, ..., xM)    == self.inputs (default)
            kernel == function to use for differentiation (i.e., cubic or Bernoulli)  == self.kernel (default)
            d1        == index of input variable(s) to use for first partial derivative  == True (default)
            d2        == index of input variable(s) to use for second partial derivative == False (default)
            draws     == number of beta terms used                                       == self.draws (default)
            betas     == draw from the posterior distribution of coefficients            == self.betas (default)
            phis      == spline coefficients for the basis functions                     == self.phis (default)
            mtx       == basis function interaction matrix from the best model           == self.mtx (default)
            minmax    == list of [min, max]'s of input data used in the normalization    == self.minmax (default)
            IndividualDraws == boolean for returning derivative(s) at each draw       == 0 (default)
            ReturnFullArray == boolean for returning NxMx2 array instead of Nx2M      == 0 (default)

        Return Outputs:
            dy == derivative of input variables (i.e., states)

        Notes:
            - To turn off all the first-derivatives, set d1=False instead of d1=0. 'd1' and 'd2', if set to an integer,
            will return the derivative with respect to the input variable indexed by that integer using Python indexing.
            In other words, for a two-input FoKL model, setting d1=1 and d2=0 will return the first-derivative with
            respect to the second input (d1=1) and the second-derivative with respect to the first input (d2=0).
            Alternatively, d1=[False, True] and d2=[True, False] will function the same.
        """

        # Process keywords:
        default = {'inputs': None, 'kernel': self.kernel, 'd1': None, 'd2': None, 'draws': self.draws, 'betas': None,
                   'phis': None, 'mtx': self.mtx, 'minmax': self.minmax, 'IndividualDraws': False,
                   'ReturnFullArray': False, 'ReturnBasis': False}
        current = _process_kwargs(default, kwargs)
        for boolean in ['IndividualDraws', 'ReturnFullArray', 'ReturnBasis']:
            current[boolean] = _str_to_bool(current[boolean])

        # Load defaults:
        if current['inputs'] is None:
            current['inputs'] = self.inputs
        if current['betas'] is None:
            current['betas'] = self.betas
        if current['phis'] is None:
            current['phis'] = self.phis

        # Define local variables from keywords:
        inputs = current['inputs']
        kernel = current['kernel']
        d1 = current['d1']
        d2 = current['d2']
        draws = current['draws']
        betas = current['betas']
        phis = current['phis']
        mtx = current['mtx']
        span = current['minmax']

        inputs = np.array(inputs)
        if inputs.ndim == 1:
            inputs = inputs[:, np.newaxis]
        if isinstance(betas, list):  # then most likely user-input, e.g., [0,1]
            betas = np.array(betas)
            if betas.ndim == 1:
                betas = betas[:, np.newaxis]
        if isinstance(mtx, int):  # then most likely user-input, e.g., 1
            mtx = np.array(mtx)
            mtx = mtx[np.newaxis, np.newaxis]
        else:
            mtx = np.array(mtx)
            if mtx.ndim == 1:
                mtx = mtx[:, np.newaxis]
        if len(span) == 2:  # if span=[0,1], span=[[0,1],[0,1]], or span=[array([0,1]),array([0,1])]
            if not (isinstance(span[0], list) or isinstance(span[0], np.ndarray)):
                span = [span]  # make list of list to handle indexing errors for one input variable case, i.e., [0,1]

        if np.max(np.max(inputs)) > 1 or np.min(np.min(inputs)) < 0:
            warnings.warn("Input 'inputs' should be normalized (0-1). Auto-normalization is in-development.",
                          category=UserWarning)

        N = np.shape(inputs)[0]  # number of datapoints (i.e., experiments/timepoints)
        B, M = np.shape(mtx) # B == beta terms in function (not including betas0), M == number of input variables

        if B != np.shape(betas)[1]-1: # second axis is beta terms (first is draws)
            betas = np.transpose(betas)
            if B != np.shape(betas)[1]-1:
                raise ValueError(
                    "The shape of 'betas' does not align with the shape of 'mtx'. Transposing did not fix this.")

        derv = []
        i = 0
        for di in [d1, d2]:
            i = i + 1
            error_di = True
            if di is None and i == 1:
                di = np.ones(M, dtype=bool)  # default is all first derivatives (i.e., gradient)
                error_di = False
            elif di is None and i == 2:
                di = np.zeros(M, dtype=bool)  # default is no second derivatives (i.e., gradient)
                error_di = False
            elif isinstance(di, str):
                if _str_to_bool(di):
                    di = np.ones(M, dtype=bool)
                else:
                    di = np.zeros(M, dtype=bool)
                error_di = False
            elif isinstance(di, list):  # e.g., d1 = [0, 0, 1, 0] for M = 4
                if len(di) == 1:  # e.g., d1 = [3] instead of d1 = 3
                    di = di[0]  # list to integer (note, 'error_di=False' defined later)
                elif len(di) == M:
                    di = np.array(di) != 0  # assume non-zero entries are True
                    error_di = False
                else:
                    raise ValueError("Keyword input 'd1' and/or 'd2', if entered as a list, must be of equal length to "
                                     "the number of input variables.")
            if isinstance(di, bool):  # not elif because maybe list to int in above elif
                di = np.ones(M, dtype=bool) * di  # single True/False to row of True/False
                error_di = False
            elif isinstance(di, int):
                di_id = di
                di = np.zeros(M, dtype=bool)
                di[di_id] = True
                error_di = False
            if error_di:
                raise ValueError(
                    "Keyword input 'd1' and/or 'd2' is limited to an integer indexing an input variable, or to a list "
                    "of booleans corresponding to the input variables.")
            derv.append(di)  # = [d1, d2], after properly formatted

        # Determine if only one or both derivatives should be run through in for loop:
        d1_log = any(derv[0])
        d2_log = any(derv[1])
        if d1_log and d2_log:
            d1d2_log = [0, 1]  # index for both first and second derivatives
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
        if kernel == self.kernels[0]:  # == 'Cubic Splines':
            X, phind, _ = self._inputs_to_phind(inputs, phis, kernel)  # phind needed for piecewise kernel
            L_phis = len(phis[0][0])  # = 499
        elif kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
            X = inputs  # twice-normalization not required
            L_phis = 1  # because continuous
        basis_nm = np.zeros([N, M, B])  # constant term for (n,md,b) that avoids repeat calculations
        dy = np.zeros([draws, N, M, 2])
        phi = np.zeros([N, M, 2])

        # Cycle through each timepoint, input variable, and perform 'bss_derivatives' like in MATLAB:
        if current['ReturnBasis']:  # mostly for development to confirm basis function for mtx=1 and betas=[0,1]
            basis = np.zeros(N)
        for n in range(N):  # loop through experiments (i.e., each timepoint/datapoint)
            for m in range(M):  # loop through input variables (i.e., to differentiate wrt each one if requested)
                for di in d1d2_log:  # for first and/or second derivatives (depending on d1d2_log)
                    if derv[di][m]:  # if integrating, then do so once or twice (depending on di) wrt to xm ONLY
                        span_L = span_m[m] / L_phis  # used multiple times in calculations below
                        span_L = [1, span_L, span_L ** 2]  # such that span_L[derp] = span_L**derp
                        derv_nm = np.zeros(M)
                        derv_nm[m] = di + 1  # if d2 = [1, 1, 1, 1], then for m = 2, derv_nm = [0, 0, 2, 0] like MATLAB

                        # The following is like the MATLAB function, with indexing for looping through n and m:
                        for b in range(B):  # loop through betas
                            phi[n, m, di] = 1  # reset after looping through non-wrt input variables (i.e., md)
                            for md in range(M):  # for input variable PER single differentiation, m of d(xm)
                                num = int(mtx[b, md])
                                if num:  # if not 0
                                    derp = int(derv_nm[md])
                                    num = int(num - 1)  # MATLAB to Python indexing
                                    if kernel == self.kernels[0]:  # == 'Cubic Splines':
                                        phind_md = int(phind[n, md])  # make integer for indexing syntax
                                        c = list(phis[num][k][phind_md] for k in range(4))
                                    elif kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
                                        c = phis[num]
                                    if derp == 0:  # if not w.r.t. x_md
                                        if basis_nm[n, md, b] == 0:  # constant per (n,b,md)
                                            basis_nm[n, md, b] = self.evaluate_basis(c, X[n, md], kernel=kernel)
                                        phi[n, m, di] *= basis_nm[n, md, b]
                                    else:  # derp > 0
                                        phi[n, m, di] *= self.evaluate_basis(c, X[n, md], kernel=kernel, d=derp) \
                                                         / span_L[derp]
                                    if current['ReturnBasis']:  # mostly for development
                                        basis[n] = self.evaluate_basis(c, X[n, md], kernel=kernel)
                                elif derv_nm[md]:  # for terms that do not contain the variable being differentiated by
                                    phi[n, m, di] = 0
                                    break

                            dy[:,n,m,di] = dy[:,n,m,di] + betas[-draws:,b+1]*phi[n,m,di]  # update after md loop

        dy = np.transpose(dy, (1, 2, 3, 0))  # move draws dimension so dy has form (N,M,di,draws)

        if not current['IndividualDraws'] and draws > 1:  # then average draws
            dy = np.mean(dy, axis=3)  # note 3rd axis index (i.e., 4th) automatically removed (i.e., 'squeezed')
            dy = dy[:, :, :, np.newaxis]  # add new axis to avoid error in concatenate below

        if not current['ReturnFullArray']:  # then return only columns with values and stack d1 and d2 next to each other
            dy = np.concatenate([dy[:, :, 0, :], dy[:, :, 1, :]], axis=1)  # (N,M,2,draws) to (N,2M,draws)
            dy = dy[:, ~np.all(dy == 0, axis=0)]  # remove columns ('N') with all zeros
        dy = np.squeeze(dy)  # remove unnecessary axes

        if current['ReturnBasis']:  # mostly for development
            return dy, basis
        else:  # standard
            return dy
        
    def save(self, filename=None, directory=None):
        """
        Save a FoKL class as a file. By default, the 'filename' is 'model_yyyymmddhhmmss.fokl' and is saved to the
        directory of the Python script calling this method. Use 'directory' to change the directory saved to, or simply
        embed the directory manually within 'filename'.

        Returned is the 'filepath'. Enter this as the argument of 'load' to later reload the model. Explicitly, that is
        'FoKLRoutines.load(filepath)' or 'FoKLRoutines.load(filename, directory)'.

        Note the directory must exist prior to calling this method.
        """
        if filename is None:
            t = time.gmtime()

            def two_digits(a):
                if a < 10:
                    a = "0" + str(a)
                else:
                    a = str(a)
                return a
            ymd = [str(t[0])]
            for i in range(1, 6):
                ymd.append(two_digits(t[i]))
            t_str = ymd[0] + ymd[1] + ymd[2] + ymd[3] + ymd[4] + ymd[5]
            filename = "model_" + t_str + ".fokl"
        elif filename[-5::] != ".fokl":
            filename = filename + ".fokl"

        if directory is not None:
            filepath = os.path.join(directory, filename)
        else:
            filepath = filename

        file = open(filepath, "wb")
        pickle.dump(self, file)
        file.close()

        time.sleep(1)  # so that next saved model is guaranteed a different filename

        return filepath