from FoKL import getKernels
from FoKL.fokl_to_pyomo import fokl_to_pyomo
import os
import sys
# # -----------------------------------------------------------------------
# # # UNCOMMENT IF USING LOCAL FOKL PACKAGE:
# dir = os.path.abspath(os.path.dirname(__file__))  # directory of script
# sys.path.append(os.path.join(dir, '..', '..'))  # package directory
# from src.FoKL import getKernels
# from src.FoKL.fokl_to_pyomo import fokl_to_pyomo
# # -----------------------------------------------------------------------
import pandas as pd
import warnings
import itertools
import math
import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import time
import pickle
import copy


def load(filename, directory=None):
    """
    Load a FoKL class from a file.

    By default, 'directory' is the current working directory that contains the script calling this method. An absolute
    or relative directory may be defined if the model to load is located elsewhere.

    For simplicity, enter the returned output from 'self.save()' as the argument here, i.e., for 'filename'. Do this
    while leaving 'directory' blank since 'filename' can simply include the directory itself.
    """
    if filename[-5::] != ".fokl":
        filename = filename + ".fokl"

    if directory is not None:
        filepath = os.path.join(directory, filename)
    else:
        filepath = filename

    file = open(filepath, "rb")
    model = pickle.load(file)
    file.close()

    return model


def _str_to_bool(s):
    """Convert potential string (e.g., 'on'/'off') to boolean True/False. Intended to handle exceptions for keywords."""
    if isinstance(s, str):
        if s in ['yes', 'y', 'on', 'all', 'true', 'both']:
            s = True
        elif s in ['no', 'n', 'off', 'none', 'n/a', 'false']:
            s = False
        else:
            warnings.warn(f"Could not understand string '{s}' as a boolean.", category=UserWarning)
    elif s is None or not s:  # 'not s' for s == []
        s = False
    else:
        try:
            if s != 0:
                s = True
            else:
                s = False
        except:
            warnings.warn(f"Could not convert non-string to a boolean.", category=UserWarning)
    return s


def _process_kwargs(default, user):
    """Update default values with user-defined keyword arguments (kwargs), or simply check all kwargs are expected."""
    if isinstance(default, dict):
        expected = default.keys()
        if isinstance(user, dict):
            for kw in user.keys():
                if kw not in expected:
                    raise ValueError(f"Unexpected keyword argument: '{kw}'")
                else:
                    default[kw] = user[kw]
        else:
            raise ValueError("Input 'user' must be a dictionary formed by kwargs.")
        return default
    elif isinstance(default, list):  # then simply check for unexpected kwargs
        for kw in user.keys():
            if kw not in default:
                raise ValueError(f"Unexpected keyword argument: '{kw}'")
        return user
    else:
        raise ValueError("Input 'default' must be a dictionary or list.")


def _set_attributes(self, attrs):
    """Set items stored in Python dictionary 'attrs' as attributes of class."""
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            setattr(self, key, value)
    else:
        warnings.warn("Input must be a Python dictionary.")
    return


def _merge_dicts(d1, d2):
    """Merge two dictionaries into single dictionary in a backward-compatible way. Values of d2 replace any shared variables in d1."""
    d = d1.copy()
    d.update(d2)
    return d
    

class FoKL:
    def __init__(self, **kwargs):
        """
        Initialization Inputs (i.e., hyperparameters and their descriptions):

            - 'kernel' is a string defining the kernel to use for building the model, which defines 'phis', a data
            structure with coefficients for the basis functions.
                - If set to 'Cubic Splines', then 'phis' defines 500 splines (i.e., basis functions) of 499 piecewise
                cubic polynomials each. (from 'splineCoefficient500_highPrecision_smoothed.txt').
                    - y = sum(phis[spline_index][k][piecewise_index] * (x ** k) for k in range(4))
                - If set to 'Bernoulli Polynomials', then 'phis' defines the first 258 non-zero Bernoulli polynomials 
                (i.e., basis functions). (from 'bernoulliNumbers258.txt').
                    - y = sum(phis[polynomial_index][k] * (x ** k) for k in range(len(phis[polynomial_index])))

            - 'phis' gets defined automatically by 'kernel', but if testing other coefficients with the same format
            implied by 'kernel' then 'phis' may be user-defined.

            - 'relats_in' is a boolean matrix indicating which terms should be excluded from the model building. For
            instance, if a certain main effect should be excluded 'relats_in' will include a row with a 1 in the column
            for that input and zeros elsewhere. If a certain two-way interaction should be excluded there should be a
            row with ones in those columns and zeros elsewhere. To exclude no terms, leave blank. For an example of
            excluding the first input main effect and its interaction with the third input for a case with three total
            inputs, 'relats_in = [[1, 0, 0], [1, 0, 1]]'.

            - 'a' and 'b' are the shape and scale parameters of the ig distribution for the observation error variance
            of the data. The observation error model is white noise. Choose the mode of the ig distribution to match the
            noise in the output dataset and the mean to broaden it some.

            - 'atau' and 'btau' are the parameters of the ig distribution for the 'tau squared' parameter: the variance
            of the beta priors is iid normal mean zero with variance equal to sigma squared times tau squared. Tau
            squared must be scaled in the prior such that the product of tau squared and sigma squared scales with the
            output dataset.

            - 'tolerance' controls how hard the function builder tries to find a better model once adding terms starts
            to show diminishing returns. A good default is 3, but large datasets could benefit from higher values.

            - 'burnin' is the total number of draws from the posterior for each tested model before the 'draws' draws.

            - 'draws' is the total number of draws from the posterior for each tested model after the 'burnin' draws.
            There draws are what appear in 'betas' after calling 'fit', and the 'burnin' draws are discarded.

            - 'gimmie' is a boolean causing the routine to return the most complex model tried instead of the model with
            the optimum bic.

            - 'way3' is a boolean specifying the calculation of three-way interactions.

            - 'threshav' and 'threshstda' form a threshold for the elimination of terms.
                - 'threshav' is a threshold for proposing terms for elimination based on their mean values, where larger
                thresholds lead to more elimination.
                - 'threshstda' is a threshold standard deviation expressed as a fraction relative to the mean.
                - terms with coefficients that are lower than 'threshav' and higher than 'threshstda' will be proposed
                for elimination but only executed based on relative BIC values.

            - 'threshstdb' is a threshold standard deviation that is independent of the mean value of the coefficient.
            All terms with a standard deviation (relative to mean) exceeding this will be proposed for elimination.

            - 'aic' is a boolean specifying the use of the aikaike information criterion.

        Default Values for Hyperparameters:
            - kernel     = 'Cubic Splines'
            - phis       = f(kernel)
            - relats_in  = []
            - a          = 4
            - b          = f(a, data)
            - atau       = 4
            - btau       = f(atau, data)
            - tolerance  = 3
            - burnin     = 1000
            - draws      = 1000
            - gimmie     = False
            - way3       = False
            - threshav   = 0.05
            - threshstda = 0.5
            - threshstdb = 2
            - aic        = False

        Other Optional Inputs:
            - UserWarnings  == boolean to print user-warnings to the command terminal          == True (default)
            - ConsoleOutput == boolean to print [ind, ev] during 'fit' to the command terminal == True (default)
        """

        # Store list of hyperparameters for easy reference later, if sweeping through values in functions such as fit:
        self.hypers = ['kernel', 'phis', 'relats_in', 'a', 'b', 'atau', 'btau', 'tolerance', 'burnin', 'draws',
                       'gimmie', 'way3', 'threshav', 'threshstda', 'threshstdb', 'aic', 'update', 'built']

        # Store list of settings for easy reference later (namely, in 'clear'):
        self.settings = ['UserWarnings', 'ConsoleOutput']

        # Store supported kernels for later logical checks against 'kernel':
        self.kernels = ['Cubic Splines', 'Bernoulli Polynomials']

        # List of attributes to keep in event of clearing model (i.e., 'self.clear'):
        self.keep = ['keep', 'hypers', 'settings', 'kernels'] + self.hypers + self.settings + self.kernels

        # Process user's keyword arguments:
        default = {
                   # Hyperparameters:
                   'kernel': 'Cubic Splines', 'phis': None, 'relats_in': [], 'a': 4, 'b': None, 'atau': 4,
                   'btau': None, 'tolerance': 3, 'burnin': 1000, 'draws': 1000, 'gimmie': False, 'way3': False,
                   'threshav': 0.05, 'threshstda': 0.5, 'threshstdb': 2, 'aic': False,

                    # Hyperparameters with Update:
                    'sigsqd0': 0.5, 'burn': 500, 'update': False, 'built' : False,

                   # Other:
                   'UserWarnings': True, 'ConsoleOutput': True
                   }
        current = _process_kwargs(default, kwargs)  # = default, but updated by any user kwargs
        for boolean in ['gimmie', 'way3', 'aic', 'UserWarnings', 'ConsoleOutput']:
            if not (current[boolean] is False or current[boolean] is True):
                current[boolean] = _str_to_bool(current[boolean])

        # Load spline coefficients:
        phis = current['phis']  # in case advanced user is testing other splines
        if isinstance(current['kernel'], int):  # then assume integer indexing 'self.kernels'
            current['kernel'] = self.kernels[current['kernel']]  # update integer to string
        if current['phis'] is None:  # if default
            if current['kernel'] == self.kernels[0]:  # == 'Cubic Splines':
                current['phis'] = getKernels.sp500()
            elif current['kernel'] == self.kernels[1]:  # == 'Bernoulli Polynomials':
                current['phis'] = getKernels.bernoulli()
            elif isinstance(current['kernel'], str):  # confirm string before printing to console
                raise ValueError(f"The user-provided kernel '{current['phis']}' is not supported.")
            else:
                raise ValueError(f"The user-provided kernel is not supported.")

        # Turn on/off FoKL warnings:
        if current['UserWarnings']:
            warnings.filterwarnings("default", category=UserWarning)
        else:
            warnings.filterwarnings("ignore", category=UserWarning)

        # Store values as class attributes:
        for key, value in current.items():
            setattr(self, key, value)

    def _format(self, inputs, data=None, AutoTranspose=True, SingleInstance=False, bit=64):
        """
        Called by 'clean' to format dataset.
            - formats inputs as 2D ndarray, where columns are input variables; n_rows > n_cols if AutoTranspose=True
            - formats data as 2D ndarray, with single column

        Note SingleInstance has priority over AutoTranspose. If SingleInstance=True, then AutoTranspose=False.
        """
        # Format and check inputs:
        AutoTranspose = _str_to_bool(AutoTranspose)
        SingleInstance = _str_to_bool(SingleInstance)
        bits = {16: np.float16, 32: np.float32, 64: np.float64}  # allowable datatypes: https://numpy.org/doc/stable/reference/arrays.scalars.html#arrays-scalars-built-in
        if SingleInstance is True:
            AutoTranspose = False
        if bit not in bits.keys():
            warnings.warn(f"Keyword 'bit={bit}' limited to values of 16, 32, or 64. Assuming default value of 64.", category=UserWarning)
            bit = 64
        datatype = bits[bit]

        # Convert 'inputs' and 'data' to numpy if pandas:
        if any(isinstance(inputs, type) for type in (pd.DataFrame, pd.Series)):
            inputs = inputs.to_numpy()
            warnings.warn("'inputs' was auto-converted to numpy. Convert manually for assured accuracy.",
                          category=UserWarning)
        if data is not None:
            if any(isinstance(data, type) for type in (pd.DataFrame, pd.Series)):
                data = data.to_numpy()
                warnings.warn("'data' was auto-converted to numpy. Convert manually for assured accuracy.",
                              category=UserWarning)

        # Format 'inputs' as [n x m] numpy array:
        inputs = np.array(inputs)  # attempts to handle lists or any other format (i.e., not pandas)
        if inputs.ndim > 2:  # remove axes with 1D for cases like (N x 1 x M) --> (N x M)
            inputs = np.squeeze(inputs)
        if inputs.dtype != datatype:
            inputs = np.array(inputs, dtype=datatype)
            warnings.warn(f"'inputs' was converted to float{bit}. May require user-confirmation that "
                          f"values did not get corrupted.", category=UserWarning)
        if inputs.ndim == 1:  # if inputs.shape == (number,) != (number,1), then add new axis to match FoKL format
            if SingleInstance is True:
                inputs = inputs[np.newaxis, :]  # make 1D into (1, M)
            else:
                inputs = inputs[:, np.newaxis]  # make 1D into (N, 1)
        if AutoTranspose is True and SingleInstance is False:
            if inputs.shape[1] > inputs.shape[0]:  # assume user is using transpose of proper format
                inputs = inputs.transpose()
                warnings.warn("'inputs' was transposed. Ignore if more datapoints than input variables, else set "
                              "'AutoTranspose=False' to disable.", category=UserWarning)

        # Format 'data' as [n x 1] numpy array:
        if data is not None:
            data = np.array(data)  # attempts to handle lists or any other format (i.e., not pandas)
            data = np.squeeze(data)
            if data.dtype != datatype:
                data = np.array(data, dtype=datatype)
                warnings.warn(f"'data' was converted to float{bit}. May require user-confirmation that "
                              f"values did not get corrupted.", category=UserWarning)
            if data.ndim == 1:  # if data.shape == (number,) != (number,1), then add new axis to match FoKL format
                data = data[:, np.newaxis]
            else:  # check user provided only one output column/row, then transpose if needed
                n = data.shape[0]
                m = data.shape[1]
                if (m != 1 and n != 1) or (m == 1 and n == 1):
                    raise ValueError("Error: 'data' must be a vector.")
                elif m != 1 and n == 1:
                    data = data.transpose()
                    warnings.warn("'data' was transposed to match FoKL formatting.", category=UserWarning)
                
        return inputs, data
    
    def _normalize(self, inputs, minmax=None, pillow=None, pillow_type='percent'):
        """
        Called by 'clean' to normalize dataset inputs.

        Inputs:
            inputs      == [n x m] ndarray where columns are input variables
            minmax      == list of [min, max] lists; upper/lower bounds of each input variable                      == self.minmax (default)
            pillow      == list of [lower buffer, upper buffer] lists; fraction of span by which to expand 'minmax' == 0 (default)
            pillow_type == string, 'percent' (i.e., fraction of span to buffer truescale) or 'absolute' (i.e., [min, max] on 0-1 scale), defining units of 'pillow' == 'percent' (default)
            
        Note 'pillow' is ignored if reading 'minmax' from previously defined 'self.minmax'; a warning is thrown if 'pillow' is defined in this case.
        
        Updates 'self.minmax'.
        """
        mm = inputs.shape[1]  # number of input variables
        
        # Process 'pillow_type':
        pillow_types = ['percent', 'absolute']
        if isinstance(pillow_type, str):
            pillow_type = [pillow_type] * mm
        elif isinstance(pillow_type, list):
            if len(pillow_type) != mm:
                raise ValueError("Input 'pillow_type' must be string or correspond to input variables (i.e., columns of 'inputs').")
        for pt in range(len(pillow_type)):
            if pillow_type[pt] not in pillow_types:
                raise ValueError(f"'pillow_type' is limited to {pillow_types}.")

        # Process 'pillow':

        _skip_pillow = False  # to skip pillow's adjustment of minmax, if pillow is default
        if pillow is None:  # default
            _skip_pillow = True
            pillow = 0.0
        if isinstance(pillow, int):  # scalar was provided
            pillow = float(pillow)
        if isinstance(pillow, float):
            pillow = [[pillow, pillow]] * mm
        elif isinstance(pillow[0], int) or isinstance(pillow[0], float):  # list was provided
            lp = len(pillow)
            if lp == 2:  # assume [lb, ub] was provided
                pillow = [[float(pillow[0]), float(pillow[1])]]  # add outer list, and ensure float
                lp = 1  # = len(pillow)
            if lp != int(mm * 2):
                raise ValueError("Input 'pillow' must correspond to input variables (i.e., columns of 'inputs').")
            else:  # assume [lb1, ub1, ..., lbm, upm] needs to be formatted to [[lb1, ub1], ..., [lbm, ubm]]
                pillow_vals = copy.deepcopy(pillow)
                pillow = []
                for i in range(0, lp, 2):
                    pillow.append([float(pillow_vals[i]), float(pillow_vals[i + 1])])  # list of [lb, ub] lists
        
        # Process 'minmax':
        
        def _minmax_error():
            raise ValueError("Input 'minmax' must correspond to input variables (i.e., columns of 'inputs').")

        if minmax is None:  # default, read 'model.normalize' or define if does not exist
            if hasattr(self, 'minmax'):
                minmax = self.minmax
            else:
                minmax = list([np.min(inputs[:, m]), np.max(inputs[:, m])] for m in range(mm))
        else:  # process 'minmax'
            if isinstance(minmax[0], int) or isinstance(minmax[0], float):  # list was provided
                lm = len(minmax)
                if lm == 2:  # assume [min, max] was provided
                    minmax = [minmax]  # add outer list
                    lm = 1  # = len(minmax)
                if lm != int(mm * 2):
                    _minmax_error()
                else:  # assume [min1, max1, ..., minm, maxm] needs to be formatted to [[min1, max1], ..., [minm, maxm]]
                    minmax_vals = copy.deepcopy(minmax)
                    minmax = []
                    for i in range(0, lm, 2):
                        minmax.append([minmax_vals[i], minmax_vals[i + 1]])  # list of [min, max] lists
            elif len(minmax) != mm:
                _minmax_error()

        if pillow is not None and _skip_pillow is False:
            minmax_vals = copy.deepcopy(minmax)
            minmax = []
            for m in range(mm):  # for input var in input vars
                x_min = minmax_vals[m][0]
                x_max = minmax_vals[m][1]
                span = x_max - x_min  # span of minmax
                if pillow_type[m] == 'percent':
                    minmax.append([x_min - span * pillow[m][0], x_max + span * pillow[m][1]])  # [min, max] with pillow buffers
                elif pillow_type[m] == 'absolute':
                    # Derivation:
                    #   Nomenclature: pillow[m] == [q, 1 - p], minmax_vals[m] == [n, m]
                    #   For [q, 1 - p] to align to 0-1 scale after normalization,
                    #       (n - min) / (max - min) = q
                    #       (m - min) / (max - min) = p
                    #   Then,
                    #       (n - min) / q = (m - min) / p
                    #       n / q - m / p = min * (1 / q - 1 / p)
                    #       min = (n / q - m / p) / (1 / q - 1 / p) = (n * p - m * q) / (p - q)
                    #   And,
                    #       max = (n - min) / q + min
                    
                    if pillow[m][0] == 0:  # then n = min
                        minmax_min = x_min
                    else:  # see above equation
                        minmax_min = (x_min * (1 - pillow[m][1]) - x_max * pillow[m][0]) / (1 - pillow[m][1] - pillow[m][0])
                    
                    if pillow[m][1] == 0:  # then m = max
                        minmax_max = x_max
                    elif pillow[m][0] == 0:  # empirically need equation rearranged in this case to avoid nan
                        minmax_max = (x_max - pillow[m][1] * minmax_min) / (1 - pillow[m][1])
                    else:  # see above equation
                        minmax_max = (x_min - minmax_min) / pillow[m][0] + minmax_min
                    
                    minmax.append([minmax_min, minmax_max])  # [min, max] such that 'pillow' values map to 0-1 scale
            
        if hasattr(self, 'minmax'):  # check if 'self.minmax' is defined, in which case give warning to re-train model
            if any(minmax[m] == self.minmax[m] for m in range(mm)) is False:
                warnings.warn("The model already contains normalization [min, max] bounds, so the currently trained model will not be valid for the new bounds requested. Train a new model with these new bounds.", category=UserWarning)
        self.minmax = minmax  # always update

        # Normalize 'inputs' to 0-1 scale according to 'minmax':
        for m in range(mm):  # for input var in input vars
            inputs[:, m] = (inputs[:, m] - minmax[m][0]) / (minmax[m][1] - minmax[m][0])
        
        return inputs
    
    def clean(self, inputs, data=None, kwargs_from_other=None, _setattr=False, **kwargs):
        """
        For cleaning and formatting inputs prior to training a FoKL model. Note that data is not required but should be
        entered if available; otherwise, leave blank.

        Inputs:
            inputs == [n x m] input matrix of n observations by m features (i.e., 'x' variables in model)
            data   == [n x 1] output vector of n observations (i.e., 'y' variable in model)

        Keyword Inputs:
            _setattr          == [NOT FOR USER] defines 'self.inputs' and 'self.data' if True == False (default)
            train             == percentage (0-1) of n datapoints to use for training      == 1 (default)
            AutoTranspose     == boolean to transpose dataset so that instances > features == True (default)
            SingleInstance    == boolean to make 1D vector (e.g., list) into (1,m) ndarray == False (default)
            bit               == floating point bits to represent dataset as               == 64 (default)
            normalize         == boolean to pass formatted dataset to '_normalize'         == True (default)
            minmax            == list of [min, max] lists; upper/lower bounds of each input variable == self.minmax (default)
            pillow            == list of [lower buffer, upper buffer] lists; fraction of span by which to expand 'minmax' == 0 (default)
            kwargs_from_other == [NOT FOR USER] used internally by fit or evaluate function

        Added Attributes:
            - self.inputs    == 'inputs' as [n x m] numpy array where each column is normalized on [0, 1] scale
            - self.data      == 'data' as [n x 1] numpy array
            - self.minmax    == [[min, max], ... [min, max]] factors used to normalize 'inputs' to 'self.inputs'
            - self.trainlog  == indices of 'self.inputs' to use as training set
        """
        # Process keywords:
        default = {'train': 1, 
                   # For '_format':
                   'AutoTranspose': True, 'SingleInstance': False, 'bit': 64,
                   # For '_normalize':
                   'normalize': True, 'minmax': None, 'pillow': None, 'pillow_type': 'percent'}
        if kwargs_from_other is not None:  # then clean is being called from fit or evaluate function
            kwargs = _merge_dicts(kwargs, kwargs_from_other)  # merge dictionaries (kwargs={} is expected but just in case)
        current = _process_kwargs(default, kwargs)
        current['normalize'] = _str_to_bool(current['normalize'])



        # Format and normalize:
        inputs, data = self._format(inputs, data, current['AutoTranspose'], current['SingleInstance'], current['bit'])
        if current['normalize'] is True:
            inputs = self._normalize(inputs, current['minmax'], current['pillow'], current['pillow_type'])
        
            # Check if any 'inputs' exceeds [0, 1], since 'normalize=True' implies this is desired:
            inputs_cap0 = inputs < 0
            inputs_cap1 = inputs > 1
            if np.max(inputs_cap0) is True or np.max(inputs_cap1) is True:
                warnings.warn("'inputs' exceeds [0, 1] normalization bounds. Capping values at 0 and 1.")
                inputs[inputs_cap0] = 0.0  # cap at 0
                inputs[inputs_cap1] = 1.0  # cap at 1

        # Define full dataset with training log as attributes when clean called from fit, or when clean called for first time:
        if hasattr(self, 'inputs') is False or _setattr is True:

            # Index percentage of dataset as training set:
            trainlog = self.generate_trainlog(current['train'], inputs.shape[0])

            # Define/update attributes with cleaned data and other relevant variables:
            attrs = {'inputs': inputs, 'data': data, 'trainlog': trainlog}
            _set_attributes(self, attrs)

        # Return formatted and possibly normalized dataset, depending on if user passed 'inputs' only or 'inputs' and 'data':
        if data is None:  # assume user only wants 'inputs' returned, e.g., 'clean_dataset = model.clean(dataset)'
            return inputs
        else:  # e.g., 'clean_inputs, clean_data = model.clean(inputs, data)'
            return inputs, data

    def generate_trainlog(self, train, n=None):
        """Generate random logical vector of length 'n' with 'train' percent as True."""
        if train < 1:
            if n is None:
                n = self.inputs.shape[0]  # number of observations
            l_log = int(n * train)  # required length of indices for training
            if l_log < 2:
                l_log = int(2)  # minimum required for training set
            trainlog_i = np.array([], dtype=int)  # indices of training data (as an index)
            while len(trainlog_i) < l_log:
                trainlog_i = np.append(trainlog_i, np.random.random_integers(n, size=l_log) - 1)
                trainlog_i = np.unique(trainlog_i)  # also sorts low to high
                np.random.shuffle(trainlog_i)  # randomize sort
            if len(trainlog_i) > l_log:
                trainlog_i = trainlog_i[0:l_log]  # cut-off extra indices (beyond 'percent')
            trainlog = np.zeros(n, dtype=bool)  # indices of training data (as a logical)
            for i in trainlog_i:
                trainlog[i] = True
        else:
            # trainlog = np.ones(n, dtype=bool)  # wastes memory, so use following method coupled with 'trainset':
            trainlog = None  # means use all observations
        return trainlog

    def trainset(self):
        """
        After running 'clean', call 'trainset' to get train inputs and train data. The purpose of this method is to
        simplify syntax, such that the code here does not need to be re-written each time the train set is defined.

        traininputs, traindata = self.trainset()
        """
        if self.trainlog is None:  # then use all observations for training
            return self.inputs, self.data
        else:  # self.trainlog is vector indexing observations
            return self.inputs[self.trainlog, :], self.data[self.trainlog]

    def _inputs_to_phind(self, inputs, phis=None, kernel=None):
        """
        Twice normalize the inputs to index the spline coefficients.

        Inputs:
            - inputs == normalized inputs as numpy array (i.e., self.inputs)
            - phis   == spline coefficients
            - kernel == form of basis functions

        Outputs:
            - X     == twice normalized inputs, used in bss_derivatives
            - phind == index of coefficients for 'Cubic Splines' kernel for 'inputs' (i.e., piecewise cubic function)
            - xsm   == unsure of description, but used in fit/gibbs (see MATLAB) as if is twice normalized
        """
        if kernel is None:
            kernel = self.kernel
        if phis is None:
            phis = self.phis

        if kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
            warnings.warn("Twice normalization of inputs is not required for the 'Bernoulli Polynomials' kernel",
                          category=UserWarning)
            return inputs, [], []

        # elif kernel == self.kernels[0]:  # == 'Cubic Splines':

        l_phis = len(phis[0][0])  # = 499, length of cubic splines in basis functions
        phind = np.array(np.ceil(inputs * l_phis), dtype=np.uint16)  # 0-1 normalization to 0-499 normalization

        if phind.ndim == 1:  # if phind.shape == (number,) != (number,1), then add new axis to match indexing format
            phind = phind[:, np.newaxis]

        set = (phind == 0)  # set = 1 if phind = 0, otherwise set = 0
        phind = phind + set  # makes sense assuming L_phis > M

        r = 1 / l_phis  # interval of when basis function changes (i.e., when next cubic function defines spline)
        xmin = np.array((phind - 1) * r, dtype=inputs.dtype)
        X = (inputs - xmin) / r  # twice normalized inputs (0-1 first then to size of phis second)

        phind = phind - 1
        xsm = np.array(l_phis * inputs - phind, dtype=inputs.dtype)

        return X, phind, xsm

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
            kernel = self.kernel
        elif isinstance(kernel, int):
            kernel = self.kernels[kernel]

        if kernel not in self.kernels:  # check user's provided kernel is supported
            raise ValueError(f"The kernel {kernel} is not currently supported. Please select from the following: "
                             f"{self.kernels}.")

        if kernel == self.kernels[0]:  # == 'Cubic Splines':
            if d == 0:  # basis function
                basis = c[0] + c[1] * x + c[2] * (x ** 2) + c[3] * (x ** 3)
            elif d == 1:  # first derivative
                basis = c[1] + 2 * c[2] * x + 3 * c[3] * (x ** 2)
            elif d == 2:  # second derivative
                basis = 2 * c[2] + 6 * c[3] * x
        elif kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
            if d == 0:  # basis function
                basis = c[0] + sum(c[k] * (x ** k) for k in range(1, len(c)))
            elif d == 1:  # first derivative
                basis = c[1] + sum(k * c[k] * (x ** (k - 1)) for k in range(2, len(c)))
            elif d == 2:  # second derivative
                basis = sum((k - 1) * k * c[k] * (x ** (k - 2)) for k in range(2, len(c)))

        return basis

    def evaluate(self, inputs=None, betas=None, mtx=None, **kwargs):
        """
        Evaluate the FoKL model for provided inputs and (optionally) calculate bounds. Note 'evaluate_fokl' may be a
        more accurate name so as not to confuse this function with 'evaluate_basis', but it is unexpected for a user to
        call 'evaluate_basis' so this function is simply named 'evaluate'.

        Input:
            inputs == input variable(s) at which to evaluate the FoKL model == self.inputs (default)

        Optional Inputs:
            betas        == coefficients defining FoKL model                       == self.betas (default)
            mtx          == interaction matrix defining FoKL model                 == self.mtx (default)
            minmax       == [min, max] of inputs used for normalization            == None (default)
            draws        == number of beta terms used                              == self.draws (default)
            clean        == boolean to automatically normalize and format 'inputs' == False (default)
            ReturnBounds == boolean to return confidence bounds as second output   == False (default)
        """

        # Process keywords:
        default = {'minmax': None, 'draws': self.draws, 'clean': False, 'ReturnBounds': False,  # for evaluate
                   '_suppress_normalization_warning': False}                                    # if called from coverage3
        default_for_clean = {'train': 1, 
                             # For '_format':
                             'AutoTranspose': True, 'SingleInstance': False, 'bit': 64,
                             # For '_normalize':
                             'normalize': True, 'minmax': None, 'pillow': None, 'pillow_type': 'percent'}
        current = _process_kwargs(_merge_dicts(default, default_for_clean), kwargs)
        for boolean in ['clean', 'ReturnBounds']:
            current[boolean] = _str_to_bool(current[boolean])
        kwargs_to_clean = {}
        for kwarg in default_for_clean.keys():
            kwargs_to_clean.update({kwarg: current[kwarg]})  # store kwarg for clean here
            del current[kwarg]  # delete kwarg for clean from current
        if current['draws'] < 40 and current['ReturnBounds']:
            current['draws'] = 40
            warnings.warn("'draws' must be greater than or equal to 40 if calculating bounds. Setting 'draws=40'.")
        draws = current['draws']  # define local variable
        if betas is None:  # default
            if draws > self.betas.shape[0]:
                draws = self.betas.shape[0]  # more draws than models results in inf time, so threshold
                self.draws = draws
                warnings.warn("Updated attribute 'self.draws' to equal number of draws in 'self.betas'.",
                              category=UserWarning)
            betas = self.betas[-draws::, :]  # use samples from last models
        else:  # user-defined betas may need to be formatted
            betas = np.array(betas)
            if betas.ndim == 1:
                betas = betas[np.newaxis, :]  # note transpose would be column of beta0 terms, so not expected
            if draws > betas.shape[0]:
                draws = betas.shape[0]  # more draws than models results in inf time, so threshold
            betas = betas[-draws::, :]  # use samples from last models
        if mtx is None:  # default
            mtx = self.mtx
        else:  # user-defined mtx may need to be formatted
            if isinstance(mtx, int):
                mtx = [mtx]
            mtx = np.array(mtx)
            if mtx.ndim == 1:
                mtx = mtx[np.newaxis, :]
                warnings.warn("Assuming 'mtx' represents a single model. If meant to represent several models, then "
                              "explicitly enter a 2D numpy array where rows correspond to models.")

        phis = self.phis

        # Automatically normalize and format inputs:
        if inputs is None:  # default
            inputs = self.inputs
            if current['clean']:
                warnings.warn("Cleaning was already performed on default 'inputs', so overriding 'clean' to False.",
                              category=UserWarning)
                current['clean'] = False
        else:  # user-defined 'inputs'
            if not current['clean']:  # assume provided inputs are already formatted and normalized
                normputs = inputs
                if current['_suppress_normalization_warning'] is False:  # to suppress warning when evaluate called from coverage3
                    warnings.warn("User-provided 'inputs' but 'clean=False'. Subsequent errors may be solved by enabling automatic formatting and normalization of 'inputs' via 'clean=True'.", category=UserWarning)
        if current['clean']:
            normputs = self.clean(inputs, kwargs_from_other=kwargs_to_clean)
        elif inputs is None:
            normputs = self.inputs
        else:
            normputs = np.array(inputs)

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

        if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
            _, phind, xsm = self._inputs_to_phind(normputs)  # ..., phis=self.phis, kernel=self.kernel) already true
        elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
            phind = None
            xsm = normputs

        for i in range(n):
            for j in range(1, mbets):
                phi = 1
                for k in range(mputs):
                    num = mtx[j - 1, k]
                    if num > 0:
                        nid = int(num - 1)
                        if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
                            coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
                        elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
                            coeffs = phis[nid]  # coefficients for bernoulli
                        phi *= self.evaluate_basis(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.

                X[i, j] = phi

        X[:, 0] = np.ones((n,))
        modells = np.zeros((n, draws))  # note n == np.shape(data)[0] if data != 'ignore'
        for i in range(draws):
            modells[:, i] = np.transpose(np.matmul(X, np.transpose(np.array(betas[setnos[i], :]))))
        mean = np.mean(modells, 1)

        if current['ReturnBounds']:
            bounds = np.zeros((n, 2))  # note n == np.shape(data)[0] if data != 'ignore'
            cut = int(np.floor(draws * 0.025) + 1)
            for i in range(n):  # note n == np.shape(data)[0] if data != 'ignore'
                drawset = np.sort(modells[i, :])
                bounds[i, 0] = drawset[cut]
                bounds[i, 1] = drawset[draws - cut]
            return mean, bounds
        else:
            return mean

    def coverage3(self, **kwargs):
        """
        For validation testing of FoKL model. Default functionality is to evaluate all inputs (i.e., train+test sets).
        Returned is the predicted output 'mean', confidence bounds 'bounds', and root mean square error 'rmse'. A plot
        may be returned by calling 'coverage3(plot=1)'; or, for a potentially more meaningful plot in terms of judging
        accuracy, 'coverage3(plot='sorted')' plots the data in increasing value.

        Optional inputs for numerical evaluation of model:
            inputs == normalized and properly formatted inputs to evaluate              == self.inputs (default)
            data   == properly formatted data outputs to use for validating predictions == self.data (default)
            draws  == number of beta terms used                                         == self.draws (default)

        Optional inputs for basic plot controls:
            plot              == binary for generating plot, or 'sorted' for plot of ordered data == False (default)
            bounds            == binary for plotting bounds                                       == True (default)
            xaxis             == integer indexing the input variable to plot along the x-axis     == indices (default)
            labels            == binary for adding labels to plot                                 == True (default)
            xlabel            == string for x-axis label                                          == 'Index' (default)
            ylabel            == string for y-axis label                                          == 'Data' (default)
            title             == string for plot title                                            == 'FoKL' (default)
            legend            == binary for adding legend to plot                                 == True (default)
            LegendLabelFoKL   == string for FoKL's label in legend                                == 'FoKL' (default)
            LegendLabelData   == string for Data's label in legend                                == 'Data' (default)
            LegendLabelBounds == string for Bounds's label in legend                              == 'Bounds' (default)

        Optional inputs for detailed plot controls:
            PlotTypeFoKL   == string for FoKL's color and line type  == 'b' (default)
            PlotSizeFoKL   == scalar for FoKL's line size            == 2 (default)
            PlotTypeBounds == string for Bounds' color and line type == 'k--' (default)
            PlotSizeBounds == scalar for Bounds' line size           == 2 (default)
            PlotTypeData   == string for Data's color and line type  == 'ro' (default)
            PlotSizeData   == scalar for Data's line size            == 2 (default)

        Return Outputs:
            mean   == predicted output values for each indexed input
            bounds == confidence interval for each predicted output value
            rmse   == root mean squared deviation (RMSE) of prediction versus known data
        """

        # Process keywords:
        default = {
            # For numerical evaluation of model:
            'inputs': None, 'data': None, 'draws': self.draws,

            # For basic plot controls:
            'plot': False, 'bounds': True, 'xaxis': False, 'labels': True, 'xlabel': 'Index', 'ylabel': 'Data',
            'title': 'FoKL', 'legend': True, 'LegendLabelFoKL': 'FoKL', 'LegendLabelData': 'Data',
            'LegendLabelBounds': 'Bounds',

            # For detailed plot controls:
            'PlotTypeFoKL': 'b', 'PlotSizeFoKL': 2, 'PlotTypeBounds': 'k--', 'PlotSizeBounds': 2, 'PlotTypeData': 'ro',
            'PlotSizeData': 2
        }
        current = _process_kwargs(default, kwargs)
        if isinstance(current['plot'], str):
            if current['plot'].lower() in ['sort', 'sorted', 'order', 'ordered']:
                current['plot'] = 'sorted'
                if current['xlabel'] == 'Index':  # if default value
                    current['xlabel'] = 'Index (Sorted)'
            else:
                warnings.warn("Keyword input 'plot' is limited to True, False, or 'sorted'.", category=UserWarning)
                current['plot'] = False
        else:
            current['plot'] = _str_to_bool(current['plot'])
        for boolean in ['bounds', 'labels', 'legend']:
            current[boolean] = _str_to_bool(current[boolean])
        if current['labels']:
            for label in ['xlabel', 'ylabel', 'title']:  # check all labels are strings
                if current[label] and not isinstance(current[label], str):
                    current[label] = str(current[label])  # convert numbers to strings if needed (e.g., title=3)

        # Check for and warn about potential issues with user's 'input'/'data' combinations:
        if current['plot']:
            warn_plot = ' and ignoring plot.'
        else:
            warn_plot = '.'
        flip = [1, 0]
        flop = ['inputs', 'data']
        for i in range(2):
            j = flip[i]  # [i, j] = [[0, 1], [1, 0]]
            if current[flop[i]] is not None and current[flop[j]] is None:  # then data does not correspond to inputs
                warnings.warn(f"Keyword argument '{flop[j]}' should be defined to align with user-defined '{flop[i]}'. "
                              f"Ignoring RMSE calculation{warn_plot}", category=UserWarning)
                current['data'] = False  # ignore data when plotting and calculating RMSE
        if current['data'] is False and current['plot'] == 'sorted':
            warnings.warn("Keyword argument 'data' must correspond with 'inputs' if requesting a sorted plot. "
                          "Returning a regular plot instead.", category=UserWarning)
            current['plot'] = True  # regular plot

        # Define 'inputs' and 'data' if default (defined here instead of in 'default' to avoid lag for large datasets):
        if current['inputs'] is None:
            current['inputs'] = self.inputs
        if current['data'] is None:
            current['data'] = self.data

        def check_xaxis(current):
            """If plotting, check if length of user-defined x-axis aligns with length of inputs."""
            if current['xaxis'] is not False and not isinstance(current['xaxis'], int):  # then assume vector
                warn_xaxis = []
                l_xaxis = len(current['xaxis'])
                try:  # because shape any type of inputs is unknown, try lengths of different orientations
                    if l_xaxis != np.shape(current['inputs'])[0] and l_xaxis != np.shape(current['inputs'])[1]:
                        warn_xaxis.append(True)
                except:
                    warn_xaxis = warn_xaxis  # do nothing
                try:
                    if l_xaxis != np.shape(current['inputs'])[0]:
                        warn_xaxis.append(True)
                except:
                    warn_xaxis = warn_xaxis  # do nothing
                try:
                    if l_xaxis != len(current['inputs']):
                        warn_xaxis.append(True)
                except:
                    warn_xaxis = warn_xaxis  # do nothing
                if any(warn_xaxis):  # then vectors do not align
                    warnings.warn("Keyword argument 'xaxis' is limited to an integer indexing the input variable to "
                                  "plot along the x-axis (e.g., 0, 1, 2, etc.) or to a vector corresponding to 'data'. "
                                  "Leave blank (i.e., False) to plot indices along the x-axis.", category=UserWarning)
                    current['xaxis'] = False
            return current['xaxis']

        # Define local variables:
        normputs = current['inputs']  # assumes inputs are normalized and formatted correctly
        data = current['data']
        draws = current['draws']

        mean, bounds = self.evaluate(normputs, draws=draws, ReturnBounds=1, _suppress_normalization_warning=True)
        n, mputs = np.shape(normputs)  # Size of normalized inputs ... calculated in 'evaluate' but not returned

        if current['plot']:  # if user requested a plot
            current['xaxis'] = check_xaxis(current)  # check if custom xaxis can be plotted, else plot indices
            if current['xaxis'] is False:  # if default then plot indices
                plt_x = np.linspace(0, n - 1, n)  # indices
            elif isinstance(current['xaxis'], int):  # if user-defined but not a vector
                try:
                    normputs_np = np.array(normputs)  # in case list
                    min = self.minmax[current['xaxis']][0]
                    max = self.minmax[current['xaxis']][1]
                    plt_x = normputs_np[:, current['xaxis']] * (max - min) + min  # un-normalized vector for x-axis
                except:
                    warnings.warn(f"Keyword argument 'xaxis'={current['xaxis']} failed to index 'inputs'. Plotting indices instead.",
                                  category=UserWarning)
                    plt_x = np.linspace(0, n - 1, n)  # indices
            else:
                plt_x = current['xaxis']  # user provided vector for xaxis

            if current['plot'] == 'sorted':  # if user requested a sorted plot
                sort_id = np.argsort(np.squeeze(data))
                plt_mean = mean[sort_id]
                plt_bounds = bounds[sort_id]
                plt_data = data[sort_id]
            else:  # elif current['plot'] is True:
                plt_mean = mean
                plt_data = data
                plt_bounds = bounds

            plt.figure()
            plt.plot(plt_x, plt_mean, current['PlotTypeFoKL'], linewidth=current['PlotSizeFoKL'],
                     label=current['LegendLabelFoKL'])
            if data is not False:
                plt.plot(plt_x, plt_data, current['PlotTypeData'], markersize=current['PlotSizeData'],
                         label=current['LegendLabelData'])
            if current['bounds']:
                plt.plot(plt_x, plt_bounds[:, 0], current['PlotTypeBounds'], linewidth=current['PlotSizeBounds'],
                         label=current['LegendLabelBounds'])
                plt.plot(plt_x, plt_bounds[:, 1], current['PlotTypeBounds'], linewidth=current['PlotSizeBounds'])
            if current['labels']:
                if current['xlabel']:
                    plt.xlabel(current['xlabel'])
                if current['ylabel']:
                    plt.ylabel(current['ylabel'])
                if current['title']:
                    plt.title(current['title'])
            if current['legend']:
                plt.legend()

            plt.show()

        if data is not False:
            rmse = np.sqrt(np.mean(mean - data) ** 2)
        else:
            rmse = []

        return mean, bounds, rmse

    def fit(self, inputs=None, data=None, **kwargs):
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
        expected = self.hypers + list(default_for_fit.keys()) + list(default_for_clean.keys())
        kwargs = _process_kwargs(expected, kwargs)
        if default_for_fit['clean'] is False:
            if any(kwarg in default_for_clean.keys() for kwarg in kwargs.keys()):
                warnings.warn("Keywords for automatic cleaning were defined but clean=False.")
            default_for_clean = {}  # not needed for future since 'clean=False'

        # Process keyword arguments and update/define class attributes:
        kwargs_to_clean = {}
        for kwarg in kwargs.keys():
            if kwarg in self.hypers:  # for case of user sweeping through hyperparameters within 'fit' argument
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
        if default_for_fit['clean'] is True:
            try:
                if inputs is None:  # assume clean already called and len(data) same as train data if data not None
                    inputs, _ = self.trainset()
                if data is None:  # assume clean already called and len(inputs) same as train inputs if inputs not None
                    _, data = self.trainset()
            except Exception as exception:
                error_clean_failed = True
            self.clean(inputs, data, kwargs_from_other=kwargs_to_clean, _setattr=True)
        else:  # user input implies that they already called clean prior to calling fit
            try:
                if inputs is None:  # assume clean already called and len(data) same as train data if data not None
                    inputs, _ = self.trainset()
                if data is None:  # assume clean already called and len(inputs) same as train inputs if inputs not None
                    _, data = self.trainset()
            except Exception as exception:
                warnings.warn("Keyword 'clean' was set to False but is required prior to or during 'fit'. Assuming "
                              "'clean' is True.", category=UserWarning)
                if inputs is None or data is None:
                    error_clean_failed = True
                else:
                    default_for_fit['clean'] = True
                    self.clean(inputs, data, kwargs_from_other=kwargs_to_clean, _setattr=True)
        if error_clean_failed is True:
            raise ValueError("'inputs' and/or 'data' were not provided so 'clean' could not be performed.")

        # After cleaning and/or handling exceptions, define cleaned 'inputs' and 'data' as local variables:
        try:
            inputs, data = self.trainset()
        except Exception as exception:
            warnings.warn("If not calling 'clean' prior to 'fit' or within the argument of 'fit', then this is the "
                          "likely source of any subsequent errors. To troubleshoot, simply include 'clean=True' within "
                          "the argument of 'fit'.", category=UserWarning)

        # Define attributes as local variables:
        phis = self.phis
        relats_in = self.relats_in
        a = self.a
        b = self.b
        atau = self.atau
        btau = self.btau
        tolerance = self.tolerance
        draws = self.burnin + self.draws  # after fitting, the 'burnin' draws will be discarded from 'betas'
        gimmie = self.gimmie
        way3 = self.way3
        threshav = self.threshav
        threshstda = self.threshstda
        threshstdb = self.threshstdb
        aic = self.aic



        # Update 'b' and/or 'btau' if set to default:
        if btau is None or b is None:  # then use 'data' to define (in combination with 'a' and/or 'atau')
            # Calculate variance and mean, both as 64-bit, but for large datasets (i.e., less than 64-bit) be careful
            # to avoid converting the entire 'data' to 64-bit:
            if data.dtype != np.float64:  # and sigmasq == math.inf  # then recalculate but as 64-bit
                sigmasq = 0
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
        if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
            _, phind, xsm = self._inputs_to_phind(inputs)  # ..., phis=self.phis, kernel=self.kernel) already true
        elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
            phind = None
            xsm = inputs

        # [BEGIN] initialization of constants (for use in gibbs to avoid repeat large calculations):

        if self.update == True:
            self.betas, self.mtx, self.evs = self.fitupdate(inputs, data)
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

        def gibbs(inputs, data, phis, Xin, discmtx, a, b, atau, btau, draws, phind, xsm, sigsqd, tausqd, dtd):
            """
            'inputs' is the set of normalized inputs -- both parameters and model
            inputs -- with columns corresponding to inputs and rows the different
            experimental designs. (numpy array)

            'data' are the experimental results: column vector, with entries
            corresponding to rows of 'inputs'

            'phis' are a data structure with the coefficients for the basis
            functions

            'discmtx' is the interaction matrix for the bss-anova function -- rows
            are terms in the function and columns are inputs (cols should line up
            with cols in 'inputs'

            'a' and 'b' are the parameters of the ig distribution for the
            observation error variance of the data

            'atau' and 'btau' are the parameters of the ig distribution for the 'tau
            squared' parameter: the variance of the beta priors

            'draws' is the total number of draws

            Additional Constants (to avoid repeat calculations found in later development):
                - phind
                - xsm
                - sigsqd
                - tausqd
                - dtd
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
                            if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
                                coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
                            elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
                                coeffs = phis[nid]  # coefficients for bernoulli
                            phi = phi * self.evaluate_basis(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.

                    X[i][j] = phi

            # # initialize tausqd at the mode of its prior: the inverse of the mode of sigma squared, such that the
            # # initial variance for the betas is 1
            # sigsqd = b / (1 + a)
            # tausqd = btau / (1 + atau)

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

            n = len(data)
            astar = a + 1 + n / 2 + (mmtx + 1) / 2

            atau_star = atau + mmtx / 2

            # Gibbs iterations

            betas = np.zeros((draws, mmtx + 1))
            sigs = np.zeros((draws, 1))
            taus = np.zeros((draws, 1))
            lik = np.zeros((draws, 1))

            for k in range(draws):

                Lamb_tausqd = np.diag(Lamb) + (1 / tausqd) * np.identity(mmtx + 1)
                Lamb_tausqd_inv = np.diag(1 / np.diag(Lamb_tausqd))

                mun = Q.dot(Lamb_tausqd_inv).dot(np.transpose(Q)).dot(Xty)
                S = Q.dot(np.diag(np.diag(Lamb_tausqd_inv) ** (1 / 2)))

                vec = np.random.normal(loc=0, scale=1, size=(mmtx + 1, 1))  # drawing from normal distribution
                betas[k][:] = np.transpose(mun + sigsqd ** (1 / 2) * (S).dot(vec))

                vecc = mun - np.reshape(betas[k][:], (len(betas[k][:]), 1))

                bstar = b + 0.5 * (betas[k][:].dot(XtX.dot(np.transpose([betas[k][:]]))) - 2 * betas[k][:].dot(Xty) +
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

                [beters, null, null, null, xers, ev] = gibbs(inputs, data, phis, X, damtx, a, b, atau, btau, draws,
                                                             phind, xsm, sigsqd0, tausqd0, dtd)

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

                        [betertest, null, null, null, Xtest, evtest] = gibbs(inputs, data, phis, X, damtx_test, a, b,
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

        self.betas = betas[-self.draws::, :]  # discard 'burnin' draws by only keeping last 'draws' draws
        self.mtx = mtx
        self.evs = evs

        return betas[-self.draws::, :], mtx, evs  # discard 'burnin'

    def clear(self, keep=None, clear=None, all=False):
        """
        Delete all attributes from the FoKL class except for hyperparameters and settings by default, but user may
        specify otherwise. If an attribute is listed in both 'clear' and 'keep', then the attribute is cleared.

        Optional Inputs:
            keep (list of strings)  == additional attributes to keep, e.g., ['mtx']
            clear (list of strings) == hyperparameters to delete, e.g., ['kernel', 'phis']
            all (boolean)           == if True then all attributes (including hyperparameters) get deleted regardless

        Tip: To remove all attributes, simply call 'self.clear(all=1)'.
        """

        if all is not False:  # if not default
            all = _str_to_bool(all)  # convert to boolean if all='on', etc.

        if all is False:
            attrs_to_keep = self.keep  # default
            if isinstance(keep, list) or isinstance(keep, str):  # str in case single entry (e.g., keep='mtx')
                attrs_to_keep += keep  # add user-specified attributes to list of ones to keep
                attrs_to_keep = list(np.unique(attrs_to_keep))  # remove duplicates
            if isinstance(clear, list) or isinstance(clear, str):
                for attr in clear:
                    attrs_to_keep.remove(attr)  # delete attribute from list of ones to keep
        else:  # if all=True
            attrs_to_keep = []  # empty list so that all attributes get deleted

        attrs = list(vars(self).keys())  # list of all currently defined attributes
        for attr in attrs:
            if attr not in attrs_to_keep:
                delattr(self, attr)  # delete attribute from FoKL class if not keeping

        return

    def to_pyomo(self, xvars, yvars, m=None, xfix=None, yfix=None, truescale=True, std=True, draws=None):
        """Passes arguments to external function. See 'fokl_to_pyomo' for more documentation."""
        return fokl_to_pyomo(self, xvars, yvars, m, xfix, yfix, truescale, std, draws)

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

# Beginning of UPDATE Code:

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
            phis = self.phis
            relats_in = self.relats_in
            a = self.a
            b = self.b
            atau = self.atau
            btau = self.btau
            tolerance = self.tolerance
            draws = self.burnin + self.draws  # after fitting, the 'burnin' draws will be discarded from 'betas'
            gimmie = self.gimmie
            way3 = self.way3
            aic = self.aic
            burn = self.burn # burn draws are disregarded prior to update fitting
            sigsqd0 = self.sigsqd0


            def modelBuilder():
                if self.built:
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
                                if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
                                    coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
                                elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
                                    coeffs = phis[nid]  # coefficients for bernoulli
                                phi = phi * self.evaluate_basis(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.

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
            if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
                _, phind, xsm = self._inputs_to_phind(inputs)  # ..., phis=self.phis, kernel=self.kernel) already true
            elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
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
                            self.built = True
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

