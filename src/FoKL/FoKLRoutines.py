from src.FoKL import getKernels  # DEV PRIOR2PULL
import pandas as pd
import warnings
import itertools
import math
import numpy as np
from numpy import linalg as LA
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import time
import os
import pickle


def load(filename, directory=None):
    """
    Load a FoKL class from a file. By default, the 'directory' is the current working directory that contains the
    file calling this method. The 'directory' may be named relative to the directory if not default.

    Enter the returned output from 'self.save()' as the argument here, leaving 'directory' blank, to later reload a
    model. This is possible because 'filename' can simply be defined as the 'filepath' for 'directory=None'.
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


def str_to_bool(s):
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


def process_kwargs(default, user):
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


def set_attributes(self, attrs):
    """Set items stored in Python dictionary 'attrs' as attributes of class."""
    if isinstance(attrs, dict):
        for key, value in attrs.items():
            setattr(self, key, value)
    else:
        warnings.warn("Input must be a Python dictionary.")
    return


def inputs_to_phind(self, inputs, phis, DefineAttributes=False, kernel=None):
    """
    Twice normalize the inputs to index the spline coefficients.

    Inputs:
        - inputs == normalized inputs as numpy array (i.e., self.inputs.np)
        - phis   == spline coefficients
    Optional Input:
        - DefineAttributes == boolean for updating class attributes with phind and xsm == False (default)

    Output (and appended class attributes):
        - phind        == index to spline coefficients
        - xsm          ==
        - inputs_2norm == twice normalized inputs
    """
    if kernel is None:
        kernel = self.kernel

    if kernel == self.kernels[0]:  # == 'Cubic Splines':
        l_phis = len(phis[0][0])  # = 499, length of cubic splines in basis functions
    elif kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
        warnings.warn("Twice normalization of inputs is not required for the 'Bernoulli Polynomials' kernel",
                      category=UserWarning)
        return inputs, [], []

    phind = np.array(np.ceil(inputs * l_phis), dtype=int)  # 0-1 normalization to 0-499 normalization

    if phind.ndim == 1:  # if phind.shape == (number,) != (number,1), then add new axis to match indexing format
        phind = phind[:, np.newaxis]

    set = (phind == 0)  # set = 1 if phind = 0, otherwise set = 0
    phind = phind + set  # makes sense assuming L_phis > M

    r = 1 / l_phis  # interval of when basis function changes (i.e., when next cubic function defines spline)
    xmin = (phind - 1) * r
    X = (inputs - xmin) / r  # twice normalized inputs (0-1 first then to size of phis second)

    phind = phind - 1
    xsm = l_phis * inputs - phind

    if DefineAttributes:
        self.inputs_2norm = X  # twice normalized inputs
        self.phind = phind  # adjust MATLAB indexing to Python indexing after twice normalization
        self.xsm = xsm

    return X, phind, xsm


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

            - 'draws' is the total number of draws from the posterior for each tested model.

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
            - draws      = 1000
            - gimmie     = False
            - way3       = False
            - threshav   = 0.05
            - threshstda = 0.5
            - threshstdb = 2
            - aic        = False

        Other Optional Inputs:
            - UserWarnings  == boolean to print user-warnings to the command terminal          == False (default)
            - ConsoleOutput == boolean to print [ind, ev] during 'fit' to the command terminal == True (default)
        """

        # Store list of hyperparameters for easy reference later, if sweeping through values in functions such as fit:
        self.hypers = ['kernel', 'phis', 'relats_in', 'a', 'b', 'atau', 'btau', 'tolerance', 'draws', 'gimmie', 'way3',
                       'threshav', 'threshstda', 'threshstdb', 'aic']

        # Store supported kernels for later logical checks against 'kernel':
        self.kernels = ['Cubic Splines', 'Bernoulli Polynomials']

        # Process user's keyword arguments:
        default = {
                   # Hyperparameters:
                   'kernel': 'Cubic Splines', 'phis': None, 'relats_in': [], 'a': 4, 'b': None, 'atau': 4,
                   'btau': None, 'tolerance': 3, 'draws': 1000, 'gimmie': False, 'way3': False, 'threshav': 0.05,
                   'threshstda': 0.5, 'threshstdb': 2, 'aic': False,

                   # Other:
                   'UserWarnings': False, 'ConsoleOutput': True
                   }
        current = process_kwargs(default, kwargs)  # = default, but updated by any user kwargs
        for boolean in ['gimmie', 'way3', 'aic', 'UserWarnings']:
            if not (current[boolean] is False or current[boolean] is True):
                current[boolean] = str_to_bool(current[boolean])

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

    def clean(self, inputs, data=None, kwargs_from_fit=None, **kwargs):
        """
        For cleaning and formatting inputs prior to training a FoKL model. Note that data is not required but should be
        entered if available; otherwise, leave blank.

        Inputs:
            inputs == NxM matrix of independent (or non-linearly dependent) 'x' variables for fitting f(x1, ..., xM)
            data   == Nx1 vector of dependent variable to create model for predicting the value of f(x1, ..., xM)

        Keyword Inputs:
            train                == percentage (0-1) of N datapoints to use for training           == 1 (default)
            TrainMethod          == method for splitting test/train set for train < 1              == 'random' (default)
            CatchOutliers        == logical for removing outliers from inputs and/or data          == False (default)
            OutliersMethod       == the method for removing outliers (e.g., 'Z-Score)              == None (default)
            OutliersMethodParams == parameters to modify OutliersMethod (format varies per method) == None (default)
            kwargs_from_fit      == used internally by fit function (not for user)

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

        Note additional methods for automatically cleaning data are in development.
        """

        # Process keywords:
        if kwargs_from_fit is not None:  # then clean is being called from fit function
            kwargs = kwargs | kwargs_from_fit  # merge dictionaries (kwargs={} is expected from fit but just in case)
        default = {'train': 1, 'TrainMethod': 'random', 'CatchOutliers': False, 'OutliersMethod': None,
                   'OutliersMethodParams': None}
        current = process_kwargs(default, kwargs)

        # Define local variables from keywords:
        p_train = current['train']  # percentage of datapoints to use as training data
        p_train_method = current['TrainMethod']
        CatchOutliers = current['CatchOutliers']
        OutliersMethod = current['OutliersMethod']
        OutliersMethodParams = current['OutliersMethodParams']

        if isinstance(CatchOutliers, bool):
            CatchOutliers = 1 * CatchOutliers
        elif isinstance(CatchOutliers, str):  # then convert user input to logical for auto_cleanData to interpret
            if CatchOutliers.lower() in ('inputs', 'input', 'in', 'x'):
                CatchOutliers = [1, 0]  # note 1 will need to be replicated later to match number of input variables
            elif CatchOutliers.lower() in ('data', 'outputs', 'output', 'out', 'y'):
                CatchOutliers = [0, 1]  # note 0 will need to be replicated later to match number of input variables
            else:  # apply to both inputs and data, i.e., not a vector but a single True/False
                CatchOutliers = 1 * str_to_bool(CatchOutliers)  # 0 or 1
        elif isinstance(CatchOutliers, np.ndarray): # then assume vector (which is goal)
            if CatchOutliers.ndim != 1:
                CatchOutliers = np.squeeze(CatchOutliers)
                if CatchOutliers.ndim != 1:
                    raise ValueError("CatchOutliers, if being applied to a user-selected inputs+data combo, must be a "
                                     "boolean 1D list (e.g., [0,1,...,1,0]) corresponding to [input1, input2, ..., "
                                     "inputM, data].")
            CatchOutliers = list(CatchOutliers)  # should return 1D list
        elif isinstance(CatchOutliers, tuple):
            CatchOutliers = list(CatchOutliers)
        elif not isinstance(CatchOutliers, list):
            raise ValueError("CatchOutliers must be defined as 'inputs', 'data', 'all', or a boolean 1D list "
                             "(e.g., [0,1,...,1,0]) corresponding to [input1, input2, ..., inputM, data].")
        # Note at this point CatchOutliers might be 0, 1, [1, 0], [0, 1, 0, 0], etc.

        # Automatically handle some data formatting exceptions:
        def auto_cleanData(inputs, p_train, CatchOutliers, OutliersMethod, OutliersMethodParams, data=data):

            # Convert 'inputs' and 'datas' to numpy if pandas:
            if any(isinstance(inputs, type) for type in (pd.DataFrame, pd.Series)):
                inputs = inputs.to_numpy()
                warnings.warn(
                    "'inputs' was auto-converted to numpy. Convert manually for assured accuracy.", UserWarning)
            if data is not None:
                if any(isinstance(data, type) for type in (pd.DataFrame, pd.Series)):
                    data = data.to_numpy()
                    warnings.warn("'data' was auto-converted to numpy. Convert manually for assured accuracy.", UserWarning)

            # Normalize 'inputs' and convert to proper format for FoKL:
            inputs = np.array(inputs) # attempts to handle lists or any other format (i.e., not pandas)
            # . . . inputs = {ndarray: (N, M)} = {ndarray: (datapoints, input variables)} =
            # . . . . . . array([[x1(t1),x2(t1),...,xM(t1)],[x1(t2),x2(t2),...,xM(t2)],...,[x1(tN),x2(tN),...,xM(tN)]])
            inputs = np.squeeze(inputs) # removes axes with 1D for cases like (N x 1 x M) --> (N x M)
            if inputs.dtype != np.float:
                inputs = np.array(inputs, dtype=np.float)
                warnings.warn("'inputs' was converted to float64. May require user-confirmation that values did not get"
                              "corrupted.", category=UserWarning)
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

            if data is not None:
                # Transpose 'data' if needed:
                data = np.array(data)  # attempts to handle lists or any other format (i.e., not pandas)
                data = np.squeeze(data)
                if data.dtype != np.float:
                    data = np.array(data, dtype=np.float)
                    warnings.warn(
                        "'data' was converted to float64. May require user-confirmation that values did not get"
                        "corrupted.", category=UserWarning)
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

                    elif not (OutliersMethod is None or OutliersMethod == []):
                        raise ValueError("Keyword argument 'OutliersMethod' is limited to 'Z-Score'. Other methods are "
                                         "in development.")

                    inputs_data_wo_outliers = inputs_data[~outliers_indices, :]
                    inputs_wo_outliers = inputs_data_wo_outliers[:, :-1]
                    data_wo_outliers = inputs_data_wo_outliers[:, -1]

                return inputs_wo_outliers, data_wo_outliers, outliers_indices

            if OutliersMethod is not None and data is not None:
                inputs, data, outliers_indices = catch_outliers(inputs, data, CatchOutliers, OutliersMethod,
                                                                OutliersMethodParams)
            else:
                outliers_indices = []

            # Spit [inputs,data] into train and test sets (depending on TrainMethod):
            if p_train < 1: # split inputs+data into training and testing sets for validation of model
                def random_train(p_train, inputs, data=data): # random split, if TrainMethod = 'random'
                    Ldata = inputs.shape[0]
                    l_log = int(Ldata * p_train)  # required length of indices for training
                    train_log_i = np.array([])  # random indices used for training data
                    while len(train_log_i) < l_log:
                        train_log_i = np.append(train_log_i, np.random.random_integers(Ldata, size=l_log) - 1)
                        train_log_i = np.unique(train_log_i)  # incidentally sorts low to high
                        np.random.shuffle(train_log_i)  # randomize order
                    if len(train_log_i) > l_log:
                        train_log_i = train_log_i[0:l_log]  # cut-off extra indices (beyond p_train)
                    train_log = np.zeros(Ldata, dtype=np.bool)  # indices used for training data (as a logical)
                    for i in train_log_i:
                        train_log[int(i)] = True
                    test_log = ~train_log

                    inputs_train = [inputs[ii] for ii, ele in enumerate(train_log) if ele]
                    inputs_test = [inputs[ii] for ii, ele in enumerate(test_log) if ele]
                    if data is not None:
                        data_train = data[train_log]
                        data_test = data[test_log]
                    else:
                        data_train = None
                        data_test = None

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
                        function_mapping[p_train_method](p_train, inputs, data=data)
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
            outliers_indices, train_log, test_log = auto_cleanData(inputs, p_train, CatchOutliers, OutliersMethod,
                                                                   OutliersMethodParams, data=data)

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

        inputs_np, do_transpose = inputslist_to_np(inputs, 'auto')
        rawinputs_np = inputslist_to_np(rawinputs, do_transpose)[0]
        traininputs_np = inputslist_to_np(traininputs, do_transpose)[0]
        testinputs_np = inputslist_to_np(testinputs, do_transpose)[0]

        # Define/update attributes with cleaned data and other relevant variables:
        attrs = {'inputs': inputs, 'data': data, 'rawinputs': rawinputs, 'traininputs': traininputs,
                 'traindata': traindata, 'testinputs': testinputs, 'testdata': testdata, 'normalize': inputs_scale,
                 'outliers': outliers_indices, 'trainlog': train_log, 'testlog': test_log, 'inputs_np': inputs_np,
                 'rawinputs_np': rawinputs_np, 'traininputs_np': traininputs_np, 'testinputs_np': testinputs_np}
        # Notes (if not in documentation):
        #   - self.normalize == [min,max] of each input before normalization
        #   - outliers       == indices removed from raw
        #   - trainlog       == indices used for training AFTER OUTLIERS WERE REMOVED from raw
        #   - testlog        == indices used for validation testing AFTER OUTLIERS WERE REMOVED from raw
        set_attributes(self, attrs)

        return

    def bss_derivatives(self, **kwargs):
        """
        For returning gradient of modeled function with respect to each, or specified, input variable.
        If user overrides default settings, then 1st and 2nd partial derivatives can be returned for any variables.

        Keyword Inputs:
            inputs == NxM matrix of 'x' input variables for fitting f(x1, ..., xM)    == self.inputs_np (default)
            kernel == function to use for differentiation (i.e., cubic or Bernoulli)  == self.kernel (default)
            d1     == index of input variable(s) to use for first partial derivative  == True (default)
            d2     == index of input variable(s) to use for second partial derivative == False (default)
            draws  == number of beta terms used                                       == self.draws (default)
            betas  == draw from the posterior distribution of coefficients            == self.betas (default)
            phis   == spline coefficients for the basis functions                     == self.phis (default)
            mtx    == basis function interaction matrix from the best model           == self.mtx (default)
            span   == list of [min, max]'s of input data used in the normalization    == self.normalize (default)
            IndividualDraws == boolean for returning derivative(s) at each draw       == 0 (default)
            ReturnFullArray == boolean for returning NxMx2 array instead of Nx2M      == 0 (default)

        Return Outputs:
            dState == derivative of input variables (i.e., states)

        Notes:
            - To turn off all the first-derivatives, set d1=False instead of d1=0. 'd1' and 'd2', if set to an integer,
            will return the derivative with respect to the input variable indexed by that integer using Python indexing.
            In other words, for a two-input FoKL model, setting d1=1 and d2=0 will return the first-derivative with
            respect to the second input (d1=1) and the second-derivative with respect to the first input (d2=0).
            Alternatively, d1=[False, True] and d2=[True, False] will function the same.
        """

        # Process keywords:
        default = {'inputs': None, 'kernel': self.kernel, 'd1': None, 'd2': None, 'draws': self.draws, 'betas': None,
                   'phis': None, 'mtx': self.mtx, 'span': self.normalize, 'IndividualDraws': False,
                   'ReturnFullArray': False, 'ReturnBasis': False}
        current = process_kwargs(default, kwargs)
        for boolean in ['IndividualDraws', 'ReturnFullArray', 'ReturnBasis']:
            current[boolean] = str_to_bool(current[boolean])

        # Load defaults:
        if current['inputs'] is None:
            current['inputs'] = self.inputs_np
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
        span = current['span']

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
                if str_to_bool(di):
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
            X, phind, _ = inputs_to_phind(self, inputs, phis, kernel=kernel)  # phind needed for non-continuous kernel
            L_phis = len(phis[0][0])  # = 499
        elif kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
            X = inputs  # twice-normalization not required
            L_phis = 1  # because continuous
        basis_nm = np.zeros([N, M, B])  # constant term for (n,md,b) that avoids repeat calculations
        dState = np.zeros([draws, N, M, 2])
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

                            dState[:,n,m,di] = dState[:,n,m,di] + betas[-draws:,b+1]*phi[n,m,di]  # update after md loop

        dState = np.transpose(dState, (1, 2, 3, 0))  # move draws dimension so dState has form (N,M,di,draws)

        if not current['IndividualDraws'] and draws > 1:  # then average draws
            dState = np.mean(dState, axis=3)  # note 3rd axis index (i.e., 4th) automatically removed (i.e., 'squeezed')
            dState = dState[:, :, :, np.newaxis]  # add new axis to avoid error in concatenate below

        if not current['ReturnFullArray']:  # then return only columns with values and stack d1 and d2 next to each other
            dState = np.concatenate([dState[:, :, 0, :], dState[:, :, 1, :]], axis=1)  # (N,M,2,draws) to (N,2M,draws)
            dState = dState[:, ~np.all(dState == 0, axis=0)]  # remove columns ('N') with all zeros
        dState = np.squeeze(dState)  # remove unnecessary axes

        if current['ReturnBasis']:  # mostly for development
            return dState, basis
        else:  # standard
            return dState

    def evaluate_basis(self, c, x, kernel=None, d=0):
        """
        Evaluate a basis function at a single point by providing coefficients, x value(s), and (optionally) the kernel.

        Inputs:
            > c == coefficients of a single basis functions
            > x == value(s) of independent variable at which to evaluate the basis function

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
            kernel = self.kernel  # assume supported
        elif kernel not in self.kernels:  # check user's provided kernel is supported
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
            inputs == input variable(s) at which to evaluate the FoKL model == self.inputs_np (default)

        Optional Inputs:
            betas        == coefficients defining FoKL model                       == self.betas (default)
            mtx          == interaction matrix defining FoKL model                 == self.mtx (default)
            normalize    == [min, max] of inputs used for normalization            == self.normalize (default)
            draws        == number of beta terms used                              == self.draws (default)
            clean        == boolean to automatically normalize and format 'inputs' == False (default)
            ReturnBounds == boolean to return confidence bounds as second output   == False (default)
        """

        # Process keywords:
        default = {'normalize': None, 'draws': self.draws, 'clean': False, 'ReturnBounds': False}
        current = process_kwargs(default, kwargs)
        for boolean in ['clean', 'ReturnBounds']:
            current[boolean] = str_to_bool(current[boolean])
        draws = current['draws']  # define local variable
        if betas is None:  # default
            betas = self.betas
        else:  # user-defined betas may need to be formatted
            betas = np.array(betas)
            if betas.ndim == 1:
                betas = betas[np.newaxis, :]  # note transpose would be column of beta0 terms, so not expected
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

        if draws > betas.shape[0]:
            draws = betas.shape[0]  # more draws than models results in inf time, so threshold
        phis = self.phis

        # Automatically normalize and format inputs:
        if inputs is None:  # default
            inputs = self.inputs_np
            if current['clean']:
                warnings.warn("Cleaning was already performed on default 'inputs', so overriding 'clean' to False.",
                              category=UserWarning)
                current['clean'] = False
        else:  # user-defined 'inputs'
            if not current['clean']:  # assume provided inputs are already normalized and formatted, but check
                normputs = inputs
                if not isinstance(normputs, np.ndarray):
                    warnings.warn("Provided inputs were converted to numpy array as float64. May want to manually "
                                  "check that values were preserved.", category=UserWarning)
                    normputs = np.array(normputs, dtype=np.float)
                if np.max(normputs) > 1 or np.min(normputs) < 0:
                    warnings.warn("Provided inputs were not normalized, so overriding 'clean' to True.")
                    current['clean'] = True
        if current['clean']:
            self.clean(inputs)
            normputs = self.inputs_np

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
            _, phind, xsm = inputs_to_phind(self, normputs, phis)
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
            modells[:, i] = np.matmul(X, betas[setnos[i], :])
        meen = np.mean(modells, 1)

        if current['ReturnBounds']:
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
        For validation testing of FoKL model. Default functionality is to evaluate all inputs (i.e., train+test sets).
        Returned is the predicted output 'meen', confidence bounds 'bounds', and root mean square error 'rmse'. A plot
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
            meen   == predicted output values for each indexed input
            rmse   == root mean squared deviation (RMSE) of prediction versus known data
            bounds == confidence interval for each predicted output value
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
        current = process_kwargs(default, kwargs)
        if isinstance(current['plot'], str):
            if current['plot'].lower() in ['sort', 'sorted', 'order', 'ordered']:
                current['plot'] = 'sorted'
                if current['xlabel'] == 'Index':  # if default value
                    current['xlabel'] = 'Index (Sorted)'
            else:
                warnings.warn("Keyword input 'plot' is limited to True, False, or 'sorted'.", category=UserWarning)
                current['plot'] = False
        else:
            current['plot'] = str_to_bool(current['plot'])
        for boolean in ['bounds', 'labels', 'legend']:
            current[boolean] = str_to_bool(current[boolean])
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
                    if l_xaxis != len(current['inputs'][:, 0]) and l_xaxis != len(current['inputs'][0, :]):
                        warn_xaxis.append(True)
                except:
                    warn_xaxis = warn_xaxis  # do nothing
                try:
                    if l_xaxis != len(current['inputs'][0]):
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

        meen, bounds = self.evaluate(normputs, draws=draws, ReturnBounds=1)
        n, mputs = np.shape(normputs)  # Size of normalized inputs ... calculated in 'evaluate' but not returned

        if current['plot']:  # if user requested a plot
            current['xaxis'] = check_xaxis(current)  # check if custom xaxis can be plotted, else plot indices
            if current['xaxis'] is False:  # if default then plot indices
                plt_x = np.linspace(0, n - 1, n)  # indices
            elif isinstance(current['xaxis'], int):  # if user-defined but not a vector
                try:
                    normputs_np = np.array(normputs)  # in case list
                    min = self.normalize[current['xaxis']][0]
                    max = self.normalize[current['xaxis']][1]
                    plt_x = normputs_np[:, current['xaxis']] * (max - min) + min  # un-normalized vector for x-axis
                except:
                    warnings.warn(f"Keyword argument 'xaxis'={current['xaxis']} failed to index 'inputs'. Plotting indices instead.",
                                  category=UserWarning)
                    plt_x = np.linspace(0, n - 1, n)  # indices

            if current['plot'] == 'sorted':  # if user requested a sorted plot
                sort_id = np.argsort(np.squeeze(data))
                plt_meen = meen[sort_id]
                plt_bounds = bounds[sort_id]
                plt_data = data[sort_id]
            else:  # elif current['plot'] is True:
                plt_meen = meen
                plt_data = data
                plt_bounds = bounds

            plt.figure()
            plt.plot(plt_x, plt_meen, current['PlotTypeFoKL'], linewidth=current['PlotSizeFoKL'],
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
            rmse = np.sqrt(np.mean(meen - data) ** 2)
        else:
            rmse = []

        return meen, bounds, rmse

    def fit(self, inputs, data, **kwargs):
        """
        For fitting model to known inputs and data (i.e., training of model).

        Inputs:
            inputs == NxM matrix of independent (or non-linearly dependent) 'x' variables for fitting f(x1, ..., xM)
            data   == Nx1 vector of dependent variable to create model for predicting the value of f(x1, ..., xM)

        Keyword Inputs (for fit):
            clean         == boolean to perform automatic cleaning and formatting               == True (default)
            ConsoleOutput == boolean to print [ind, ev] to console during FoKL model generation == True (default)

        Keyword Inputs (for clean):
            train                == percentage (0-1) of N datapoints to use for training  == True (default)
            TrainMethod          == method for splitting test/train set for train < 1     == 'random' (default)
            CatchOutliers        == boolean for removing outliers from inputs and/or data == False (default)
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
            - Various ... please see description of 'clean()'
        """

        # Check for unexpected keyword arguments:
        default_for_fit = {}
        default_for_fit['ConsoleOutput'] = str_to_bool(kwargs.get('ConsoleOutput', self.ConsoleOutput))
        default_for_fit['clean'] = str_to_bool(kwargs.get('clean', True))
        default_for_clean = {'train': 1, 'TrainMethod': 'random', 'CatchOutliers': False, 'OutliersMethod': None,
                             'OutliersMethodParams': None}
        expected = self.hypers + list(default_for_fit.keys()) + list(default_for_clean.keys())
        kwargs = process_kwargs(expected, kwargs)
        if default_for_fit['clean'] is False:
            if any(kwarg in default_for_clean.keys() for kwarg in kwargs.keys()):
                warnings.warn("Keywords for automatic cleaning were defined but clean=False.")
            default_for_clean = {}  # not needed for future since 'clean=False'

        # Process keyword arguments and update/define class attributes:
        kwargs_to_clean = {}
        for kwarg in kwargs.keys():
            if kwarg in self.hypers:  # for case of user sweeping through hyperparameters within 'fit' argument
                if kwarg in ['gimmie', 'way3', 'aic']:
                    setattr(self, kwarg, str_to_bool(kwargs[kwarg]))
                else:
                    setattr(self, kwarg, kwargs[kwarg])
            elif kwarg in default_for_clean.keys():
                if kwarg in ['CatchOutliers']:
                    kwargs_to_clean.update({kwarg: str_to_bool(kwargs[kwarg])})
                else:
                    kwargs_to_clean.update({kwarg: kwargs[kwarg]})
        self.ConsoleOutput = default_for_fit['ConsoleOutput']

        # Perform automatic cleaning of 'inputs' and 'data' (unless user already called 'fit' and now specifies not to):
        if default_for_fit['clean']:
            self.clean(inputs, data, kwargs_from_fit=kwargs_to_clean)
        else:  # user input implies that they already called clean but are fitting again (e.g., to check stochasticity)
            try:
                inputs = self.traininputs
            except:
                warnings.warn("Keyword 'clean' was set to False but is required when first running fit. Assuming "
                              "'clean' is True so that attributes (i.e., 'traininputs', etc.) get defined.",
                              category=UserWarning)
                self.clean(inputs, data, kwargs_from_fit=kwargs_to_clean)

        # Define attributes as local variables:
        inputs = self.traininputs
        data = self.traindata
        inputs_np = self.traininputs_np
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
        if btau is None or b is None:  # then use 'data' to define (in combination with 'a' and/or 'atau')
            sigmasq = np.var(data)
            if b is None:
                b = sigmasq * (a + 1)
                self.b = b
            if btau is None:
                scale = np.abs(np.mean(data))
                btau = (scale / sigmasq) * (atau + 1)
                self.btau = btau

        def perms(x):
            """Python equivalent of MATLAB perms."""
            # from https://stackoverflow.com/questions/38130008/python-equivalent-for-matlabs-perms
            a = np.vstack(list(itertools.permutations(x)))[::-1]
            return a

        # Prepare phind and xsm if using cubic splines, else match variable names required for gibbs argument
        if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
            _, phind, xsm = inputs_to_phind(self, inputs_np, phis)
        elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
            phind = None
            xsm = inputs_np

        def gibbs(inputs, data, phis, Xin, discmtx, a, b, atau, btau, draws, phind, xsm):
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

            Additional Function Arguments for Constants to Avoid Repeat Calculations and Optimize Performance (v3.1.1):
                - phind
                - xsm
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
                            # phi = phi * (phis[nid][0][phind[i, k]] + phis[nid][1][phind[i, k]] * xsm[i, k] + \
                            #     phis[nid][2][phind[i, k]] * xsm[i, k] ** 2 + phis[nid][3][phind[i, k]] * xsm[i, k] ** 3)

                            # [v3.1.1] With inclusion of Bernoulli polynomial kernel, the above function may change. So,
                            if self.kernel == self.kernels[0]:  # == 'Cubic Splines':
                                coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
                            elif self.kernel == self.kernels[1]:  # == 'Bernoulli Polynomials':
                                coeffs = phis[nid]  # coefficients for bernoulli
                            phi = phi * self.evaluate_basis(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.

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
                [dam, null] = np.shape(damtx)

                [beters, null, null, null, xers, ev] = gibbs(inputs_np, data, phis, X, damtx, a, b, atau, btau, draws,
                                                             phind, xsm)

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

                        [betertest, null, null, null, Xtest, evtest] = gibbs(inputs_np, data, phis, X, damtx_test, a, b,
                                                                             atau, btau, draws, phind, xsm)
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

        self.betas_avg = np.mean(betas, axis=0)

        self.betas = betas
        self.mtx = mtx
        self.evs = evs

        return betas, mtx, evs

    def clear(self, keep=None, clear=None, all=False):
        """
        Delete all attributes from the FoKL class except for hyperparameters by default, but user may specify otherwise.
        If an attribute is listed in both 'clear' and 'keep', then the attribute is cleared.

        Optional Inputs:
            keep (list of strings)  == attributes to keep in addition to hyperparameters, e.g., ['inputs_np', 'mtx']
            clear (list of strings) == hyperparameters to delete, e.g., ['kernel', 'phis']
            all (boolean)           == if True then all attributes (including hyperparameters) get deleted regardless

        Tip: To remove all attributes, simply call 'self.clear(all=1)'.
        """

        if all is not False:  # if not default
            all = str_to_bool(all)  # convert to boolean if all='on', etc.

        if all is False:
            attrs_to_keep = self.hypers  # default
            if isinstance(keep, list) or isinstance(keep, str):  # str in case of single entry, e.g., keep='mtx' != ['mtx']
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
