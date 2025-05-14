import warnings
import copy
import numpy as np
import pandas as pd
from ..utils import _str_to_bool, _process_kwargs, _merge_dicts, _set_attributes
class dataFormat:
    def __init__(self, fokl, config):
        self.fokl = fokl
        self.config = config


    def _format(self, inputs, data=None, AutoTranspose=True, SingleInstance=False, bit=64):
       """
       Called by 'clean' to format dataset.
           - formats inputs as 2D ndarray, where columns are input variables; n_rows > n_cols if AutoTranspose=True
           - formats data as 2D ndarray, with single column Note SingleInstance has priority over AutoTranspose. If SingleInstance=True, then AutoTranspose=False.
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
                             category=UserWarning)   # Format 'inputs' as [n x m] numpy array:
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
                             "'AutoTranspose=False' to disable.", category=UserWarning)   # Format 'data' as [n x 1] numpy array:
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

        def minmax_error():
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
                    minmax_error()
                else:  # assume [min1, max1, ..., minm, maxm] needs to be formatted to [[min1, max1], ..., [minm, maxm]]
                    minmax_vals = copy.deepcopy(minmax)
                    minmax = []
                    for i in range(0, lm, 2):
                        minmax.append([minmax_vals[i], minmax_vals[i + 1]])  # list of [min, max] lists
            elif len(minmax) != mm:
                minmax_error()
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

        inputs = np.array(inputs)

        return inputs, minmax

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
            inputs, minmax = self._normalize(inputs, current['minmax'], current['pillow'], current['pillow_type'])
        
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
            return inputs, data, minmax
        
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
            kernel = self.config.DEFAULT['kernel']
        if phis is None:
            phis = self.config.DEFAULT['phis'] 
        if kernel == self.config.KERNELS[1]:  # == 'Bernoulli Polynomials':
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
        xmin = np.array((phind - 1) * r, dtype= inputs.dtype) # inputs.dtype
        X = (inputs - xmin) / r  # twice normalized inputs (0-1 first then to size of phis second)

        phind = phind - 1
        xsm = np.array(l_phis * inputs - phind, dtype= inputs.dtype) # inputs.dtype
        if np.max(phind) > 499 or np.min(phind) < 0:
            raise ValueError('Inputs are not normalized correctly, try calling clean=True within evaluate to evaluate with normalization of model training')
        return X, phind, xsm