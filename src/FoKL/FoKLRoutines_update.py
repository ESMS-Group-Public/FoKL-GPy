import warnings
import numpy as np
from .utils import _str_to_bool, _process_kwargs, load
from .config import FoKLConfig
from .fokl_to_pyomo import fokl_to_pyomo
from .preprocessing.kernels import getKernels
from .preprocessing.dataFormat import dataFormat
from .samplers.fit import fitSampler
from .postprocessing.postprocessing import postprocess
from .FoKL_Function.functions import functions



class FoKL:
    def __init__(self, **kwargs):
        self.config = FoKLConfig()
        self.dataFormat = dataFormat(self, self.config)
        self.functions = functions(self, self.config, self.dataFormat)
        self.fitSampler = fitSampler(self, self.config, self.dataFormat, self.functions)
        self.postprocessing = postprocess(self, self.config, self.dataFormat, self.functions)

        current = _process_kwargs(self.config.DEFAULT, kwargs) # = default, but updated by any user kwargs
        for boolean in ['gimmie', 'way3', 'aic', 'UserWarnings', 'ConsoleOutput']:
            if not (current[boolean] is False or current[boolean] is True): 
                current[boolean] = _str_to_bool(current[boolean])

        # Load spline coefficients:
        phis = current['phis']  # in case advanced user is testing other splines
        if isinstance(current['kernel'], int):  # then assume integer indexing 'self.kernels'
            current['kernel'] = self.config.KERNELS[current['kernel']]  # update integer to string
        if current['phis'] is None: # if default
            if current['kernel'] == self.config.KERNELS[0]: # == 'Cubic Splines':
                current['phis'] = getKernels.sp500()
            elif current['kernel'] == self.config.KERNELS[1]:   # == 'Bernoulli Polynomials':
                current['phis'] = getKernels.bernoulli()
            elif isinstance(current['kernel'], str):    # confirm string before printing to console
                raise ValueError(f"The user-provided kernel '{current['phis']}' is not supported.")
            else:
                raise ValueError(f"The user-provided kernel is not supported.")
            
        if current['UserWarnings']:
            warnings.filterwarnings("default", category=UserWarning)
        else: 
            warnings.filterwarnings("ignore", category=UserWarning)

        for key, value, in current.items():
            setattr(self, key, value)
                
    def evaluate(self, inputs=None, betas=None, mtx=None, draws = None, **kwargs):
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
        return self.postprocessing.evaluate(inputs, betas, mtx, draws, **kwargs)
    
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
        return self.postprocessing.coverage3(**kwargs)
    
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
        self.inputs, self.data, self.betas, self.minmax, self.mtx, evs = self.fitSampler.fit(inputs, data, **kwargs)
        return self.betas, self.mtx, self.minmax, evs
    
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
            attrs_to_keep = self.config.KEEP  # default
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
    
    def save(self, filename=None, **kwargs):
        """
        Save a FoKL class as a file. By default, the 'filename' is 'model_yyyymmddhhmmss.fokl' and is saved to the
        directory of the Python script calling this method. Use 'directory' to change the directory saved to, or simply
        embed the directory manually within 'filename'.

        Returned is the 'filepath'. Enter this as the argument of 'load' to later reload the model. Explicitly, that is
        'FoKLRoutines.load(filepath)' or 'FoKLRoutines.load(filename, directory)'.

        Note the directory must exist prior to calling this method.
        """
        return self.functions.save(filename, **kwargs)
    
    def load(self, filename=None, **kwargs):
        """
        Load a FoKL class from a file.
    
        By default, 'directory' is the current working directory that contains the script calling this method. An absolute
        or relative directory may be defined if the model to load is located elsewhere.
    
        For simplicity, enter the returned output from 'self.save()' as the argument here, i.e., for 'filename'. Do this
        while leaving 'directory' blank since 'filename' can simply include the directory itself.
        """
        return load(filename, **kwargs)