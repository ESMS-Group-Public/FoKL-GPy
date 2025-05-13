import warnings
import matplotlib.pyplot as plt
import numpy as np
from ..utils import _process_kwargs, _str_to_bool, _merge_dicts

class postprocess:
    def __init__(self, fokl, config, dataFormat, functions):
        self.fokl = fokl
        self.config = config
        self.dataFormat = dataFormat
        self.functions = functions
        self.setnos = None

    def evaluate(self, inputs=None, betas=None, mtx=None, draws=None, **kwargs):
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
        # Check if self.minmax exists before including it in defaults
        if not hasattr(self.fokl, 'minmax'):
            raise ValueError("To set minmax manually call model.minmax = ([input_1_min, input_1_max],[input_2_min, input_2_max],...)"
                " or set clean=True to automtically define min and max from model.inputs")
        # Process keywords:
        default = {'minmax': None, 'draws': self.config.DEFAULT['draws'], 'clean': False, 'ReturnBounds': False,  # for evaluate
                   '_suppress_normalization_warning': False}                                    # if called from coverage3
        default_for_clean = {'train': 1, 
                             # For '_format':
                             'AutoTranspose': True, 'SingleInstance': False, 'bit': 64,
                             # For '_normalize':
                             'normalize': True, 'minmax': self.fokl.minmax, 'pillow': None, 'pillow_type': 'percent'}
        current = _process_kwargs(_merge_dicts(default, default_for_clean), kwargs)
        for boolean in ['clean', 'ReturnBounds']:
            current[boolean] = _str_to_bool(current[boolean])
        kwargs_to_clean = {}
        for kwarg in default_for_clean.keys():
            kwargs_to_clean.update({kwarg: current[kwarg]})  # store kwarg for clean here
            del current[kwarg]  # delete kwarg for clean from current
        if current['draws'] < 40 and current['ReturnBounds']:
            warnings.warn("'draws' must be greater than or equal to 40 if calculating bounds. Setting 'draws=40'.")
        if betas is None:  # default
            betas = self.betas 
        if draws is None:  # default 
            if betas.shape[0] < draws: 
                draws = len(betas.shape[0])  
            elif betas.shape[0] > draws: 
                draws = current['draws']
        else: 
            if betas.shape[0] < draws:
                raise ValueError("The number of draws requested exceeds the number of beta terms in the model.")
        if mtx is None:  # default
            mtx = self.fokl.mtx
        else:  # user-defined mtx may need to be formatted
            if isinstance(mtx, int):
                mtx = [mtx]
            mtx = np.array(mtx)
            if mtx.ndim == 1:
                mtx = mtx[np.newaxis, :]
                warnings.warn("Assuming 'mtx' represents a single model. If meant to represent several models, then "
                              "explicitly enter a 2D numpy array where rows correspond to models.")

        phis = self.config.DEFAULT['phis']

        # Automatically normalize and format inputs:
        if inputs is None:  # default
            inputs = self.fokl.inputs
            if current['clean']:
                warnings.warn("Cleaning was already performed on default 'inputs', so overriding 'clean' to False.",
                              category=UserWarning)
                current['clean'] = False
        else:  # user-defined 'inputs'
            if not current['clean']:  # assume provided inputs are already formatted and normalized
                normputs = inputs
                # if current['_suppress_normalization_warning'] is False:  # to suppress warning when evaluate called from coverage3
                #     warnings.warn("User-provided 'inputs' but 'clean=False'. Subsequent errors may be solved by enabling automatic formatting and normalization of 'inputs' via 'clean=True'.", category=UserWarning)
        if current['clean']:
            normputs = self.dataFormat.clean(inputs, kwargs_from_other=kwargs_to_clean)
        elif inputs is None:
            normputs = self.inputs
        else:
            normputs = np.array(inputs)

        m, mbets = np.shape(betas)  # Size of betas
        n = np.shape(normputs)[0] # Size of normalized inputs
        mputs = int(np.size(normputs)/n)

        if self.setnos is None:
            setnos = np.random.choice(m,draws,replace=False) # random draw selection
            self.setnos = setnos
        else:
            setnos = self.setnos

        if draws == 1: 
            setnos = [0]

        X = np.zeros((n, mbets))
        normputs = np.asarray(normputs)

        if self.config.KERNELS[0] == self.config.DEFAULT['kernel']:  # == 'Cubic Splines':
            _, phind, xsm = self.dataFormat._inputs_to_phind(normputs)  # ..., phis=self.phis, kernel=self.kernel) already true
        elif self.config.KERNELS[1] == self.config.DEFAULT['kernel']:  # == 'Bernoulli Polynomials':
            phind = None
            xsm = normputs

        for i in range(n):
            for j in range(1, mbets):
                phi = 1
                for k in range(mputs):
                    num = mtx[j - 1, k]
                    if num > 0:
                        nid = int(num - 1)
                        if self.config.KERNELS[0] == self.config.DEFAULT['kernel']:  # == 'Cubic Splines':
                            coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
                        elif self.config.KERNELS[1] == self.config.DEFAULT['kernel']:  # == 'Bernoulli Polynomials':
                            coeffs = phis[nid]  # coefficients for bernoulli
                        phi *= self.functions.evaluate_basis(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.

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
            'inputs': None, 'data': None, 'draws': self.config.DEFAULT['draws'], 'betas': self.fokl.betas,

            # For basic plot controls:
            'plot': False, 'bounds': True, 'xaxis': False, 'labels': True, 'xlabel': 'Index', 'ylabel': 'Data',
            'title': 'FoKL', 'legend': True, 'LegendLabelFoKL': 'FoKL', 'LegendLabelData': 'Data',
            'LegendLabelBounds': 'Bounds', 'ReturnBounds': True,

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
            current['inputs'] = self.fokl.inputs
        if current['data'] is None:
            current['data'] = self.fokl.data

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

        if current['ReturnBounds'] == True:
            mean, bounds = self.evaluate(normputs, betas=current['betas'], draws=draws, ReturnBounds=1, _suppress_normalization_warning=True)
        else: 
            mean = self.evaluate(normputs, betas=current['betas'], draws=draws, ReturnBounds=0, _suppress_normalization_warning=True)

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
                if current['ReturnBounds'] == True:
                    plt_mean = mean
                    plt_data = data
                    plt_bounds = bounds
                else: 
                    plt_mean = mean
                    plt_data = data

            if current['ReturnBounds'] == True:
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
            else: 
                plt.figure()
                plt.plot(plt_x, plt_mean, current['PlotTypeFoKL'], linewidth=current['PlotSizeFoKL'],
                         label=current['LegendLabelFoKL'])
                if data is not False:
                    plt.plot(plt_x, plt_data, current['PlotTypeData'], markersize=current['PlotSizeData'],
                             label=current['LegendLabelData'])
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

        if current['ReturnBounds'] == True:
            return mean, bounds, rmse
        else:
            return mean, rmse