class FoKLConfig:
    def __init__(self):
        self.DEFAULT = {
         # Hyperparameters:
         'kernel': 'Cubic Splines', 'phis': None, 'relats_in': [], 'a': 4, 'b': None, 'atau': 4,
         'btau': None, 'tolerance': 3, 'burnin': 1000, 'draws': 1000, 'gimmie': False, 'way3': False,
         'threshav': 0.05, 'threshstda': 0.5, 'threshstdb': 2, 'aic': False,

          # Hyperparameters with Update:
          'sigsqd0': 0.5, 'burn': 500, 'update': False, 'built' : False,

         # Other:
         'UserWarnings': True, 'ConsoleOutput': True
         }
    
        # Process user's keyword arguments:
        self.HYPERS = ['kernel', 'phis', 'relats_in', 'a', 'b', 'atau', 'btau', 'tolerance', 'burnin', 'draws',
                           'gimmie', 'way3', 'threshav', 'threshstda', 'threshstdb', 'aic', 'update', 'built']
        
    
        # Store list of settings for easy reference later (namely, in 'clear'):
        self.SETTINGS = ['UserWarnings', 'ConsoleOutput']

        # Store supported kernels for later logical checks against 'kernel':
        self.KERNELS = ['Cubic Splines', 'Bernoulli Polynomials']

        self.samplers = ['gibbs', 'gibbs_update']

        # Store list of attributes to keep in event of clearing model (i.e., 'self.clear'):
        self.attrs = ['dataFormat', 'functions', 'fitSampler', 'postprocessing', 'inputs', 'data', 'betas', 'minmax', 'mtx', 'evs', 'config']

        # List of attributes to keep in event of clearing model (i.e., 'self.clear'):
        self.KEEP = ['keep', 'hypers', 'settings', 'kernels'] + self.HYPERS + self.SETTINGS + self.KERNELS + self.attrs
        

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
