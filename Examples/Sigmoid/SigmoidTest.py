import numpy as np
import FoKL
from FoKL import FoKLRoutines
from FoKL import getKernels

# Setting input parameters

#inputs
X = np.loadtxt('X.csv',dtype=float,delimiter=',')
Y = np.loadtxt('Y.csv',dtype=float,delimiter=',')

m, n = np.shape(X)  # reading dimensions of input variables
X_reshape = np.reshape(X, (m*n,1), order='F')  # reshaping, fortran index order

m, n = np.shape(Y)
Y_reshape = np.reshape(Y, (m*n,1), order='F')

inputs = []

for i in range(len(X_reshape)):
    inputs.append([float(X_reshape[i]), float(Y_reshape[i])])

# data
data_load = np.loadtxt('DATA_nois.csv',dtype=float,delimiter=',')

data = np.reshape(data_load, (m*n,1), order='F')

phis = getKernels.sp500()

# a
a = 9

# b
b = 0.01

# atau
atau = 3

# btau
btau = 4000

# tolerance
tolerance = 3
relats_in = []
# draws
draws = 1000

gimmie = False
way3 = False
threshav = 0.05
threshstda = 0.5
threshstdb = 2
aic = True

# define model
model = FoKLRoutines.FoKL(phis, relats_in, a, b, atau, btau, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)
"""
        initialization inputs:

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
        
         'threshav' is a threshold for proposing terms for elimination based on
         their mean values (larger thresholds lead to more elimination)
        
         'threshstda' is a threshold standard deviation -- expressed as a fraction 
         relative to the mean -- that pairs with 'threshav'.
         terms with coefficients that are lower than 'threshav' and higher than
         'threshstda' will be proposed for elimination (elimination will happen or not 
         based on relative BIC values)
        
         'threshstdb' is a threshold standard deviation that is independent of the
         mean value of the coefficient -- all with a standard deviation (fraction 
         relative to mean) exceeding
         this value will be proposed for elimination
         
        'gimmie' is a boolean causing the routine to return the most complex
        model tried instead of the model with the optimum bic

        'aic' is a boolean specifying the use of the aikaike information
        criterion 
    """
# Running emulator routine and visualization, fit model to data
betas, mtx, evs = model.fit(inputs, data)
"""
    inputs: 
        'inputs' - normalzied inputs
        
        'data' - results
    
    outputs:
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
"""

meen, bounds, rmse = model.coverage3(inputs, data, draws, 1)
"""
    inputs:
        'inputs' - normalized test inputs
        'data' - test set results
        'draws' - number of beta models used from model.fit()
        plotting binary - 1 = show plot, 0 = do not show plot
        
    outputs:
        'meen' - indexed predicted value
        'bounds' - 95% confidence interval for each point
        'rmse' - root mean square error
"""
