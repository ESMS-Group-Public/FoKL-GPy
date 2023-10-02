import numpy as np
from src.FoKL import FoKLRoutines # from FoKL import FoKLRoutines

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

# define model using only user-overridden hypers as inputs
model = FoKLRoutines.FoKL(a=9, b=0.01, atau=3, btau=4000, aic=True)

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

meen, bounds, rmse = model.coverage3(inputs, data, model.draws, 1)
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
