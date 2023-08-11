import numpy as np
import FoKL
from FoKL import getKernels
from FoKL import FoKLRoutines

#%% Setting input parameters

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

# load basis functions
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

#%% Running emulator_Xin routine and visualization, fit model to data
betas, mtx, evs = model.fit(inputs, data)

meen, bounds, rmse = model.coverage3(inputs, data, 1000, 1)

