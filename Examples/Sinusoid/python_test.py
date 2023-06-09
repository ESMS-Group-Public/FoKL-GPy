
import numpy as np
import FoKL
from FoKL import FoKLRoutines


fkl = FoKLRoutines.FoKL()


#%% Setting input parameters
# sigsqd0
sigsqd0 = 0.009

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

# generating phis from the spline coefficients text file in emulator_python
# combined with the splineconvert script

phis = fkl.splineconvert500('spline_coefficient_500.txt')

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
relats_in  = []
# draws
draws = 1000

gimmie = False
way3 = False
threshav = 0.05
threshstda = 0.5
threshstdb = 2
aic = True


#%% Running emulator_Xin routine and visualization
betas, mtx, evs = fkl.emulator(inputs, data, phis, relats_in, a, b, atau, btau, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)

meen, bounds, rmse = fkl.coverage3(betas, inputs, data, phis, mtx, 1000,1)

