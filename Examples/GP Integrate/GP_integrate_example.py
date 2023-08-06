import csv

import fkl as fkl
import numpy as np
import FoKL
from FoKL import FoKLRoutines
import matplotlib.pyplot as plt

fkl = FoKLRoutines.FoKL()

#inputs
traininputs = np.loadtxt('traininputs.csv.txt',dtype=float,delimiter=',')
traindata1 = np.loadtxt('traindata1.txt',dtype=float,delimiter=',')
traindata2 = np.loadtxt('traindata2.txt',dtype=float,delimiter=',')
y = np.loadtxt('y.txt',dtype=float,delimiter=',')
utest = np.loadtxt('utest.csv',dtype=float,delimiter=',')
# data
relats_in = [1,1,1,1,1,1]

# generating phis from the spline coefficients text file in emulator_python
# combined with the splineconvert script

phis = getKernels()

# a
a = 1000

# b
b = 1

# atau
atau = 4

# btau
btau = 0.6091

# tolerance
tolerance = 3
relats_in  = [1,1,1,1,1,1]
# draws
draws = 2000

gimmie = False
way3 = True
threshav = 0
threshstda = 0
threshstdb = 100
aic = False
n,m = np.shape(y)
norms1 = [np.min(y[0,0:int(m/2)]),np.max(y[0,0:int(m/2)])]
norms2 = [np.min(y[1,0:int(m/2)]),np.max(y[1,0:int(m/2)])]
norms = np.transpose([norms1,norms2])
# Running emulator_Xin routine and visualization
betas1, mtx1, evs1 = fkl.emulator(traininputs, traindata1, phis, relats_in, a, b, atau, btau, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)
betas2, mtx2, evs2 = fkl.emulator(traininputs, traindata2, phis, relats_in, a, b, atau, 1, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)

betas1 = betas1[1000:]
betas2 = betas2[1000:]

start = 4
stop = 3750*4
stepsize = 4
used_inputs = [[1,1,1],[1,1,1]]
ic = y[:,int(m/2)-1]

T,Y = fkl.GP_Integrate([np.mean(betas1,axis=0),np.mean(betas2,axis=0)], [mtx1,mtx2], utest, norms, phis, start, stop, ic, stepsize, used_inputs)

plt.plot(T,Y[1],T,y[3750:7500])