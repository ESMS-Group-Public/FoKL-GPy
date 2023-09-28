import numpy as np
import FoKL
from FoKL import FoKLRoutines
from FoKL import getKernels
import matplotlib.pyplot as plt


#inputs
traininputs = np.loadtxt('traininputs.csv.txt',dtype=float,delimiter=',')
traindata1 = np.loadtxt('traindata1.txt',dtype=float,delimiter=',')
traindata2 = np.loadtxt('traindata2.txt',dtype=float,delimiter=',')
y = np.loadtxt('y.txt',dtype=float,delimiter=',')
utest = np.loadtxt('utest.csv',dtype=float,delimiter=',')

# user-defined hypers to override default values
relats_in = [1,1,1,1,1,1]
a = 1000
b = 1
atau = 4
btau_model1 = 0.6091
btau_model2 = 1
draws = 2000
way3 = True
threshav = 0
threshstda = 0
threshstdb = 100

n,m = np.shape(y)
norms1 = [np.min(y[0,0:int(m/2)]),np.max(y[0,0:int(m/2)])]
norms2 = [np.min(y[1,0:int(m/2)]),np.max(y[1,0:int(m/2)])]
norms = np.transpose([norms1,norms2])
# Running emulator_Xin routine and visualization
model1 = FoKLRoutines.FoKL(relats_in=relats_in, a=a, b=b, atau=atau, btau=btau_model1, draws=draws, way3=way3, threshav=threshav, threshstda=threshstda, threshstdb=threshstdb)
model2 = FoKLRoutines.FoKL(relats_in=relats_in, a=a, b=b, atau=atau, btau=btau_model2, draws=draws, way3=way3, threshav=threshav, threshstda=threshstda, threshstdb=threshstdb)

betas1, mtx1, evs1 = model1.fit(traininputs, traindata1)
betas2, mtx2, evs2 = model2.fit(traininputs, traindata2)

betas1 = betas1[1000:]
betas2 = betas2[1000:]

start = 4
stop = 3750*4
stepsize = 4
used_inputs = [[1,1,1],[1,1,1]]
ic = y[:,int(m/2)-1]

T,Y = FoKLRoutines.FoKL.GP_Integrate([np.mean(betas1,axis=0),np.mean(betas2,axis=0)], [mtx1,mtx2], utest, norms, phis, start, stop, ic, stepsize, used_inputs)

plt.plot(T,Y[0],T,y[0][3750:7500])
plt.plot(T,Y[1],T,y[1][3750:7500])
