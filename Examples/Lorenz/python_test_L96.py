import numpy as np
from sklearn.preprocessing import normalize
from routines import splineconvert500
from emulator import emulator
from routines import coverage3
import pandas as pd
import FoKL
from FoKL import FoKLRoutines

fkl = FoKLRoutines.FoKL()

data = pd.read_csv('L96_simulation_data_30k.csv')
data = data.to_numpy()

x1 = data[:,0]
x2 = data[:,1]
x3 = data[:,2]
x4 = data[:,3]
x5 = data[:,4]

xdot1 = np.array([data[:,5]]).T
xdot2 = np.array([data[:,6]]).T
xdot3 = np.array([data[:,7]]).T
xdot4 = np.array([data[:,8]]).T
xdot5 = np.array([data[:,9]]).T

x1n = normalize([x1])
x2n = normalize([x2])
x3n = normalize([x3])
x4n = normalize([x4])
x5n = normalize([x5])

inputs1 = np.hstack((x1n.T,x2n.T,x4n.T,x5n.T))
inputs2 = np.hstack((x1n.T,x2n.T,x3n.T,x5n.T))
inputs3 = np.hstack((x1n.T,x2n.T,x3n.T,x4n.T))
inputs4 = np.hstack((x2n.T,x3n.T,x4n.T,x5n.T))
inputs5 = np.hstack((x1n.T,x3n.T,x4n.T,x5n.T))

phis = fkl.splineconvert500('spline_coefficient_500.txt')

# a
a = 4

# b
sigsqd = 0.01
b = sigsqd*(a+1)

# atau
atau = 4
var_b1 = np.var([xdot1])
var_b2 = np.var([xdot2])
var_b3 = np.var([xdot3])
var_b4 = np.var([xdot4])
var_b5 = np.var([xdot5])
# btau
btau1 = var_b1*(atau+1)/sigsqd
btau2 = var_b2*(atau+1)/sigsqd
btau3 = var_b3*(atau+1)/sigsqd
btau4 = var_b4*(atau+1)/sigsqd
btau5 = var_b5*(atau+1)/sigsqd

# tolerance
tolerance = 4
relats_in  = []
# draws
draws = 1000

gimmie = False
way3 = False
threshav = 0.05
threshstda = 0.5
threshstdb = 1
aic = False

betas1, mtx1, evs1 = fkl.emulator(inputs1, xdot1, phis, relats_in, a, b, atau, btau1, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)
betas2, mtx2, evs2 = fkl.emulator(inputs2, xdot2, phis, relats_in, a, b, atau, btau2, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)
betas3, mtx3, evs3 = fkl.emulator(inputs3, xdot3, phis, relats_in, a, b, atau, btau3, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)
betas4, mtx4, evs4 = fkl.emulator(inputs4, xdot4, phis, relats_in, a, b, atau, btau4, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)
betas5, mtx5, evs5 = fkl.emulator(inputs5, xdot5, phis, relats_in, a, b, atau, btau5, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)


meen1, bounds1, rmse1 = fkl.coverage3(betas1[500:1000,:], inputs1, xdot1, phis, mtx1, 500, 1)
meen2, bounds2, rmse2 = fkl.coverage3(betas2[500:1000,:], inputs2, xdot2, phis, mtx2, 500, 1)
meen3, bounds3, rmse3 = fkl.coverage3(betas3[500:1000,:], inputs3, xdot3, phis, mtx3, 500, 1)
meen4, bounds4, rmse4 = fkl.coverage3(betas4[500:1000,:], inputs4, xdot4, phis, mtx4, 500, 1)
meen5, bounds5, rmse5 = fkl.coverage3(betas5[500:1000,:], inputs5, xdot5, phis, mtx5, 500, 1)
rmses = [rmse1, rmse2, rmse3, rmse4, rmse5]
pd.DataFrame(rmses).to_csv('rmses.csv')