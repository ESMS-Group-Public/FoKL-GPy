import FoKL
import pandas as pd
import numpy as np
import random
from FoKL import FoKLRoutines
from FoKL import getKernels

data_load = pd.read_csv("testdatatest.csv", dtype=float)
inputs = data_load[['x','y']]
data = data_load['data']

# Making data for when parameters are defaulted

random.seed(102823, version = 1)
np.random.seed(102823)

model = FoKLRoutines.FoKL()
   
betas, mtx, evs = model.fit(inputs, data, clean = True) 

meen, bounds, rmse = model.coverage3()

np.savetxt("betas.csv", betas)
np.savetxt("mtx.csv", mtx)
np.savetxt("evs.csv", evs)
np.savetxt("meen.csv", meen)

# Making data for when parameters are changed 

random.seed(102923, version = 1)
np.random.seed(102923)

model = FoKLRoutines.FoKL()
   
betas, mtx, evs = model.fit(inputs, data, aic=True, a=3, b=1.8, atau=17, btau=2100.5, tolerance=3, clean = True)

meen, bounds, rmse = model.coverage3()

np.savetxt("betasChange.csv", betas)
np.savetxt("mtxChange.csv", mtx)
np.savetxt("evsChange.csv", evs)
np.savetxt("meenChange.csv", meen)