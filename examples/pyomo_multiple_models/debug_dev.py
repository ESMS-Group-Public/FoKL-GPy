"""
dev/debug of fokl_to_pyomo
"""
# -----------------------------------------------------------------------
# Local version of 'from FoKL import ...':
import os
import sys
dir = os.path.abspath(os.path.dirname(__file__))  # directory of script
sys.path.append(dir)
sys.path.append(os.path.join(dir, '..', '..'))  # package directory
from src.FoKL import FoKLRoutines
from src.FoKL.fokl_to_pyomo import fokl_to_pyomo
# -----------------------------------------------------------------------
import numpy as np
import pyomo.environ as pyo


# a = np.linspace(0, 1, 10)
a = -(np.linspace(-1, 1, 50) ** 2)
b = np.random.rand(50) * 0.01
c0 = a + b + 4392.1
c1 = a * b

gp0 = FoKLRoutines.FoKL(kernel=1)
gp1 = FoKLRoutines.FoKL(kernel=1)

gp0.fit([a, b], c0, clean=True, pillow=0.1)
gp1.fit([a, b], c1, clean=True, pillow=0.05)

gp0.coverage3(plot=True)

# -----------------------------------------------------------------------------------------------

m = fokl_to_pyomo([gp0, gp1], [['a', 'b']] * 2, ['c0', 'c1'], 3)
# m = fokl_to_pyomo([gp0, gp1], [['a0', 'b0'], ['a1', 'b1']], ['c0', 'c1'], 3)

m.obj = pyo.Objective(expr=m.c0, sense=pyo.maximize)

m.pprint()

opt = pyo.SolverFactory('ipopt')
opt.solve(m, tee=True)


b=1

