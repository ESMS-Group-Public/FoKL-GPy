"""
Development Information:

Description:
    - Developed by the Energy Systems and Materials Simulations of West Virginia University.
    - Example of automatic conversion from arbitrary FoKL model to Pyomo model.

Citations:
    - K. Hayes, M.W. Fouts, A. Baheri and D.S. Mebane, "Forward variable selection enables fast and accurate dynamic
        system identification with Karhunen-Loève decomposed Gaussian processes", arXiv:2205.13676.

Primary Developer(s) Contact Information:
    - Jacob P. Krell (JPK)
        - Aerospace Engineering Undergraduate Student
        - Statler College of Engineering & Mineral Resources
        - Dept. Mechanical and Aerospace Engineering
        - West Virginia University (WVU)
        - jpk0024@mix.wvu.edu

Development History:
Date              Developer        Comments
---------------   -------------    -------------------------------------------------------------------------------------
Feb. 12, 2024     JPK              'to_pyomo' modified for 'Bernoulli Polynomials' kernel and moved to FoKLRoutines;
"""
from FoKL_Bernoulli import FoKLRoutines
import pyomo.environ as pyo
import numpy as np
import copy
import warnings


def main(x, y):

    fokl = FoKLRoutines.FoKL(kernel=1)
    fokl.fit(x, y)
    fokl.coverage3(plot=1, bounds=0)

    if 1:
        m = fokl.to_pyomo(x=[None, 0.7, 0], y=178, ReturnObjective=True)
    else:
        m = fokl.to_pyomo(x=[None, 0.7, 0])
        m.obj = pyo.Objective(expr=x[0] - x[1], sense=pyo.minimize)

    solver = pyo.SolverFactory('ipopt')
    solver.options['tol'] = 1e-1  # Absolute tolerance for optimality
    solver.options['dual_inf_tol'] = 1e-1  # Dual infeasibility tolerance
    solver.options['constr_viol_tol'] = 1e-1  # Constraint violation tolerance
    solver.solve(m, tee=True)

    print(m.x[0]())

    breakpoint()

if __name__ == '__main__':
    res = int(1e4)
    t = np.linspace(0, 1, res)
    x0 = 100 * np.exp(t)
    x1 = 50 * np.sin(4 * np.pi * x0)
    x2 = 5 * np.random.rand(res) - 10
    y = x0 + x1 + x2

    main([x0, x1, x2], y)

