"""
[EXAMPLE]: Pyomo Maximize

This is an example of FoKL converting its GP model to a symbolic Pyomo expression and solving the generated Pyomo model
algebraically for the maximum value of the GP model. The dataset used is arbitrarily defined and is a toy problem.
"""
from FoKL import FoKLRoutines
import os
dir = os.path.abspath(os.path.dirname(__file__))  # directory of script
# # -----------------------------------------------------------------------
# # UNCOMMENT IF USING LOCAL FOKL PACKAGE:
# import sys
# sys.path.append(os.path.join(dir, '..', '..'))  # package directory
# from src.FoKL import FoKLRoutines
# # -----------------------------------------------------------------------
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *


def dataset():
    res = int(1e2)
    t = np.linspace(0, 1, res)
    window = np.sin(t * np.pi)
    noise = np.random.rand(res) * 100

    x0 = np.sin(t * np.pi * 3)
    x1 = np.cos(t * np.pi * 3)

    def b1(x):
        return -11.2291203558315 + 22.4582407116631 * x

    def b2(x):
        return 3.93718725725502 - 23.6231235435301 * x + 23.6231235435301 * x ** 2

    def b3(x):
        return -0.960140559513144 + 11.5216867141577 * x - 28.8042167853943 * x ** 2 + 19.2028111902629 * x ** 3

    y = -window * (b1(x0) + b2(x1) + b2(x0) * b3(x1)) + noise

    return t, x0, x1, y


def main():
    # Define/Load dataset:
    t, x0, x1, y = dataset()  # t is parameter for x used for plot's x-axis, and dataset is <y|x>

    # Train/Load FoKL model:
    try:
        f = FoKLRoutines.load(os.path.join(dir, 'pyomo_maximize.fokl'))
    except Exception as exception:
        f = FoKLRoutines.FoKL(kernel=1, UserWarnings=False)
        print("\nTraining FoKL model...")
        f.fit([x0, x1], y, clean=True)
        print("Done!")
        f.save(os.path.join(dir, 'pyomo_maximize.fokl'))

    # Plot to visualize dataset and model:
    f.coverage3(plot=True, xlabel='t', xaxis=t, ylabel='y')

    # Convert FoKL's GP model to Pyomo:
    scenarios = f.draws  # Pyomo 'scenarios' is synonym for FoKL 'draws'
    xvars = ['x0', 'x1']
    m = f.to_pyomo(xvars, 'y', draws=scenarios)  # default is 'draws=self.draws' so this is unnecessary but demonstrates idea

    # Add known constraints (if any) to enforce 'physics':
    m.t = pyo.Var(within=pyo.Reals, bounds=[0, 1])
    m.known = ConstraintList()
    m.known.add(m.x0 == sin(m.t * np.pi * 3))
    m.known.add(m.x1 == cos(m.t * np.pi * 3))

    # Set objective:
    m.obj = pyo.Objective(expr=m.y, sense=pyo.maximize)

    # Solve (need multistart 'https://pyomo.readthedocs.io/en/latest/contributed_packages/multistart.html'):
    solver = pyo.SolverFactory('multistart')  # multistart to try local solver at various points for global solution
    print("\nRunning Pyomo solver...")
    solver.solve(m, solver='ipopt', suppress_unbounded_warning=True)  # IPOPT is local solver, and warning not needed
    print("Done!")

    print("\nPyomo solution:")
    print(f"     y = {m.obj()}")
    x = []
    x_norm = []
    for j in m.GP0_j:  # = range(f.inputs.shape[1]) = range(len(xvars))
        print(f"    x{j} = {m.component(xvars[j])()}")
        x.append(m.component(xvars[j])())
        x_norm.append(m.component(f"GP0_{xvars[j]}_norm")())
    print(f"     t = {m.t()}")

    # Check solution against FoKL model:

    print("\nNumerical FoKL evaluation of Pyomo solution [x0, x1] to confirm equivalence of models from...")

    x = f.clean(x, SingleInstance=True)  # automatically format and normalize the true scale solution
    y_check = f.evaluate(x, draws=scenarios)
    print(f"    - true scale:    y = {y_check[0]}")

    # Checking accuracy of solution using normalized values (to avoid numerical errors):
    x_norm = f.clean(x_norm, SingleInstance=True, normalize=False)  # automatically format the normalized solution
    y_check_norm = f.evaluate(x_norm, draws=scenarios)
    print(f"    - normalized:    y = {y_check_norm[0]}")


if __name__ == '__main__':
    print("\nStart of Pyomo Maximize example.")
    main()
    print("\nEnd of Pyomo Maximize example.")

