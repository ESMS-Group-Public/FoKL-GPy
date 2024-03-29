"""
[EXAMPLE]: Pyomo Maximize

This is an example of FoKL converting its GP model to a symbolic Pyomo expression and solving the generated Pyomo model
algebraically for the maximum value of the GP model. The dataset used is arbitrarily defined and is a toy problem.
"""
from FoKL import FoKLRoutines
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
        f = FoKLRoutines.load('pyomo_maximize.fokl')
    except Exception as exception:
        f = FoKLRoutines.FoKL(kernel=1, UserWarnings=False)
        print("\nTraining FoKL model...")
        f.fit([x0, x1], y, clean=True)
        print("Done!")
        f.save('pyomo_maximize.fokl')

    # Plot to visualize dataset and model:
    f.coverage3(plot=True, xlabel='t', xaxis=t, ylabel='y')

    # Convert FoKL's GP model to Pyomo:
    scenarios = f.draws  # Pyomo 'scenarios' is synonym for FoKL 'draws'
    m = f.to_pyomo(draws=scenarios)  # default is 'draws=self.draws' so this is unnecessary but demonstrates idea

    # Add constraints (if any/known) to enforce 'physics':
    m.t = pyo.Var(within=pyo.Reals, bounds=[0, 1])
    m.x0 = pyo.Constraint(expr=m.fokl_x[0] == (sin(m.t * np.pi * 3) + 1) / 2)
    m.x1 = pyo.Constraint(expr=m.fokl_x[1] == (cos(m.t * np.pi * 3) + 1) / 2)

    m.fokl_y_avg = pyo.Expression(expr=sum(m.fokl_y[i] for i in m.fokl_scenarios) / scenarios)  # average of scenarios
    m.obj = pyo.Objective(expr=m.fokl_y_avg, sense=pyo.maximize)

    # Solve (need multistart 'https://pyomo.readthedocs.io/en/latest/contributed_packages/multistart.html'):
    solver = pyo.SolverFactory('multistart')  # multistart to try local solver at various points for global solution
    print("\nRunning Pyomo solver...")
    solver.solve(m, solver='ipopt', suppress_unbounded_warning=True)  # IPOPT is local solver, and warning not needed
    print("Done!")

    print("\nPyomo solution:")
    print(f"     y = {m.obj()}")
    x = []
    for j in m.fokl_j:
        print(f"    x{j} = {m.fokl_x[j]()}")
        x.append(m.fokl_x[j]())
    print(f"     t = {m.t()}")

    # Check solution against FoKL model:
    x = np.array(x)[np.newaxis, :]
    y_check = f.evaluate(x, draws=scenarios)
    print("\nNumerical FoKL evaluation of Pyomo solution [x0, x1] to confirm equivalence of models:")
    print(f"     y = {y_check[0]}")


if __name__ == '__main__':
    print("\nStart of Pyomo Maximize example.")
    main()
    print("\nEnd of Pyomo Maximize example.")

