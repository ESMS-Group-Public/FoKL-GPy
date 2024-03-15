from src.FoKL import FoKLRoutines
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import *


def main(scenarios):
    res = int(1e2)
    t = np.linspace(0, 1, res)
    window = np.sin(t * np.pi)

    x0 = np.sin(t * np.pi * 3)
    x1 = np.cos(t * np.pi * 3)

    def b1(x):
        return -11.2291203558315 + 22.4582407116631 * x

    def b2(x):
        return 3.93718725725502 - 23.6231235435301 * x + 23.6231235435301 * x ** 2

    def b3(x):
        return -0.960140559513144 + 11.5216867141577 * x - 28.8042167853943 * x ** 2 + 19.2028111902629 * x ** 3

    y = -window * (b1(x0) + b2(x1) + b2(x0) * b3(x1))

    try:
        f = FoKLRoutines.load('pyomo_example.fokl')
    except Exception as e:
        f = FoKLRoutines.FoKL(kernel=1, draws=2000)
        f.fit([x0, x1], y, clean=True)
        f.save('pyomo_example.fokl')

    f.coverage3(draws=scenarios, plot=True, xlabel='t', xaxis=t, ylabel='y')

    m = f.to_pyomo(scenarios)

    # additional constraints to enforce 'physics':
    m.t = pyo.Var(within=pyo.Reals, bounds=[0, 1])
    m.x0 = pyo.Constraint(expr=m.fokl_x[0] == (sin(m.t * np.pi * 3) + 1) / 2)
    m.x1 = pyo.Constraint(expr=m.fokl_x[1] == (cos(m.t * np.pi * 3) + 1) / 2)

    m.obj = pyo.Objective(expr=m.fokl_y_avg, sense=pyo.maximize)

    # solve:
    solver = pyo.SolverFactory('ipopt')
    solver.solve(m, tee=True)

    print(f"\ny_avg = {m.obj()}")
    x = []
    for j in m.fokl_j:
        print(f"x{j} = {m.fokl_x[j]()}")
        x.append(m.fokl_x[j]())
    print(f"t = {m.t()}")

    # checking solution against FoKL model:
    x = np.array(x)[np.newaxis, :]
    y_check = f.evaluate(x, draws=scenarios)

    breakpoint()


if __name__ == '__main__':
    scenarios = 1000
    main(scenarios)

