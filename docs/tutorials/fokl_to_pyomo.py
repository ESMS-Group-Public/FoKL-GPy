"""
[TUTORIAL]: self.to_pyomo(self, m=None, y=None, x=None, ReturnObjective=False)

This is a tutorial for the 'to_pyomo' method, an auxiliary tool for automatically converting a FoKL model to a Pyomo
expression. In the following script, it will be shown how 'to_pyomo' may be used to generate a new Pyomo model or used
to append to an already existing Pyomo model. Variations will show how the FoKL model may be set as a Pyomo objective in
order to minimize the model's error, or set as a Pyomo constraint in order to focus on a different relationship.

In this tutorial, the following will be demonstrated:

    1) Train a FoKL model using the 'Bernoulli Polynomials' kernel.

    2) Convert the FoKL model to a constraint of a new Pyomo model with...

        a) all unknown inputs and data.
        b) some known inputs and data.

    3) Convert the FoKL model to the objective of a new Pyomo model with...

        a) all unknown inputs and data.
        b) some known inputs and data.

    4) Demonstrate how a pre-existing Pyomo model could have been for (2) and (3), with only (2b) for example.

    5) Define objective for the Pyomo models of (2).

    6) Define constraint(s) for the Pyomo models of (3).

    7) Solve the Pyomo models.

    8) Retrieve the solution.
"""
from FoKL import FoKLRoutines
import pyomo.environ as pyo
import numpy as np


def main():
    print("\nThe following is a FoKL Tutorial for the 'to_pyomo' method, assuming both Pyomo and IPOPT are installed.")

    # Known dataset:

    res = int(1e4)
    t = np.linspace(0, 1, res)
    x0 = 100 * np.exp(t)                                                    # first input
    x1 = 50 * np.sin(4 * np.pi * x0)                                        # second input
    x2 = 5 * np.random.rand(res) - 10                                       # third input
    y = x0 + x1 + x2                                                        # data

    # (1) Train a FoKL model using the 'Bernoulli Polynomials' kernel.

    print("\nCurrently training...\n")

    f = FoKLRoutines.FoKL(kernel=1)
    f.fit([x0, x1, x2], y, clean=True)

    # (2) Convert the FoKL model to a constraint of a new Pyomo model with...

    print("\nCurrently converting to Pyomo...")

    m_2a = f.to_pyomo()                                                     # (a) all unknown inputs and data.
    m_2b = f.to_pyomo(x=[None, 0.7, None], y=213)                           # (b) some known inputs and data.

    # (3) Convert the FoKL model to an objective of a new Pyomo model with...

    m_3a = f.to_pyomo(ReturnObjective=True)                                 # (a) all unknown inputs and data.
    m_3b = f.to_pyomo(x=[None, 0.7, None], y=213, ReturnObjective=True)     # (b) some known inputs and data.

    # (4) Demonstrate how a pre-existing Pyomo model could have been used for (2) and (3), with only (2b) for example.

    m_2b_preexisting = pyo.ConcreteModel()
    f.to_pyomo(x=[None, 0.7, None], y=213, m=m_2b_preexisting)

    # Note how the Pyomo model object 'm_2b_preexisting' still gets updated despite not being an output to 'f.to_pyomo'.

    # (5) Define objective for the Pyomo models of (2).

    m_2a.obj = pyo.Objective(expr=abs(m_2a.x[0] - m_2a.x[2]), sense=pyo.minimize)
    m_2b.obj = pyo.Objective(expr=abs(m_2b.x[0] - m_2b.x[2]), sense=pyo.minimize)

    # Note setting 'ReturnObjective=True' in (2) would be equivalent to defining here in (5) the following objective:
    #     - m_2b.obj = pyo.Objective(expr=abs(m_2b.fokl - m_2b.y), sense=pyo.minimize)

    # (6) Define constraint(s) for the Pyomo models of (3).

    m_3a.con1 = pyo.Constraint(expr=abs(m_3a.x[0] - m_3a.x[2]) == 0)
    m_3b.con1 = pyo.Constraint(expr=abs(m_3b.x[0] - m_3b.x[2]) == 0)

    # (7) Solve the Pyomo models, using (2b) for example.

    print("\nCurrently solving...\n")

    solver = pyo.SolverFactory('ipopt')
    # solver.options['tol'] = 1e-1

    # solver.solve(m_2a, tee=True)
    solver.solve(m_2b, tee=True)
    # solver.solve(m_3a, tee=True)
    # solver.solve(m_3b, tee=True)

    # (8) Retrieve the solution, using (2b) for example.

    print("\nCurrently retrieving solution...\n")

    solution = []
    for i in range(3):  # for input variable index in number of input variables
        solution.append(m_2b.x[i]())  # == solution.append(pyo.value(m_3b.x[i]))
    print("x = ", solution)  # [0.5825, 0.7, 0.5825] is expected

    # (9) Print the symbolic FoKL model, using (2b) for example.

    print("\ny = ", m_2b.fokl.expr)


if __name__ == '__main__':
    main()
    print("\nEnd of FoKL to Pyomo tutorial.")

