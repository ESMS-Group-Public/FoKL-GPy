"""
[TUTORIAL]: self.to_pyomo(self, m=None, y=None, x=None, **kwargs)

This is a tutorial for the 'to_pyomo' method, an auxiliary tool for automatically converting a FoKL model to a Pyomo
expression. In the following script, it will be shown how 'to_pyomo' may be used to generate a new Pyomo model or used
to append to an already existing Pyomo model.

In this tutorial, the following will be demonstrated:

    1) Train a FoKL model using the 'Bernoulli Polynomials' kernel.

    2) Convert the FoKL model to a new Pyomo model with...

        a) all unknown inputs and data.
        b) some known inputs and data.

    3) Demonstrate how a pre-existing Pyomo model could have been used, with (2b) for example.
"""
from FoKL import FoKLRoutines
import pyomo.environ as pyo
import numpy as np


def main():
    print("\nThe following is a FoKL Tutorial for the 'to_pyomo' method, assuming both Pyomo and IPOPT are installed.")

    # Known dataset:

    res = int(1e4)
    t = np.linspace(0, 1, res)
    x0 = 100 * np.exp(t)                # first input
    x1 = 50 * np.sin(4 * np.pi * x0)    # second input
    x2 = 5 * np.random.rand(res) - 10   # third input
    y = x0 + x1 + x2                    # data

    # (1) Train a FoKL model using the 'Bernoulli Polynomials' kernel.

    print("\nCurrently training...")

    model = FoKLRoutines.FoKL(kernel=1)
    model.fit([x0, x1, x2], y, clean=True)

    print("Done!")

    # (2) Convert the FoKL model to a new Pyomo model with...

    print("\nCurrently converting to Pyomo...")

    m_2a = model.to_pyomo()                             # (a) all unknown inputs and data.
    m_2b = model.to_pyomo(x=[None, 0.7, None], y=213)   # (b) some known inputs and data.

    print("Done!")

    # (3) Demonstrate how a pre-existing Pyomo model could have been used, with (2b) for example.

    m_2b_preexisting = pyo.ConcreteModel()
    model.to_pyomo(x=[None, 0.7, None], y=213, m=m_2b_preexisting)

    # Note how the Pyomo model 'm_2b_preexisting' still gets updated without 'm_2b_preexisting = model.to_pyomo(...)'.
    # Either is fine and functions the same.


if __name__ == '__main__':
    main()
    print("\nEnd of FoKL to Pyomo tutorial.")

