"""
[EXAMPLE]: Pyomo Multiple Models

This is an example of embedding multiple FoKL GP models in a single Pyomo model, where one is an input of another.

The following will be modeled:
     Cp = f(T)
      G = f(T, Cp)

Then, T will be found to maximize abs(G).
"""
from FoKL import FoKLRoutines
from FoKL.fokl_to_pyomo import fokl_to_pyomo
import os
dir = os.path.abspath(os.path.dirname(__file__))  # directory of script
# # -----------------------------------------------------------------------
# # UNCOMMENT IF USING LOCAL FOKL PACKAGE:
# import sys
# sys.path.append(os.path.join(dir, '..', '..'))  # package directory
# from src.FoKL import FoKLRoutines
# from src.FoKL.fokl_to_pyomo import fokl_to_pyomo
# # -----------------------------------------------------------------------
from thermochem.janaf import Janafdb
import pyomo.environ as pyo
import numpy as np
import pandas as pd


def main():
    """Train FoKL models, convert to Pyomo, and solve NLP."""
    try:
        model_Cp = FoKLRoutines.load(os.path.join(dir, 'model_Cp.fokl'))
        model_G = FoKLRoutines.load(os.path.join(dir, 'model_G.fokl'))

    except Exception as exception:
        # Load dataset:
        try:
            CO2 = Janafdb().getphasedata(filename='C-095').rawdata  # https://janaf.nist.gov/tables/C-095.html
        except Exception as exception:
            CO2 = pd.read_csv(os.path.join(dir, 'C-095.txt'), delimiter="\t", header=1)
            CO2.values[1::, 5:7] = CO2.values[1::, 5:7] * 1e3
        T = CO2.values[1::, 0]
        Cp = CO2.values[1::, 1]
        G = CO2.values[1::, 6] * 1e-3  # $\text{G} \equiv \Delta_f G^{\circ}$

        # Initialize models:
        model_Cp = FoKLRoutines.FoKL(kernel=1, UserWarnings=False)
        model_G = FoKLRoutines.FoKL(kernel=1, UserWarnings=False)

        # Train models:
        print("\nTraining Cp = f(T):")
        model_Cp.fit(T, Cp, clean=True)
        print("Done!")
        print("\nTraining G = f(T, Cp):")
        model_G.fit([T, Cp], G, clean=True)
        print("Done!")

        # Save models:
        model_Cp.save(os.path.join(dir, 'model_Cp.fokl'))
        model_G.save(os.path.join(dir, 'model_G.fokl'))

    # Validation of models:
    model_Cp.coverage3(plot=True, xaxis=0, xlabel='T (K)', ylabel='Cp', title="Validation of 'model_Cp'")
    model_G.coverage3(plot=True, xaxis=0, xlabel='T (K)', ylabel='G', title="Validation of 'model_G'")

    # Embed GP's in Pyomo model:
    m = fokl_to_pyomo([model_Cp, model_G], [['T'], ['T', 'Cp']], ['Cp', 'G'])

    # Set up objective (to maximize G and minimize Cp variance):
    m.obj = pyo.Objective(expr=abs(m.G), sense=pyo.maximize)

    # Solve:
    opt = pyo.SolverFactory('multistart')
    print("\nRunning Pyomo solver:")
    opt.solve(m, solver='ipopt', suppress_unbounded_warning=True)
    print("Done!")

    return m


def results(m):
    """Print solution results to terminal."""

    print("\nResults:")
    print("\n|       |            T |               Cp |           G |")
    print("|-------|--------------|------------------|-------------|")
    print("| Janaf | (1700, 1900) | (59.317, 60.049) | <= -396.353 |")
    print("| Pyomo |     ", "%.2f" % m.T(), "|          ", "%.3f" % m.Cp(), "|   ", "%.3f" % m.G(), "|")

    print("\nThe standard deviations of the GP models are:")
    print("    std(Cp) =", "%.3f" % np.sqrt(m.GP0_Cp_var()))
    print("     std(G) =", "%.3f" % np.sqrt(m.GP1_G_var()))

    print(f"\nThe error of the T solution is: {np.round((m.T() - 1800) / 18, 1)} %")


if __name__ == '__main__':
    m = main()
    results(m)
    print("\nEnd of 'pyomo_multiple_models' example.")

