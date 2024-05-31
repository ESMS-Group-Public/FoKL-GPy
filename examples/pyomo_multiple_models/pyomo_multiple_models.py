"""
[EXAMPLE]: Pyomo Multiple Models

This is an example of embedding multiple FoKL GP models in a single Pyomo model, with user-defined variable names.
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
from thermochem.janaf import Janafdb
import pyomo.environ as pyo


def main():

    try:
        bFoKL__GmH0_T = FoKLRoutines.load('FoKL__GmH0_T.fokl')
        FoKL__Delta_fG = FoKLRoutines.load('FoKL__Delta_fG.fokl')
    except Exception as exception:

        # Load dataset:

        CO2 = Janafdb().getphasedata(filename='C-095').rawdata  # https://janaf.nist.gov/tables/C-095.txt

        T = CO2.values[1::, 0]
        Cp = CO2.values[1::, 1]
        GmH0_T = CO2.values[1::, 3]  # [G - H(Tr)]/T
        Delta_fG = CO2.values[1::, 6]

        # Train models:

        FoKL__GmH0_T = FoKLRoutines.FoKL(kernel=1)
        FoKL__Delta_fG = FoKLRoutines.FoKL(kernel=1)

        FoKL__GmH0_T.fit([T, Cp], GmH0_T, clean=True)
        FoKL__Delta_fG.fit([T, Cp], Delta_fG, clean=True)

        # Initialize Pyomo model and embed GP's:

        m = pyo.ConcreteModel()
        FoKL__GmH0_T.to_pyomo(m, ['T', 'Cp'], 'GmH0_T')
        FoKL__Delta_fG.to_pyomo(m, ['T', 'Cp'], 'Delta_fG', overwrite=False)

        # save
        FoKL__GmH0_T.save('FoKL__GmH0_T')
        FoKL__Delta_fG.save('FoKL__Delta_fG')

    # debug fokl_to_pyomo
    m = fokl_to_pyomo([FoKL__GmH0_T, FoKL__Delta_fG], [['T', 'Cp'], ['T', 'Cp']], ['GmH0_T', 'Delta_fG'], 3)

    breakpoint()


if __name__ == '__main__':
    main()

