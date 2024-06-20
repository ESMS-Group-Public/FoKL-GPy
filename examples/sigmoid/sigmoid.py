"""
[ExAMPLE]: Sigmoid

This is an example of FoKL modeling a dataset based on an arbitrary sigmoid function. In the following it will be shown
how to initialize the FoKL class (i.e., model), how to train the model on the dataset by calling 'fit', and how to
perform some very basic post-processing with a 'coverage3' plot and included RMSE calculation.
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


def main():
    # Known dataset:
    x_grid = np.loadtxt(os.path.join(dir, 'x.csv'), dtype=float, delimiter=',')  # first input
    y_grid = np.loadtxt(os.path.join(dir, 'y.csv'), dtype=float, delimiter=',')  # second input
    z_grid = np.loadtxt(os.path.join(dir, 'z.csv'), dtype=float, delimiter=',')  # data

    # Some formatting of dataset (i.e., reshaping grid matrices into vectors via fortran index order):
    m, n = np.shape(x_grid) # == np.shape(y_grid) == np.shape(z_grid) == dimensions of grid
    x = np.reshape(x_grid, (m * n, 1), order='F')
    y = np.reshape(y_grid, (m * n, 1), order='F')
    z = np.reshape(z_grid, (m * n, 1), order='F')

    # Initializing FoKL model with some user-defined hyperparameters (leaving others blank for default values) and
    # turning off user-warnings (i.e., warnings from FoKL) since working example requires no troubleshooting:
    model = FoKLRoutines.FoKL(a=9, b=0.01, atau=3, btau=4000, aic=True, UserWarnings=False)

    # Running emulator routine (i.e., 'fit') to train model:
    print("\nCurrently training model...\n")
    a, b, ev = model.fit([x, y], z, clean=True)

    # Evaluating and visualizing predicted values of data as a function of all inputs (train set plus test set):
    print("\nDone! Please close the figure to continue.")
    _, _, rmse = model.coverage3(plot=True, title='Sigmoid Example')

    # Post-processing:
    print(f"\nThe 'coverage3' method returned:\n    RMSE = {rmse}")


if __name__ == '__main__':
    main()
    print("\nEnd of Sigmoid example.")

