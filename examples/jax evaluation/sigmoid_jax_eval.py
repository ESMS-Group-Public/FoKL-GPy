"""
[ExAMPLE]: Sigmoid

This is an example of FoKL modeling a dataset based on an arbitrary sigmoid function. In the following it will be shown
how to initialize the FoKL class (i.e., model), how to train the model on the dataset by calling 'fit', and how to
perform some very basic post-processing with a 'coverage3' plot and included RMSE calculation.
"""
import timeit

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
from FoKL.JAX_Eval import *
import matplotlib.pyplot as plt


def main():
    # Load Previously built FoKL Model

    model = FoKLRoutines.load("sigmoid_model.fokl")

    n = [3, 5, 9, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000] # Number of evaluation points

    # initialize vectors
    to_vec = []
    tn_vec = []

    for i in range(len(n)):
        points = n[i]
        inputs = np.transpose(np.array([np.linspace(0,1,int(points)),np.linspace(0,1,int(points))]))
        model.inputs = inputs

        t1 = timeit.default_timer()
        mo = model.evaluate()
        t2 = timeit.default_timer() - t1

        t3 = timeit.default_timer()
        mn = evaluate_jax(model)
        t4 = timeit.default_timer() - t3

        to_vec.append(t2)
        tn_vec.append(t4)

        print(f'loop {i} (# of points = {points}) finished with \n time original = {to_vec[i]} \n time JAX = {tn_vec[i]} \n')

    plt.plot(to_vec)
    plt.plot(tn_vec)
    plt.show()


if __name__ == '__main__':
    main()
    print("\nEnd of Sigmoid example.")

