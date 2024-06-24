"""
[TUTORIAL]: self.fit(self, inputs=None, data=None, **kwargs)
            self.evaluate(self, inputs=None, betas=None, mtx=None, **kwargs)

This is a tutorial for the 'fit' and 'evaluate' methods, necessary steps in most all FoKL models. In the
following script, it will be shown how 'clean' may be called directly or from within the 'fit' and 'evaluate' arguments.
If calling from 'fit' or 'evaluate', set 'clean=True' and include any desired keyword arguments from 'clean'.

In this tutorial, the following will be demonstrated:

    1) Clean a dataset...

        a) with known data.
        b) without known data.

        -   In both cases, (a) and (b), the known inputs become normalized and formatted for use in further FoKL
        methods. The data will also be formatted if known, but has no need to be normalized. Note there is an option
        within the argument of both 'evaluate' and 'fit' to automatically call 'clean' internally, so that (1) is not
        usually necessary.

    2) Evaluate a model...

        a) trained here on known data, which is possible only if the dataset has known data.
        b) provided from elsewhere or hypothesized, which is necessary if the dataset does not have known data.

        -   When training a model, like for (a), it is often more succinct to call 'clean' from the 'fit' argument like
        is shown in the commented-out equivalent to the 'a.fit()' line, and in this way (1) may be skipped.

        -   When not training a model, like for (b), then it is often more succinct to call 'clean' from the 'evaluate'
        argument like is shown in the commented-out equivalents to the 'y_b1' and 'y_b2' lines, and in either of these
        ways (1) may be skipped. It is recommended to use (b, alt. 2) or (b, alt. 3) if looping through several models
        or performing a quick evaluation of a single model, whereas (b, alt. 1) is more apt for when there is a single
        FoKL model per FoKL class that should persist as an object. However, (b, alt. 1) is highly sensitive to
        formatting issues so please ensure both 'betas' and 'mtx' are 2D numpy arrays, where columns of 'betas' index
        the 0th-Nth terms in the model and rows of 'mtx' index the 1st-Nth terms.
"""
from FoKL import FoKLRoutines
import os
dir = os.path.abspath(os.path.dirname(__file__))  # directory of script
# # # -----------------------------------------------------------------------
# # UNCOMMENT IF USING LOCAL FOKL PACKAGE:
# import sys
# sys.path.append(os.path.join(dir, '..', '..'))  # package directory
# from src.FoKL import FoKLRoutines
# # -----------------------------------------------------------------------
import numpy as np


def main():
    a = FoKLRoutines.FoKL()
    b = FoKLRoutines.FoKL()

    # Known dataset:

    x0 = np.linspace(0, 1, 10)              # first input
    x1 = np.random.rand(10)                 # second input
    x2 = np.exp(x0)                         # third input
    y = x0 + x1 + x2                        # data

    # (1) Clean a dataset...

    a.clean([x0, x1, x2], y, train=0.6)     # (a) with known data, using 60% of the dataset for the training set.
    b.clean([x0, x1, x2])                   # (b) without known data.

    # (2) Evaluate a model...

    # (a) trained here on known data.
    a.fit()                                 # == a.fit([x0, x1, x2], y, clean=True, train=0.6)
    y_a = a.evaluate()                      # == y_a = a.evaluate(a.inputs, a.betas, a.mtx)

    # (b, alt. 1) provided from elsewhere or hypothesized, using class attributes to define the model semi-permanently.
    b.betas = np.array([2.49, 1.52, 2.15])[np.newaxis]
    b.mtx = np.array([[0, 0, 1], [0, 1, 0]])
    y_b1 = b.evaluate()                     # == y_b1 = b.evaluate([x0, x1, x2], clean=True)

    # (b, alt. 2) provided from elsewhere or hypothesized, using positional arguments.
    betas = [2.49, 1.52, 2.15]
    mtx = [[0, 0, 1], [0, 1, 0]]
    y_b2 = b.evaluate(None, betas, mtx)     # == y_b2 = b.evaluate([x0, x1, x2], betas, mtx, clean=True)

    # (b, alt. 3) provided from elsewhere or hypothesized, using keyword arguments.
    y_b3 = b.evaluate(betas=[2.49, 1.52, 2.15], mtx=[[0, 0, 1], [0, 1, 0]])


if __name__ == '__main__':
    main()
    print("\nEnd of clean tutorial.")

