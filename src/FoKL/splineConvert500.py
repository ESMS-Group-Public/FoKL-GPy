import numpy as np

def splineconvert500(A):
    """

    Converts coefficients in the spline_coefficient_500.txt file that's fed into the
    function and returns a 'phi' variable for use with emulator or emulator_Xin
    functions below

    For Yinkai's orthogonal basis functions that are based on a normalized
    interval

    """

    coef = np.loadtxt(A)

    phi = []
    for i in range(500):
        a = coef[i * 499:(i + 1) * 499, 0]
        b = coef[i * 499:(i + 1) * 499, 1]
        c = coef[i * 499:(i + 1) * 499, 2]
        d = coef[i * 499:(i + 1) * 499, 3]

        phi.append([a, b, c, d])

    return phi
