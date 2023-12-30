import math
import numpy as np
import copy
import matplotlib.pyplot as plt
import warnings
from inspect import getsourcefile
from os.path import abspath
import time


def smooth_coefficients(phis):
    '''
    Smooth the endpoints of spline coefficients by holding nth-order derivative constant.
    '''

    ReturnPlot = 0  # for development and demonstration of smoothing

    # ----------------
    # OPTIMIZATION PARAMETERS:

    const = 'third'  # order of derivative held constant
    noise = {0: 0, 1: 0, 2: 9, 3: 11}  # number of values per x^n coefficients that are errors

    splineMin_x2 = 0  # min and max id's of splines to apply smoothing of the x^2 coefficients
    splineMax_x2 = 18

    splineMin_x3 = 0  # min and max id's of splines to apply smoothing of the x^3 coefficients
    splineMax_x3 = 34

    # END OF OPTIMIZATION PARAMETERS.
    # -----------------------

    dx = 1/(499-1)
    dx_x2 = 2*dx
    dx2 = math.pow(dx,2)
    dx3_x2 = 2*math.pow(dx,3)
    dx4 = math.pow(dx,4)

    x = np.linspace(0, 1, 499)
    id = np.linspace(1,499,499)
    phis0 = copy.deepcopy(phis)
    for spline in range(1,500,2):  # determines which splines to smooth (set to range(1,500,2) for even only)
        for order in range(4):

            # # Plot individual coefficients (for development):
            # plt.figure(1)
            # plt.title(f'spline = {spline + 1}, coeff. = {order}')
            # plt.xlabel('index')
            # plt.plot(id, phis[spline][order])
            # plt.show()

            if (order==2 and splineMin_x2<spline<splineMax_x2) or (order==3 and splineMin_x3<spline<splineMax_x3):

                if const == 'first' or const == 'second':

                    f_xm1_L = phis[spline][order][noise.get(order)]  # first valid coefficient (not noise)
                    f_x_L = phis[spline][order][noise.get(order) + 1]
                    f_xp1_L = phis[spline][order][noise.get(order) + 2]

                    f_xp1_R = phis[spline][order][-noise.get(order) - 1]  # last valid coefficient (not noise)
                    f_x_R = phis[spline][order][-noise.get(order) - 2]
                    f_xm1_R = phis[spline][order][-noise.get(order) - 3]

                    if const == 'first':

                        d1_L = (f_xp1_L - f_xm1_L) / dx_x2
                        d1_R = (f_xp1_R - f_xm1_R) / dx_x2

                        for i in reversed(range(noise.get(order))):

                            # LHS:

                            f_xp1_L = f_x_L  # = phis0[spline][order][i + 2]
                            f_x_L = f_xm1_L  # = phis0[spline][order][i + 1]

                            f_xm1_L = -d1_L*dx_x2 + f_xp1_L
                            phis[spline][order][i] = f_xm1_L

                            # RHS:

                            i = i + 1

                            f_xm1_R = f_x_R  # = phis0[spline][order][-i - 2]
                            f_x_R = f_xp1_R  # = phis0[spline][order][-i - 1]

                            f_xp1_R = d1_R * dx_x2 + f_xm1_R
                            phis[spline][order][-i] = f_xp1_R

                    elif const == 'second':

                        d2_L = (f_xp1_L - 2*f_x_L + f_xm1_L) / dx2
                        d2_R = (f_xp1_R - 2 * f_x_R + f_xm1_R) / dx2

                        for i in reversed(range(noise.get(order))):

                            # LHS:

                            f_xp1_L = f_x_L  # = phis0[spline][order][i + 2]
                            f_x_L = f_xm1_L  # = phis0[spline][order][i + 1]

                            f_xm1_L = d2_L*dx2 - f_xp1_L + 2*f_x_L
                            phis[spline][order][i] = f_xm1_L

                            # RHS:

                            i = i + 1

                            f_xm1_R = f_x_R  # = phis0[spline][order][-i - 2]
                            f_x_R = f_xp1_R  # = phis0[spline][order][-i - 1]

                            f_xp1_R = d2_R * dx2 - f_xm1_R + 2 * f_x_R
                            phis[spline][order][-i] = f_xp1_R

                elif const == 'third' or const == 'fourth':

                    f_xm2_L = phis[spline][order][noise.get(order)] # first valid coefficient (not noise)
                    f_xm1_L = phis[spline][order][noise.get(order) + 1]
                    f_x_L = phis[spline][order][noise.get(order) + 2]
                    f_xp1_L = phis[spline][order][noise.get(order) + 3]
                    f_xp2_L = phis[spline][order][noise.get(order) + 4]

                    f_xp2_R = phis[spline][order][-noise.get(order) - 1]  # last valid coefficient (not noise)
                    f_xp1_R = phis[spline][order][-noise.get(order) - 2]
                    f_x_R = phis[spline][order][-noise.get(order) - 3]
                    f_xm1_R = phis[spline][order][-noise.get(order) - 4]
                    f_xm2_R = phis[spline][order][-noise.get(order) - 5]

                    if const == 'third':

                        d3_L = (f_xp2_L - 2*f_xp1_L + 2*f_xm1_L - f_xm2_L) / dx3_x2
                        d3_R = (f_xp2_R - 2 * f_xp1_R + 2 * f_xm1_R - f_xm2_R) / dx3_x2

                        for i in reversed(range(noise.get(order))):

                            # LHS:

                            f_xp2_L = f_xp1_L  # = phis0[spline][order][i + 4]
                            f_xp1_L = f_x_L  # = phis0[spline][order][i + 3]
                            f_x_L = f_xm1_L  # = phis0[spline][order][i + 2]
                            f_xm1_L = f_xm2_L  # = phis0[spline][order][i + 1]

                            f_xm2_L = -d3_L*dx3_x2 + f_xp2_L - 2*f_xp1_L + 2*f_xm1_L
                            phis[spline][order][i] = f_xm2_L

                            # RHS:

                            i = i + 1

                            f_xm2_R = f_xm1_R  # = phis0[spline][order][-i - 4]
                            f_xm1_R = f_x_R  # = phis0[spline][order][-i - 3]
                            f_x_R = f_xp1_R  # = phis0[spline][order][-i - 2]
                            f_xp1_R = f_xp2_R  # = phis0[spline][order][-i - 1]

                            f_xp2_R = d3_R * dx3_x2 + f_xm2_R + 2 * f_xp1_R - 2 * f_xm1_R
                            phis[spline][order][-i] = f_xp2_R

                    elif const == 'fourth':

                        d4_L = (f_xp2_L - 4 * f_xp1_L + 6 * f_x_L - 4 * f_xm1_L + f_xm2_L) / dx4
                        d4_R = (f_xp2_R - 4 * f_xp1_R + 6 * f_x_R - 4 * f_xm1_R + f_xm2_R) / dx4

                        for i in reversed(range(noise.get(order))):

                            # LHS:

                            f_xp2_L = f_xp1_L  # = phis0[spline][order][i + 4]
                            f_xp1_L = f_x_L  # = phis0[spline][order][i + 3]
                            f_x_L = f_xm1_L  # = phis0[spline][order][i + 2]
                            f_xm1_L = f_xm2_L  # = phis0[spline][order][i + 1]

                            f_xm2_L = d4_L * dx4 - f_xp2_L + 4 * f_xp1_L - 6 * f_x_L + 4 * f_xm1_L
                            phis[spline][order][i] = f_xm2_L

                            # RHS:

                            i = i + 1

                            f_xm2_R = f_xm1_R  # = phis0[spline][order][-i - 4]
                            f_xm1_R = f_x_R  # = phis0[spline][order][-i - 3]
                            f_x_R = f_xp1_R  # = phis0[spline][order][-i - 2]
                            f_xp1_R = f_xp2_R  # = phis0[spline][order][-i - 1]

                            f_xp2_R = d4_R * dx4 - f_xm2_R + 4 * f_xp1_R - 6 * f_x_R + 4 * f_xm1_R
                            phis[spline][order][-i] = f_xp2_R

        if ReturnPlot:

            plt.figure(1)
            plt.title(f'spline = {spline + 1}; {const} deriv. const.; removed ({noise.get(2)}, {noise.get(3)}) points for (x^2, x^3) coefficients')

            plt.subplot(3, 2, 1)
            plt.plot(id, phis0[spline][2])
            plt.title('original')
            plt.ylabel('coeff. = 2')

            plt.subplot(3, 2, 3)
            plt.plot(id, phis0[spline][3])
            plt.ylabel('coeff. = 3')

            plt.subplot(3, 2, 2)
            plt.plot(id, phis[spline][2])
            plt.title('smoothed')

            plt.subplot(3, 2, 4)
            plt.plot(id, phis[spline][3])

            plt.subplot(3, 2, 5)
            plt.plot(x, phis0[spline][0]+phis0[spline][1]*x+phis0[spline][2]*x*x+phis0[spline][3]*x*x*x)
            plt.ylabel('basis')
            plt.xlabel('index')

            plt.subplot(3, 2, 6)
            plt.plot(x, phis[spline][0] + phis[spline][1] * x + phis[spline][2] * x*x + phis[spline][3] * x*x*x)
            plt.xlabel('index')

            plt.show()
            plt.cla()

    return phis


def sp500(**kwargs):
    '''
    Return 'phis', a [500 x 4 x 499] Python tuple of lists, of double-precision basis functions' coefficients.
    '''

    kwargs_upd = {'filename': 'splineCoefficient500_highPrecision_smoothed', 'Smooth': 0, 'Save': 0}
    for kwarg in kwargs.keys():
        if kwarg not in kwargs_upd.keys():
            raise ValueError(f"Unexpected keyword argument: {kwarg}")
        else:
            kwargs_upd[kwarg] = kwargs.get(kwarg, kwargs_upd.get(kwarg))
    filename = kwargs_upd.get('filename')
    Smooth = kwargs_upd.get('Smooth')
    Save = kwargs_upd.get('Save')
    if Save == 1 and Smooth == 0:
        warnings.warn("The spline coefficients were not save because they were not requested to be smoothed.",category=UserWarning)

    # Merge filename with path to filename:
    path_to_py = abspath(getsourcefile(lambda: 0))
    for i in range(1,len(path_to_py)):
        if path_to_py[-i] == '\\':
            path_to_txt = f'{path_to_py[0:-i]}\kernels\{filename}.txt'
            if Save:
                save_to_txt = f'{path_to_py[0:-i]}\kernels\{filename}_smoothed{round(time.time())}.txt'
            break

    # Read double-precision values from file to [249500 x 4] numpy array:
    phis_raw = np.loadtxt(path_to_txt, delimiter=',', dtype=np.double)

    # Process [249500 x 4] numpy array to [500 x 4 x 499] list (with outer layer as tuple to match FoKL v2):
    phis = []
    for id_spline in range(500):
        id_lo = id_spline*499
        id_hi = id_lo+499
        phis_ele = phis_raw[id_lo:id_hi,:]
        phis.append([phis_ele[:,0].tolist(), phis_ele[:,1].tolist(), phis_ele[:,2].tolist(), phis_ele[:,3].tolist()])

    phis = tuple(phis)

    # Smooth and save coefficients to new text file:
    if Smooth:
        phis = smooth_coefficients(phis)
        if Save:
            phis_out = np.transpose(np.array([phis[0][0],phis[0][1],phis[0][2],phis[0][3]]))
            for i in range(1,500):
                phis_out_i = np.transpose(np.array([phis[i][0],phis[i][1],phis[i][2],phis[i][3]]))
                phis_out = np.concatenate((phis_out,phis_out_i))
            np.savetxt(save_to_txt, phis_out, delimiter=',')

    return phis

