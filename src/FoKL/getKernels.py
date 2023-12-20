from inspect import getsourcefile
from os.path import abspath
import numpy as np


def sp500():
    '''
    Return 'phis', a [500 x 4 x 499] Python tuple of lists, of double-precision basis functions' coefficients.
    '''
    filename = 'splineCoefficient500_highPrecision.txt'

    # Merge filename with path to filename:
    path_to_py = abspath(getsourcefile(lambda: 0))
    # DEV - REWRITE TO SUBTRACT FOLDERS
    for i in range(1,len(path_to_py)):
        if path_to_py[-i] == '\\':
            path_to_txt = f'{path_to_py[0:-i]}\{filename}'
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

    return phis

