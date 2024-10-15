import getKernels
__all__ = ['loadSplines']

def loadSplines(model, current):
        phis = current['phis']  # in case advanced user is testing other splines
        if isinstance(current['kernel'], int):  # then assume integer indexing  model.kernels'
            current['kernel'] = model.kernels[current['kernel']]  # update integer to string
        if current['phis'] is None:  # if default
            if current['kernel'] == model.kernels[0]:  # == 'Cubic Splines':
                current['phis'] = getKernels.sp500()
            elif current['kernel'] == model.kernels[1]:  # == 'Bernoulli Polynomials':
                current['phis'] = getKernels.bernoulli()
            elif isinstance(current['kernel'], str):  # confirm string before printing to console
                raise ValueError(f"The user-provided kernel '{current['phis']}' is not supported.")
            else:
                raise ValueError(f"The user-provided kernel is not supported.")
        return current