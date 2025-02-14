import jaxlib
import jax
import numpy as np
import jax.numpy as jnp
import warnings
from FoKL.FoKLRoutines import _process_kwargs, _merge_dicts, _str_to_bool

__all__ = ["evaluate_preprocess", "evaluate_jax", "evaluate_basis_jax"]
def evaluate_preprocess(model, inputs=None, betas=None, mtx=None, avgbetas=False, **kwargs):

    # Process keywords:
    default = {'minmax': None, 'draws': model.draws, 'clean': False, 'ReturnBounds': False,  # for evaluate
               '_suppress_normalization_warning': False}  # if called from coverage3
    default_for_clean = {'train': 1,
                         # For '_format':
                         'AutoTranspose': True, 'SingleInstance': False, 'bit': 64,
                         # For '_normalize':
                         'normalize': True, 'minmax': model.minmax, 'pillow': None, 'pillow_type': 'percent'}
    current = _process_kwargs(_merge_dicts(default, default_for_clean), kwargs)
    for boolean in ['clean', 'ReturnBounds']:
        current[boolean] = _str_to_bool(current[boolean])
    kwargs_to_clean = {}
    for kwarg in default_for_clean.keys():
        kwargs_to_clean.update({kwarg: current[kwarg]})  # store kwarg for clean here
        del current[kwarg]  # delete kwarg for clean from current
    if current['draws'] < 40 and current['ReturnBounds']:
        current['draws'] = 40
        warnings.warn("'draws' must be greater than or equal to 40 if calculating bounds. Setting 'draws=40'.")
    draws = current['draws']  # define local variable
    if betas is None:  # default
        if avgbetas:
            betas = model.avg_betas
        else:
            if draws > model.betas.shape[0]:
                draws = model.betas.shape[0]  # more draws than models results in inf time, so threshold
                model.draws = draws
                warnings.warn("Updated attribute 'self.draws' to equal number of draws in 'self.betas'.",
                              category=UserWarning)
            betas = model.betas[-draws::, :]  # use samples from last models
    else:  # user-defined betas may need to be formatted
        betas = np.array(betas)
        if betas.ndim == 1:
            betas = betas[np.newaxis, :]  # note transpose would be column of beta0 terms, so not expected
        if draws > betas.shape[0]:
            draws = betas.shape[0]  # more draws than models results in inf time, so threshold
        betas = betas[-draws::, :]  # use samples from last models
    if mtx is None:  # default
        mtx = model.mtx
    else:  # user-defined mtx may need to be formatted
        if isinstance(mtx, int):
            mtx = [mtx]
        mtx = np.array(mtx)
        if mtx.ndim == 1:
            mtx = mtx[np.newaxis, :]
            warnings.warn("Assuming 'mtx' represents a single model. If meant to represent several models, then "
                          "explicitly enter a 2D numpy array where rows correspond to models.")

    phis = model.phis

    # Automatically normalize and format inputs:
    if inputs is None:  # default
        inputs = model.inputs
        if current['clean']:
            warnings.warn("Cleaning was already performed on default 'inputs', so overriding 'clean' to False.",
                          category=UserWarning)
            current['clean'] = False
    else:  # user-defined 'inputs'
        if not current['clean']:  # assume provided inputs are already formatted and normalized
            normputs = inputs
            # if current['_suppress_normalization_warning'] is False:  # to suppress warning when evaluate called from coverage3
            #     warnings.warn("User-provided 'inputs' but 'clean=False'. Subsequent errors may be solved by enabling automatic formatting and normalization of 'inputs' via 'clean=True'.", category=UserWarning)
    if current['clean']:
        normputs = model.clean(inputs, kwargs_from_other=kwargs_to_clean)
    elif inputs is None:
        normputs = model.inputs
    else:
        normputs = np.array(inputs)

    m, mbets = np.shape(betas)  # Size of betas
    n = np.shape(normputs)[0]  # Size of normalized inputs
    mputs = int(np.size(normputs) / n)

    if model.setnos is None:
        setnos = np.random.choice(m, draws, replace=False)  # random draw selection
        model.setnos = setnos
    else:
        setnos = model.setnos

    minmax = model.minmax

    return normputs, setnos, phis, betas, mtx, minmax, draws, current

def evaluate_jax(model, inputs=None, betas=None, mtx=None, avgbetas=False, **kwargs):
    """
    Evaluate the FoKL model for provided inputs and (optionally) calculate bounds. Note 'evaluate_fokl' may be a
    more accurate name so as not to confuse this function with 'evaluate_basis', but it is unexpected for a user to
    call 'evaluate_basis' so this function is simply named 'evaluate'.

    Input:
        inputs == input variable(s) at which to evaluate the FoKL model == self.inputs (default)

    Optional Inputs:
        betas        == coefficients defining FoKL model                       == self.betas (default)
        mtx          == interaction matrix defining FoKL model                 == self.mtx (default)
        minmax       == [min, max] of inputs used for normalization            == None (default)
        draws        == number of beta terms used                              == self.draws (default)
        clean        == boolean to automatically normalize and format 'inputs' == False (default)
        ReturnBounds == boolean to return confidence bounds as second output   == False (default)
    """
    normputs, setnos, phis, betas, mtx, minmax, draws, current = evaluate_preprocess(model, inputs, betas, mtx, avgbetas, **kwargs)

    m, mbets = jnp.shape(betas)  # Size of betas
    n = jnp.shape(normputs)[0]  # Size of normalized inputs
    mputs = int(np.size(normputs) / n)

    X = jnp.zeros((n, mbets))
    normputs = jnp.asarray(normputs)

    phis = jnp.array(phis)
    setnos = jnp.array(setnos)
    mtx = jnp.array(mtx)

    if model.kernel == model.kernels[0]:  # == 'Cubic Splines':
        _, phind, xsm = model._inputs_to_phind(normputs)  # ..., phis=self.phis, kernel=self.kernel) already true
    else:
        raise ValueError("Kernel must be either 'Cubic Splines'")


    phind = jnp.ceil(normputs * 499)
    sett = (phind == 0)
    phind = phind + sett
    l_phis = 499
    r = 1 / l_phis  # interval of when basis function changes (i.e., when next cubic function defines spline)
    xmin = jnp.array((phind - 1) * r)
    X = (normputs - xmin) / r
    phind = phind.astype(int) - 1

    A = jnp.array([1, 2, 3])

    X_sc = np.stack([X**a for a in A], axis=2)
    def cubic_func(phis, phind, X, num, bet):

        return jax.numpy.where(
            num > 0,
            phis[num - 1][0][phind]
            + phis[num - 1][1][phind] * X[0]
            + phis[num - 1][2][phind] * X[1]
            + phis[num - 1][3][phind] * X[2],
            1.0
        )


    map_inputs = jax.vmap(cubic_func, in_axes=(None,0,0,0,None))
    map_dimensions = jax.vmap(
            map_inputs,
            in_axes=(None,None,None, 0 ,1)  # This maps over rows of phind and X
        )
    map_instances = jax.vmap(
        map_dimensions,
        in_axes=(None, 0, 0, None, None)  # This maps over columns of phind and X
    )
    X_vec = jax.numpy.prod(map_instances(phis, phind, X_sc, mtx.astype(int), betas[:,1:]), axis = 2)

    X = np.hstack([np.ones((n,1)),X_vec])

    def batched_matmul(X, betas, setnos):
        # Gather rows corresponding to `setnos`
        betas_subset = jax.lax.dynamic_slice(
            betas,
            start_indices=(setnos, 0),  # Start slicing at setnos
            slice_sizes=(1, betas.shape[1])  # Slice one row and all columns
        )
        betas_subset = jax.numpy.squeeze(betas_subset, axis=0)  # Remove singleton dimension

        # Perform the matrix multiplication
        return jax.numpy.transpose(jax.numpy.matmul(X, jax.numpy.transpose(betas_subset)))
    #

    jfunc = jax.vmap(batched_matmul, in_axes=(None, None, 0))
    modells = jfunc(X,betas,setnos.astype(int))
    mean = jax.numpy.mean(modells, axis=0)

    if current['ReturnBounds']:
        bounds = np.zeros((n, 2))  # note n == np.shape(data)[0] if data != 'ignore'
        cut = int(np.floor(draws * 0.025) + 1)
        for i in range(n):  # note n == np.shape(data)[0] if data != 'ignore'
            drawset = np.sort(modells[i, :])
            bounds[i, 0] = drawset[cut]
            bounds[i, 1] = drawset[draws - cut]
        return mean, bounds
    else:
        return mean

def evaluate_basis_jax(c, x):
    """
    Evaluate a basis function at a single point by providing coefficients, x value(s), and (optionally) the kernel.

    Inputs:
        > c == coefficients of a single basis functions
        > x == value of independent variable at which to evaluate the basis function

    Optional Input:
        > kernel == 'Cubic Splines' or 'Bernoulli Polynomials' == self.kernel (default)
        > d      == integer representing order of derivative   == 0 (default)

    Output (in Python syntax, for d=0):
        > if kernel == 'Cubic Splines':
            > basis = c[0] + c[1]*x + c[2]*(x**2) + c[3]*(x**3)
        > if kernel == 'Bernoulli Polynomials':
            > basis = sum(c[k]*(x**k) for k in range(len(c)))
    """

    basis = c[0] + c[1] * x + c[2] * (x ** 2) + c[3] * (x ** 3)

    return basis

def eval_loop_cubic(mtx, mbets, xsm, phis, phind, n, mputs):
    for i in range(n):
        for j in range(1, mbets):
            phi = 1
            for k in range(mputs):
                num = mtx[j - 1, k]
                if num:
                    nid = int(num - 1)
                    coeffs = [phis[nid][order][phind[i, k]] for order in range(4)]  # coefficients for cubic
                    phi *= evaluate_basis_jax(coeffs, xsm[i, k])  # multiplies phi(x0)*phi(x1)*etc.
            return phi



