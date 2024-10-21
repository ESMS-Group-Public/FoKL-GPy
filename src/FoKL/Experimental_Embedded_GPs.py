# Import Relevant Libraries
# Base FoKL to get Kernels for basis functions
from FoKL import getKernels
import itertools
import numpy as np

# Jax for data processing
import jax
import jax.numpy as jnp
from jax import grad
from jax import random
from jax import jit
from jax.lax import fori_loop, cond, while_loop
from jax import lax
from jax.scipy.special import gamma

def inverse_gamma_pdf(y, alpha, beta):
    """
    Calculate the PDF of the inverse gamma distribution.

    Parameters:
    y (float or array-like): The value(s) at which to evaluate the PDF.
    alpha (float): Shape parameter of the inverse gamma distribution.
    beta (float): Scale parameter of the inverse gamma distribution.

    Returns:
    float or array-like: The PDF value(s) at y.
    """
    # Ensure y is a JAX array for compatibility
    y = jnp.array(y)
    
    # Calculate the PDF of the inverse gamma distribution
    pdf = (beta ** alpha / gamma(alpha)) * y ** (-alpha - 1) * jnp.exp(-beta / y)
    return pdf
                    
class GP:
    """
    This class is for a GP object which represents a generalized mathematical function to be used inside of 
    a given physics model.  This class does not store any information explicitly, but is rather used 
    to provide a more friendly user API when writing to likelihood equation and interface with the
    main Embedded_GP_Model object. 
    """
    def __init__(self):
        """
        Initializes an instance of the GP (Gaussian Process) class. This constructor sets up
        initial configuration or state for the class instance.

        Attributes:
            init (str): A simple initialization attribute set to 'test' as a placeholder.

        Example:
            # Creating an instance of the GP class
            gp_instance = GP()
        """
        self.init = 'test'

    def inputs_to_phind(self, inputs, phis):
        """
        Processes and normalizes inputs to compute indices for accessing spline coefficients
        and additional transformations needed for spline evaluation.

        This method normalizes the input values twice: first, to an index corresponding to the
        spline coefficients and second, to calculate the offset within the specific segment of the spline.

        Args:
            inputs (jax.numpy.ndarray): Normalized input values (expected to be in the range [0, 1]) that represent
                                        positions within the spline domain.
            phis (jax.numpy.ndarray): Spline coefficients organized in an array where each sub-array
                                    contains coefficients for a different segment of the spline.

        Returns:
            tuple: Contains:
                - phind (jax.numpy.ndarray): Array of indices pointing to the specific segments of the spline
                                            coefficients adjusted for zero-based indexing.
                - xsm (jax.numpy.ndarray): Normalized distance of the inputs from the beginning of their respective
                                        spline segment, adjusted for further computations.

        Notes:
            - The method assumes that `inputs` are scaled between 0 and 1. It multiplies `inputs` by the length
            of the coefficients to map these inputs to an index in the coefficients array.
            - `xsm` is computed to represent the distance from the exact point within the spline segment that `inputs`
            corresponds to, facilitating precise spline evaluations.
        """

        L_phis = len(phis[0][0])  # length of coeff. in basis funtions
        phind = jnp.array(jnp.ceil(inputs * L_phis), dtype=int)  # 0-1 normalization to 0-499 normalization

        phind = jnp.expand_dims(phind, axis=-1) if phind.ndim == 1 else phind

        set = (phind == 0)  # set = 1 if phind = 0, otherwise set = 0
        phind = phind + set

        xsm = L_phis * inputs - phind
        phind = phind - 1

        return phind, xsm

    def GP_eval(self, inputs, discmtx, phis, betas):
        """
        Evaluates a Gaussian Process (GP) model using the provided inputs, interaction matrix,
        spline coefficients, and beta coefficients. The function computes the GP prediction by constructing
        a feature matrix from inputs and applying transformations based on the spline coefficients.

        Args:
            inputs (jax.numpy.ndarray): The input data matrix where each row represents a different input instance.
                                        Assumed normalized from 0-1.
            discmtx (jax.numpy.ndarray): Interaction matrix that indicates which basis functions are active
                                        for each feature. Can be a scalar if the model has only one input.
            phis (jax.numpy.ndarray): Array containing spline coefficients. Each set of coefficients corresponds
                                    to a different basis function.
            betas (jax.numpy.ndarray): Coefficient vector for the linear combination of basis functions to form
                                    the final prediction.

        Returns:
            jax.numpy.ndarray: The predicted values computed as a linear combination of transformed basis functions,
                            shaped according to the input matrix.

        Notes:
            This function constructs a feature matrix from the `inputs` by mapping them
            to their corresponding spline coefficients in `phis`. It then computes a transformation for each
            input feature, accumulating contributions from each basis function specified in `discmtx`. The
            result is a linear combination of these features weighted by the `betas`.

            The function handles different dimensions of input and discriminative matrices gracefully, padding
            with zeros where necessary to align dimensions.

        Example:
            # Example usage assuming predefined matrices `inputs`, `discmtx`, `phis`, and `betas`
            gp_model = GP()
            prediction = gp_model.GP_eval(inputs, discmtx, phis, betas)
            print("GP Predictions:", prediction)
        """

        # building the matrix by calculating the corresponding basis function outputs for each set of inputs
        minp, ninp = jnp.shape(inputs)

        if jnp.shape(discmtx) == ():  # part of fix for single input model
            mmtx = 1
        else:
            mmtx, null = jnp.shape(discmtx)

        # if jnp.size(Xin) == 0:
        Xin = jnp.ones((minp, 1))
        mxin, nxin = jnp.shape(Xin)
        
        if mmtx - nxin < 0:
            X = Xin
        else:
            X = jnp.append(Xin, jnp.zeros((minp, mmtx - nxin)), axis=1)

        phind, xsm = self.inputs_to_phind(inputs=inputs, phis=phis)

        null, nxin2 = jnp.shape(X)
        additional_cols = max(0, mmtx + 1 - nxin2)
        X = jnp.concatenate((X, jnp.zeros((minp, additional_cols))), axis=1)

        def body_fun_k(k, carry):
            i, j, phi_j = carry
            num = discmtx[j - 1, k] if discmtx.ndim > 1 else discmtx
            # Define the operation to perform if num is not zero
            def true_fun(_):
                nid = num - 1
                term = (phis[nid, 0, phind[i, k]] +
                        phis[nid, 1, phind[i, k]] * xsm[i, k] +
                        phis[nid, 2, phind[i, k]] * xsm[i, k] ** 2 +
                        phis[nid, 3, phind[i, k]] * xsm[i, k] ** 3)
                return phi_j * term

            # Define the operation to perform if num is zero
            def false_fun(_):
                return phi_j

            # Using lax.cond to conditionally execute true_fun or false_fun
            phi_j = lax.cond(num != 0, true_fun, false_fun, None)
            
            return (i, j, phi_j)

        def body_fun_j(j, carry):
            i, X = carry
            phi_j_initial = 1.0
            # Carry includes `j` to be accessible inside body_fun_k
            carry_initial = (i, j, phi_j_initial)
            carry = lax.fori_loop(0, ninp, body_fun_k, carry_initial)
            _, _, phi_j = carry  # Unpack the final carry to get phi_j
            X = X.at[i, j].set(phi_j)
            return (i, X)
        
        def body_fun_i(i, X):
            carry_initial = (i, X)
            _, X = lax.fori_loop(nxin, mmtx + 1, body_fun_j, carry_initial)
        
            return X

        X = lax.fori_loop(0, minp, body_fun_i, X)
        prediction = jnp.matmul(X, betas)
        return prediction   
       
class Embedded_GP_Model:
    """
    Manages multiple Gaussian Process (GP) models, allowing for physical models
    that involve multiple GP models to be evaluate simultaneously.

    Attributes:
        GP (tuple of GaussianProcess): Stores the tuple of Gaussian Processes.
        key (jax.random.PRNGKey): Pseudo-random number generator key.
        discmtx (jax.numpy.ndarray): Interaction Matrix for model proposals.
        betas (jax.numpy.ndarray): Coefficients of model terms. Single vector containing
            all beta values for all GP models that is split later on.

    Args:
        *GP (GaussianProcess): A variable number of Gaussian Process objects.
    """
    
    def __init__(self, *GP):
        """
        Initializes a Multiple_GP_Model instance with several Gaussian Process models.

        Parameters:
            *GP (tuple of GaussianProcess): A tuple of Gaussian Process objects,
            each of which should be an instance of a Gaussian Process model with 
            the custom GP class previously created.
        """
        # Define critical parameters
        self.GP = GP
        self.key = random.PRNGKey(0)
        # Define placeholder values needed for GP_Processing to run when the 
        # set_equation function is ran.
        self.discmtx = jnp.array([[1]])
        self.betas = jnp.ones(len(GP)*(len(self.discmtx)+1) + 1) # Add one for sigma sampling
        # self.sigma_alpha = 2
        # self.sigma_beta = 0.01

    def GP_Processing(self):
        """
        Processes multiple Gaussian Processes (GPs) using a shared beta vector that is split among the processes.
        Each GP is evaluated with a subset of the beta parameters and additional inputs specific to each GP.
        The easiest way to think of this is given a set of betas, an interaction matrix, and the inputs, it 
        computes the results of the model and stores them in a two dimensional vector that is of
        size [# of GPs, # of inputs] needed to calculate the negative log likelihood.

        This method updates the `Processed_GPs` attribute of the class with the results of the GP evaluations.

        Side Effects:
            Modifies the `Processed_GPs` attribute by storing the results of the evaluations.

        Assumptions:
            - The class must have an attribute `betas` which is a JAX numpy array of beta coefficients.
            - The class must have an attribute `GP`, a tuple of GaussianProcess instances.
            - Each GaussianProcess instance in `GP` must have a method `GP_eval` (from the custom class) which
            expects the inputs, interaction matrix (`discmtx`), phis, and relevant segment of betas.
            - `inputs` and `phis` attributes must be set in the class manually by user, used as parameters 
            in the GP evaluations.
        """
        # Separate betas by GP into matrix of size [# of GPs, # of betas]
        num_functions = len(self.GP)
        num_betas = int((len(self.betas) - 1)/num_functions)
        betas_list = self.betas[:-1].reshape(num_functions,num_betas)

        # Evaluate GPs and save results into matrix of size [# of GPs, # of inputs]
        GP_results = jnp.empty((0,len(self.inputs)))
        for idx, _ in enumerate (self.GP):
              result = self.GP[idx].GP_eval(self.inputs, self.discmtx, jnp.array(self.phis), betas_list[idx])
              GP_results = jnp.append(GP_results, result.reshape(1, -1), axis=0)

        self.Processed_GPs = GP_results
    
    def set_equation(self, equation_func):
        """
        Sets the mathematical equation that contains multiple GPs and stores it in the `equation` attribute. 
        It first processes the GPs by invoking `GP_Processing` because that is what the user needs to define
        in their model as a placeholder for the GP model (probably a better way to do that and recommended
        area for improvement).

        Args:
            equation_func (callable): A function that defines the equation to be used with the processed
                                    Gaussian Processes results. This is the proposed physical model.

        Side Effects:
            - Calls `GP_Processing` which processes all Gaussian Processes as defined in the `GP` attribute
            and updates the `Processed_GPs` attribute with the results.
            - Sets the `equation` attribute to the passed `equation_func`, which can be used later
            in conjunction with the processed GP results.

        Example:
            # Define an equation function
            def my_equation():
                # Example of CSTR Reaction Kinetics
                r_co2 = -(jnp.exp(-(multi_gp_model.Processed_GPs[0]))*C_CO2*C_Sites - jnp.exp(-(multi_gp_model.Processed_GPs[1]))*C_CO2_ADS)
                return r_co2

            # Assuming `set_equation` is a method of the `Multiple_GP_Model` class and an instance `multi_gp_model` has been created:
            multi_gp_model.set_equation(my_equation)  # This will process the GPs and set the new equation function.

        Note:
            Ensure that `equation_func` is compatible with the output format of `Processed_GPs` as generated
            by `GP_Processing` to avoid runtime errors.  This means using JAX numpy.
        """
        self.GP_Processing()
        self.equation = equation_func
    
    def neg_log_likelihood(self, betas):
        """
        Calculates the overall negative log likelihood for the model results using a single specified
        set of beta coefficients. This method updates the betas, processes the Gaussian Processes, and
        applies the set equation to calculates the negative log likelihood based on the difference
        between observed data and model results.

        Args:
            betas (jax.numpy.ndarray): An array of beta coefficients to be used in the Gaussian Processes.
                                    These coefficients are set as the new value of the `betas` attribute.

        Returns:
            float: The sum of the negative log likelihood for the model results across all data points.

        Side Effects:
            - Updates the `betas` attribute with the new beta coefficients.
            - Processes the Gaussian Processes by invoking `GP_Processing`.
            - Uses the equation set in the `equation` attribute to compute results.
            - Updates internal state based on computations.

        Note:
            - This method assumes that `self.data` and `self.sig_sqd` (variance of the data) are properly set
            within the class by the user to compute the likelihood.
        """
        # Set up of method
        self.betas = betas
        self.GP_Processing()
        results = self.equation()

        # Calculate neg log likelihood
        error = self.data - results
        ln_variance = betas[-1]
        neg_log_likelihood = 0.5 * jnp.log(2 * jnp.pi * jnp.exp(ln_variance)) + (error ** 2 / (2 * jnp.exp(ln_variance)))
        neg_log_prior = -jnp.log(jax.scipy.stats.multivariate_normal.pdf(betas[:-1], jnp.zeros((len(self.betas) - 1)), 1000*jnp.eye((len(self.betas) - 1))))
        # Calculate variance prior (Will be used in future Calculations)
        # neg_log_ln_p_variance = -jnp.log(inverse_gamma_pdf(jnp.exp(ln_variance), alpha = self.sigma_alpha, beta = self.sigma_beta))
        return jnp.sum(neg_log_likelihood) + neg_log_prior # + neg_log_ln_p_variance
    
    def d_neg_log_likelihood_create(self):
        """
        Creates and stores the gradient function of the negative log likelihood with respect to the
        beta coefficients in the `d_neg_log_likelihood` attribute. This method uses the JAX `grad`
        function to automatically differentiate `neg_log_likelihood`.  The reason for this is for 
        more readable code later on as well as testing flexibility, though this could be done before 
        sampling explicitly.

        Side Effects:
            - Sets the `d_neg_log_likelihood` attribute to the gradient (derivative) function of
            `neg_log_likelihood`, allowing it to be called later to compute gradient values.

        Note:
            - This method must be called before using `d_neg_log_likelihood` to compute gradients.
            - The `neg_log_likelihood` method must be correctly defined and compatible with JAX's automatic
            differentiation, which includes ensuring that all operations within `neg_log_likelihood` are
            differentiable and supported by JAX.
        """
        self.d_neg_log_likelihood = grad(self.neg_log_likelihood)
            
    def HMC(self, epsilon, L, current_q, M, Cov_Matrix, key):
        """
        Performs one iteration of the Hamiltonian Monte Carlo (HMC) algorithm to sample from
        a probability distribution proportional to the exponential of the negative log likelihood
        of the model. This method updates positions and momenta using Hamiltonian dynamics.

        Args:
            epsilon (float): Step size for the leapfrog integrator.
            L (int): Number of leapfrog steps to perform in each iteration.
            current_q (jax.numpy.ndarray): Current position (parameter vector representing betas of all GPs).
            M (jax.numpy.ndarray): Mass matrix, typically set to the identity matrix for inital sampling.
            Cov_Matrix (jax.numpy.ndarray): Covariance matrix (inverse of M) used to scale the kinetic energy.
            key (jax.random.PRNGKey): Pseudo-random number generator key.

        Returns:
            tuple: Contains the following elements:
                - new_q (jax.numpy.ndarray): The new position (parameters) after one HMC iteration. Will be current_q if not accepted.
                - accept (bool): Boolean indicating whether the new state was accepted based on the Metropolis-Hastings algorithm.
                - new_neg_log_likelihood (float): The negative log likelihood evaluated at the new position, providing a measure of the fit or suitability of the new parameters.
                - updated_key (jax.random.PRNGKey): The updated PRNG key after random operations, necessary for subsequent random operations to maintain randomness properties.


        Side Effects:
            - Updates the pseudo-random number generator key by splitting it for use in stochastic steps.

        Note:
            - The `grad_U` function refers to the gradient of the `neg_log_likelihood` method and must be
            created and stored in `d_neg_log_likelihood` before calling this method.
            - This method assumes that all necessary mathematical operations within are supported by JAX
            and that `M` and `Cov_Matrix` are appropriately defined for the problem at hand.
        """
        # Reassign for brevity in coding
        U = self.neg_log_likelihood
        grad_U = self.d_neg_log_likelihood

        # Random Momentum Sampling
        key, subkey = random.split(key)
        mean = jnp.zeros(len(M))
        p = random.multivariate_normal(subkey, mean, self.M)
        current_p = p

        ### Begin Leapfrog Integration
        # Make half step for momentum at the beginning
        p = p - epsilon * grad_U(current_q) / 2

        def loop_body(i, val):
            q, p = val
            q = q + epsilon * (Cov_Matrix @ p.reshape(-1, 1)).flatten()
            p_update = epsilon * grad_U(q)
            last_iter_factor = 1 - (i == L - 1)
            p = p - last_iter_factor * p_update
            return (q, p)

        q, p = fori_loop(0, L, loop_body, (current_q, p))

        # Make half step for momentum at the end
        p = p - epsilon * grad_U(q) / 2
        ### End Leapfrog Integration

        # Metropolis Hastings Criteria Evaluation
        # Negate momentum for detail balance
        p = -p

        current_U = U(current_q)
        current_K = sum(current_p @ Cov_Matrix @ current_p.reshape(-1, 1)) / 2
        proposed_U = U(q)
        proposed_K = sum(p @ Cov_Matrix @ p.reshape(-1, 1)) / 2

        accept_prob = jnp.exp(current_U - proposed_U + current_K - proposed_K)

        # If statement of Metropolis Hastings Criteria in JAX for optimized performance
        def true_branch(_):
            return q, True

        def false_branch(_):
            return current_q, False

        final, accept = cond(random.uniform(subkey) < accept_prob, true_branch, false_branch, None)

        return final, accept, U(final), key
    
    def create_jit_HMC(self):
        """
        Compiles the Hamiltonian Monte Carlo (HMC) method using JAX's Just-In-Time (JIT) compilation.
        This process optimizes the HMC method for faster execution by compiling it to machine code
        tailored to the specific hardware it will run on. The compiled function is stored in the
        `jit_HMC` attribute of the class.  The reason it is done in this fashion is for a) code
        brevity later on and b) the fact that JIT needs to occur post the user giving their model
        so this automates this process.

        Side Effects:
            - Sets the `jit_HMC` attribute to a JIT-compiled version of the `HMC` method. This allows
            the HMC method to execute more efficiently by reducing Python's overhead and optimizing
            execution at the hardware level.

        Usage:
            After invoking this method, `jit_HMC` can be used in place of `HMC` to perform Hamiltonian
            Monte Carlo sampling with significantly improved performance, especially beneficial
            in scenarios involving large datasets/complex models and where `HMC` is called a significant
            number of times (which is most models).

         Note:
            - The `create_jit_HMC` method should be called before using `jit_HMC` for the first time to ensure
            that the JIT compilation is completed.
        """
        self.jit_HMC = jit(self.HMC)
    
    def leapfrog(self, theta, r, grad, epsilon, f, Cov_Matrix):
        """ 
        This function is a part of this GitHub Repo: https://github.com/mfouesneau/NUTS/tree/master
        Perfom a leapfrog jump in the Hamiltonian space
        INPUTS
        ------
        theta: ndarray[float, ndim=1]
            initial parameter position

        r: ndarray[float, ndim=1]
            initial momentum

        grad: float
            initial gradient value

        epsilon: float
            step size

        f: callable
            it should return the log probability and gradient evaluated at theta
            logp, grad = f(theta)

        OUTPUTS
        -------
        thetaprime: ndarray[float, ndim=1]
            new parameter position
        rprime: ndarray[float, ndim=1]
            new momentum
        gradprime: float
            new gradient
        logpprime: float
            new lnp
        """
        # make half step in r
        rprime = r + 0.5 * epsilon * grad
        # make new step in theta
        thetaprime = theta + epsilon * (Cov_Matrix @ rprime.reshape(-1, 1)).flatten()
        #compute new gradient
        logpprime, gradprime = f(thetaprime)
        # make half step in r again
        rprime = rprime + 0.5 * epsilon * gradprime
        return thetaprime, rprime, gradprime, logpprime

    def find_reasonable_epsilon(self, theta0, key):
        """ 
        This function is a part of this GitHub Repo: https://github.com/mfouesneau/NUTS/tree/master
        Heuristic for choosing an initial value of epsilon.
        Algorithm 4 from original paper 
        """
        def f(theta):
            return self.neg_log_likelihood(theta)*-1, self.d_neg_log_likelihood(theta)*-1
        
        logp0, grad0 = f(theta0)
        epsilon = 1.
        mean = jnp.zeros(len(self.M))
        key, subkey = random.split(key)
        r0 = random.multivariate_normal(key, mean, self.M)

        # Figure out what direction we should be moving epsilon.
        _, rprime, gradprime, logpprime = self.leapfrog(theta0, r0, grad0, epsilon, f, self.Cov_Matrix)

        def cond_fun(k):
            _, _, gradprime, logpprime = self.leapfrog(theta0, r0, grad0, epsilon * k, f, self.Cov_Matrix)
            is_inf = jnp.isinf(logpprime) | jnp.isinf(gradprime).any()
            return is_inf

        def body_fun(k):
            k *= 0.5
            _, _, gradprime, logpprime = self.leapfrog(theta0, r0, grad0, epsilon * k, f, self.Cov_Matrix)
            is_inf = jnp.isinf(logpprime) | jnp.isinf(gradprime).any()
            return k # lax.select(is_inf, k * 0.5, k) # cond(is_inf, lambda _: k * 0.5, lambda _: k, None)

        k = 1.
        k = while_loop(cond_fun, body_fun, k)

        epsilon = 0.5 * k * epsilon

        # The goal is to find the current acceptance probability and then move
        # epsilon in a direction until it crosses the 50% acceptance threshold
        # via doubling of epsilon
        logacceptprob = logpprime-logp0-0.5*((rprime @ rprime)-(r0 @ r0))
        a = lax.select(logacceptprob > jnp.log(0.5), 1., -1.)
        # Keep moving epsilon in that direction until acceptprob crosses 0.5.

        def cond_fun(carry):
            epsilon, logacceptprob = carry
            return a * logacceptprob > -a * jnp.log(2.)

        def body_fun(carry):
            epsilon, logacceptprob = carry
            epsilon = epsilon * (2. ** a)
            _, rprime, _, logpprime = self.leapfrog(theta0, r0, grad0, epsilon, f, self.Cov_Matrix)
            logacceptprob = logpprime - logp0 - 0.5 * ((rprime @ rprime) - (r0 @ r0))
            return epsilon, logacceptprob

        # epsilon = 1.
        epsilon, logacceptprob = lax.while_loop(cond_fun, body_fun, (epsilon, logacceptprob))

        return epsilon
    
    def create_jit_find_reasonable_epsilon(self):
        """
        Compiles the `find_reasonable_epsilon` function using JAX's Just-In-Time (JIT) compilation to
        enhance its performance. This method optimizes the function for faster execution by compiling it
        to machine-specific code, significantly reducing runtime. The reason it is done in this fashion 
        is for a) code brevity later on and b) the fact that JIT needs to occur post the user giving their 
        model so this automates this process.

        Side Effects:
            - Sets the `jit_find_reasonable_epsilon` attribute to a JIT-compiled version of the
            `find_reasonable_epsilon` method. This allows the method to execute more efficiently by
            reducing Python overhead and leveraging optimized low-level operations.

        Usage:
            This method should be called before any intensive sampling procedures where `find_reasonable_epsilon`
            is expected to be called, to minimize computational overhead and improve overall
            performance of the sampling process.

        Note:
            - JIT compilation happens the first time the JIT-compiled function is called, not when
            `create_jit_find_reasonable_epsilon` is executed.
        """
        self.jit_find_reasonable_epsilon = jit(self.find_reasonable_epsilon)
    
    def full_sample(self, draws):
        """
        Conducts a full HMC sampling, creating multiple draws from the posterior distribution
        of the model parameters. This function initializes and updates sampling parameters, executes
        Hamiltonian Monte Carlo using a JIT-compiled version of the sampling routine, and
        dynamically adjusts the step size based on acceptance rates.

        Args:
            draws (int): Number of samples to draw from the posterior distribution.

        Returns:
            tuple: A tuple containing:
                - samples (jax.numpy.ndarray): An array of sampled parameter vectors.
                - acceptance_array (jax.numpy.ndarray): An array indicating whether each sample was accepted.
                - neg_log_likelihood_array (jax.numpy.ndarray): An array of negative log likelihood values for each sample.

        Procedure:
            1. Initialize the covariance and mass matrices.
            2. Create a JIT-compiled Hamiltonian Monte Carlo (HMC) sampler.
            3. Iteratively sample using HMC, adjusting the leapfrog step size (`epsilon`) based on acceptance rates.
            4. Adjust the mass matrix based on warm up.

        Notes:
            - This method assumes `find_reasonable_epsilon` and `create_jit_HMC` are available to set reasonable
            values for `epsilon` and to compile the HMC sampling method, respectively.
            - The dynamic adjustment of `epsilon` aims to optimize the sampling efficiency by tuning the
            acceptance rate to a desirable range.
            - The mass matrix (`M`) and the covariance matrix (`Cov_Matrix`) are recalibrated during the sampling
            based on the properties of the collected samples to enhance sampling accuracy and efficiency.
            - The function also monitors for stagnation in parameter space and makes significant adjustments to 
            `epsilon` and recalibrates `M` and `Cov_Matrix` as needed.

        Example Usage:
            # Assuming an instance of the model `model_instance` has been created:
            samples, accepts, nlls = model_instance.full_sample(1000)
            print("Sampled Parameters:", samples)
            print("Acceptance Rates:", accepts)
            print("Negative Log Likelihoods:", nlls)
        """
        # Initialize parameters for new interaction matrix
        self.Cov_Matrix = jnp.eye(len(self.GP)*(len(self.discmtx)+1) +1)
        self.M = jnp.linalg.inv(self.Cov_Matrix)
        neg_log_likelihood_array = jnp.zeros(draws+1, dtype=float)
        acceptance_array = jnp.zeros(draws+1, dtype=bool)
        samples = jnp.ones((draws+1, len(self.GP)*(len(self.discmtx)+1)+1)) # Starting point always all betas = 1

        # Create relevant functions
        self.d_neg_log_likelihood_create()
        self.create_jit_find_reasonable_epsilon()
        self.create_jit_HMC()

        # Create Initial Epsilon Estimate
        self.epsilon = self.jit_find_reasonable_epsilon(samples[0], self.key)

        # Loop for HMC Sampling        
        for i in range(draws):
            # Print iteration in loop
            print(i)
            # Actual HMC Sampling
            sample, accept, neg_log_likelihood_sample, self.key = self.jit_HMC(epsilon = self.epsilon,
                                                                               L = 20, 
                                                                               current_q = samples[i],
                                                                               M = self.M,
                                                                               Cov_Matrix = self.Cov_Matrix, 
                                                                               key = self.key)
            
            # Save HMC sampling results
            samples = samples.at[i+1].set(sample)
            acceptance_array = acceptance_array.at[i+1].set(accept)
            neg_log_likelihood_array = neg_log_likelihood_array.at[i+1].set(neg_log_likelihood_sample)  

            # To make epsilon adaptive, modify based on acceptance rate (ideal 65% per paper)
            if (i+1) % 50 == 0:
                if sum(acceptance_array[i-50:i]) < 15:
                    self.epsilon = self.epsilon*0.5
                    print('Massive Decrease to Epsilon')
                if sum(acceptance_array[i-50:i]) < 30 and sum(acceptance_array[i-50:i]) >= 15:
                    self.epsilon = self.epsilon*0.8
                    print('Decreased Epsilon')
                if sum(acceptance_array[i-50:i]) > 30 and sum(acceptance_array[i-50:i]) <= 45:
                    self.epsilon = self.epsilon*1.2
                    print('Increased Epsilon')
                if sum(acceptance_array[i-50:i]) > 45:
                    self.epsilon = self.epsilon*1.5
                    print('Massive Increase to Epsilon')

            # Update Mass Matrix after warmup (NOTE: breaks detail balance)
            if (i+1) in [500] and len(jnp.unique(samples[i-100:i],axis=0)) >= 5:
                print('M Update')
                # Take the last 100 values of the vector and create Covariance and Mass Matrixes
                last_100_values = jnp.unique(samples[i-100:i],axis=0)
                cov_matrix = jnp.cov(last_100_values, rowvar=False)
                self.Cov_Matrix = cov_matrix.diagonal()*jnp.identity(len(cov_matrix))
                self.M = jnp.linalg.inv(cov_matrix)*jnp.identity(len(cov_matrix))
                print(self.M)

                # Update epsilon
                theta = samples[i]
                self.epsilon = self.jit_find_reasonable_epsilon(theta, self.key)
        
        return samples, acceptance_array, neg_log_likelihood_array
    
    def full_routine(self, draws, tolerance, way3 = 0):
        """
        Creates the interaction matrixes and compares the models against eachother.  Taken from methodology
        with a singular GP.
        """
        # relats_in = jnp.array([])
        
        def perms(x):
            """Python equivalent of MATLAB perms."""
            a = jnp.array(jnp.vstack(list(itertools.permutations(x)))[::-1])
            return a

        # 'n' is the number of datapoints whereas 'm' is the number of inputs
        n, m = jnp.shape(self.inputs)
        mrel = n
        damtx = jnp.array([])
        evs = jnp.array([])

        # Conversion of Lines 79-100 of emulator_Xin.m
        # if jnp.logical_not(all([isinstance(index, int) for index in relats_in])):  # checks if relats is an array
        #     if jnp.any(relats_in):
        #         relats = jnp.zeros((sum(jnp.logical_not(relats_in)), m))
        #         ind = 1
        #         for i in range(0, m):
        #             if jnp.logical_not(relats_in[i]):
        #                 relats[ind][i] = 1
        #                 ind = ind + 1
        #         ind_in = m + 1
        #         for i in range(0, m - 1):
        #             for j in range(i + 1, m):
        #                 if jnp.logical_not(relats_in[ind_in]):
        #                     relats[ind][i] = 1
        #                     relats[ind][j] = 1
        #                     ind = ind + 1
        #             ind_in = ind_in + 1
        #     mrel = sum(np.logical_not(relats_in)).all()
        # else:
        #     mrel = sum(np.logical_not(relats_in))
        mrel = 0
        # End conversion

        # 'ind' is an integer which controls the development of new terms
        ind = 1
        greater = 0
        finished = 0
        X = []
        killset = []
        killtest = []
        if m == 1:
            sett = 1
        elif way3:
            sett = 3
        else:
            sett = 2

        while True:
            # first we have to come up with all combinations of 'm' integers that
            # sums up to ind
            indvec = np.zeros((m))
            summ = ind

            while summ:
                for j in range(0,sett):
                    indvec[j] = indvec[j] + 1
                    summ = summ - 1
                    if summ == 0:
                        break

            while 1:
                vecs = jnp.unique(perms(indvec),axis=0)
                if ind > 1:
                    mvec, nvec = np.shape(vecs)
                else:
                    mvec = jnp.shape(vecs)[0]
                    nvec = 1
                killvecs = []
                if mrel != 0:
                    for j in range(1, mvec):
                        testvec = jnp.divide(vecs[j, :], vecs[j, :])
                        testvec[jitnp.isnan(testvec)] = 0
                        for k in range(1, mrel):
                            if sum(testvec == relats[k, :]) == m:
                                killvecs.append(j)
                                break
                    nuvecs = jnp.zeros(mvec - jnp.size(killvecs), m)
                    vecind = 1
                    for j in range(1, mvec):
                        if not (j == killvecs):
                            nuvecs[vecind, :] = vecs[j, :]
                            vecind = vecind + 1

                    vecs = nuvecs
                if ind > 1:
                    vm, vn = jnp.shape(vecs)
                else:
                    vm = jnp.shape(vecs)[0]
                    vn = 1
                if jnp.size(damtx) == 0:
                    damtx = vecs
                else:
                    damtx = jnp.append(damtx, vecs, axis=0)
                [dam,null] = jnp.shape(damtx)
                self.discmtx = damtx.astype(int)
                print(damtx)

                beters, null, neg_log_likelihood = self.full_sample(draws)

                ev = (2*len(self.discmtx) + 1) * jnp.log(n) - 2 * jnp.max(neg_log_likelihood*-1)

                # if aic:
                #     ev = ev + (2 - np.log(n)) * (dam + 1)

                # This is the means and bounds fo the model just sampled
                # betavs = np.abs(np.mean(beters[int(np.ceil((draws / 2)+1)):draws, (dam - vm + 1):dam+1], axis=0))
                # betavs2 = np.divide(np.std(np.array(beters[int(np.ceil(draws/2)+1):draws, dam-vm+1:dam+1]), axis=0), \
                #     np.abs(np.mean(beters[int(np.ceil(draws / 2)):draws, dam-vm+1:dam+2], axis=0)))
                #     # betavs2 error in std deviation formatting
                # betavs3 = np.array(range(dam-vm+2, dam+2))
                # betavs = np.transpose(np.array([betavs,betavs2, betavs3]))
                # if np.shape(betavs)[1] > 0:
                #     sortInds = np.argsort(betavs[:, 0])
                #     betavs = betavs[sortInds]

                # killset = []
                evmin = ev


                # This is for deletion of terms
                # for i in range(0, vm):


                #     # if betavs[i, 1] > threshstdb or betavs[i, 1] > threshstda and betavs[i, 0] < threshav * \
                #     #     np.mean(np.abs(np.mean(beters[int(np.ceil(draws/2 +1)):draws, 0]))):
                #     if betavs[i, 1] > threshstdb or betavs[i, 1] > threshstda and betavs[i, 0] < threshav * \
                #         np.mean(np.abs(np.mean(beters[int(np.ceil(draws/2)):draws, 0]))):  # index to 'beters' \
                #         # adjusted for matlab to python [JPK DEV v3.1.0 20240129]

                #         killtest = np.append(killset, (betavs[i, 2] - 1))
                #         if killtest.size > 1:
                #             killtest[::-1].sort()  # max to min so damtx_test rows get deleted in order of end to start
                #         damtx_test = damtx
                #         for k in range(0, np.size(killtest)):
                #             damtx_test = np.delete(damtx_test, int(np.array(killtest[k])-1), 0)
                #         damtest, null = np.shape(damtx_test)

                #         [betertest, null, null, null, Xtest, evtest] = hmc(inputs_np, data, phis, X, damtx_test, a, b,
                #                                                              atau, btau, draws)
                #         if aic:
                #             evtest = evtest + (2 - np.log(n))*(damtest+1)
                #         if evtest < evmin:
                #             killset = killtest
                #             evmin = evtest
                #             xers = Xtest
                #             beters = betertest
                # for k in range(0, np.size(killset)):
                #     damtx = np.delete(damtx, int(np.array(killset[k]) - 1), 0)

                ev = jnp.min(evmin)
                # X = xers

                # print(ev)
                # print(evmin)
                print([ind, ev])
                if jnp.size(evs) > 0:
                    if ev < jnp.min(evs):

                        betas = beters
                        mtx = damtx
                        greater = 1
                        evs = jnp.append(evs, ev)

                    elif greater < tolerance:
                        greater = greater + 1
                        evs = jnp.append(evs, ev)
                    else:
                        finished = 1
                        evs = jnp.append(evs, ev)

                        break
                else:
                    greater = greater + 1
                    betas = beters
                    mtx = damtx
                    evs = jnp.append(evs, ev)
                if m == 1:
                    break
                elif way3:
                    if indvec[1] > indvec[2]:
                        indvec[0] = indvec[0] + 1
                        indvec[1] = indvec[1] - 1
                    elif indvec[2]:
                        indvec[1] = indvec[1] + 1
                        indvec[2] = indvec[2] - 1
                        if indvec[1] > indvec[0]:
                            indvec[0] = indvec[0] + 1
                            indvec[1] = indvec[1] - 1
                    else:
                        break
                elif indvec[1]:
                    indvec[0] = indvec[0] + 1
                    indvec[1] = indvec[1] - 1
                else:
                    break



            if finished != 0:
                break

            ind = ind + 1

            if ind > len(self.phis):
                break

        # # Implementation of 'gimme' feature
        # if gimmie:
        #     betas = beters
        #     mtx = damtx

        self.betas = betas
        self.mtx = mtx
        self.evs = evs

        return betas, mtx, evs
    
    # def evaluate(self):
        
    #     results = self.equation()

    #     return results
    
    def inputs_to_phind(self, inputs, phis):
        """
        Twice normalize the inputs to index the spline coefficients.

        Inputs:
            - inputs == normalized inputs as numpy array (i.e., self.inputs.np)
            - phis   == spline coefficients

        Output (and appended class attributes):
            - phind == index to spline coefficients
            - xsm   ==
        """

        L_phis = len(phis[0][0])  # = 499, length of coeff. in basis funtions
        phind = np.array(np.ceil(inputs * L_phis), dtype=int)  # 0-1 normalization to 0-499 normalization

        if phind.ndim == 1:  # if phind.shape == (number,) != (number,1), then add new axis to match indexing format
            phind = phind[:, np.newaxis]

        set = (phind == 0)  # set = 1 if phind = 0, otherwise set = 0
        phind = phind + set  # makes sense assuming L_phis > M

        r = 1 / L_phis  # interval of when basis function changes (i.e., when next cubic function defines spline)
        xmin = (phind - 1) * r
        X = (inputs - xmin) / r  # twice normalized inputs (0-1 first then to size of phis second)

        self.phind = phind - 1  # adjust MATLAB indexing to Python indexing after twice normalization
        self.xsm = L_phis * inputs - phind

        return self.phind, self.xsm
    
    def evaluate(self, inputs, GP_number, **kwargs):
        """
        Evaluate the inputs and output the predicted values of corresponding data. Optionally, calculate bounds.

        Input:
            inputs == matrix of independent (or non-linearly dependent) 'x' variables for evaluating f(x1, ..., xM)
            GP_number == the GP you would like to evaluate from the training

        Keyword Inputs:
            draws        == number of beta terms used                              == 100 (default)
            nform        == logical to automatically normalize and format 'inputs' == 1 (default)
            ReturnBounds == logical to return confidence bounds as second output   == 0 (default)
        """

        # Default keywords:
        kwargs_all = {'draws': 100, 'ReturnBounds': 0}

        # Update keywords based on user-input:
        for kwarg in kwargs.keys():
            if kwarg not in kwargs_all.keys():
                raise ValueError(f"Unexpected keyword argument: {kwarg}")
            else:
                kwargs_all[kwarg] = kwargs.get(kwarg, kwargs_all.get(kwarg))

        # Define local variables:
        # for kwarg in kwargs_all.keys():
        #     locals()[kwarg] = kwargs_all.get(kwarg) # defines each keyword (including defaults) as a local variable
        draws = kwargs_all.get('draws')
        # nform = kwargs_all.get('nform')
        ReturnBounds = kwargs_all.get('ReturnBounds')

        # # Process nform:
        # if isinstance(nform, str):
        #     if nform.lower() in ['yes','y','on','auto','default','true']:
        #         nform = 1
        #     elif nform.lower() in ['no','n','off','false']:
        #         nform = 0
        # else:
        #     if nform not in [0,1]:
        #         raise ValueError("Keyword argument 'nform' must a logical 1 (default) or 0.")

        # Automatically normalize and format inputs:
        # def auto_nform(inputs):

        #     # Convert 'inputs' to numpy if pandas:
        #     if any(isinstance(inputs, type) for type in (pd.DataFrame, pd.Series)):
        #         inputs = inputs.to_numpy()
        #         warnings.warn("'inputs' was auto-converted to numpy. Convert manually for assured accuracy.", UserWarning)

        #     # Normalize 'inputs' and convert to proper format for FoKL:
        #     inputs = np.array(inputs) # attempts to handle lists or any other format (i.e., not pandas)
        #     # . . . inputs = {ndarray: (N, M)} = {ndarray: (datapoints, input variables)} =
        #     # . . . . . . array([[x1(t1),x2(t1),...,xM(t1)],[x1(t2),x2(t2),...,xM(t2)],...,[x1(tN),x2(tN),...,xM(tN)]])
        #     inputs = np.squeeze(inputs) # removes axes with 1D for cases like (N x 1 x M) --> (N x M)
        #     if inputs.ndim == 1:  # if inputs.shape == (number,) != (number,1), then add new axis to match FoKL format
        #         inputs = inputs[:, np.newaxis]
        #     N = inputs.shape[0]
        #     M = inputs.shape[1]
        #     if M > N: # if more "input variables" than "datapoints", assume user is using transpose of proper format above
        #         inputs = inputs.transpose()
        #         warnings.warn("'inputs' was transposed. Ignore if more datapoints than input variables.", category=UserWarning)
        #         N_old = N
        #         N = M # number of datapoints (i.e., timestamps)
        #         M = N_old # number of input variables
        #     minmax = self.normalize
        #     inputs_min = np.array([minmax[ii][0] for ii in range(len(minmax))])
        #     inputs_max = np.array([minmax[ii][1] for ii in range(len(minmax))])
        #     inputs = (inputs - inputs_min) / (inputs_max - inputs_min)

        #     nformputs = inputs.tolist() # convert to list, which is proper format for FoKL, like:
        #     # . . . {list: N} = [[x1(t1),x2(t1),...,xM(t1)],[x1(t2),x2(t2),...,xM(t2)],...,[x1(tN),x2(tN),...,xM(tN)]]

        #     return nformputs

        # if nform:
        #     normputs = auto_nform(inputs)
        # else: # assume provided inputs are already normalized and formatted
        normputs = inputs

        betas = self.betas[-draws:, 0:-1] # Get all the betas but the sigmas
        num_functions = len(self.GP)
        num_betas = int(len(betas[0])/num_functions)
        betas_list = betas[:, GP_number*num_betas:(GP_number+1)*num_betas]

        betas = betas_list
        mtx = self.mtx
        phis = self.phis

        m, mbets = np.shape(betas)  # Size of betas
        n, mputs = np.shape(normputs)  # Size of normalized inputs

        setnos_p = np.random.randint(m, size=(1, draws))  # Random draws  from integer distribution
        i = 1
        while i == 1:
            setnos = np.unique(setnos_p)

            if np.size(setnos) == np.size(setnos_p):
                i = 0
            else:
                setnos_p = np.append(setnos, np.random.randint(m, size=(1, draws - np.shape(setnos)[0])))

        X = np.zeros((n, mbets))
        normputs = np.asarray(normputs)

        phind, xsm = self.inputs_to_phind(normputs, phis)
        for i in range(n):
            for j in range(1, mbets):
                phi = 1
                for k in range(mputs):
                    num = mtx[j - 1, k]
                    if num > 0:
                        nid = int(num - 1)
                        phi = phi * (phis[nid][0][phind[i, k]] + phis[nid][1][phind[i, k]] * xsm[i, k] + \
                            phis[nid][2][phind[i, k]] * xsm[i, k] ** 2 + phis[nid][3][phind[i, k]] * xsm[i, k] ** 3)

                X[i, j] = phi

        X[:, 0] = np.ones((n,))
        modells = np.zeros((n, draws))  # note n == np.shape(data)[0] if data != 'ignore'
        for i in range(draws):
            modells[:, i] = np.matmul(X, betas[setnos[i], :])
        meen = np.mean(modells, 1)

        if ReturnBounds:
            bounds = np.zeros((n, 2))  # note n == np.shape(data)[0] if data != 'ignore'
            cut = int(np.floor(draws * .025))
            for i in range(n):  # note n == np.shape(data)[0] if data != 'ignore'
                drawset = np.sort(modells[i, :])
                bounds[i, 0] = drawset[cut]
                bounds[i, 1] = drawset[draws - cut]
            return meen, bounds
        else:
            return meen
