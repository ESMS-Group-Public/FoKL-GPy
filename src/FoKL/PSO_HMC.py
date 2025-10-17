


import numpy as np
from FoKL import FoKLRoutines
import os

# num_cpus = os.cpu_count()
# jax_cpus = num_cpus - 2
# os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={jax_cpus}"

import sys

if "jax" in sys.modules:
    import jax
    if not jax.config.x64_enabled:
        raise RuntimeError(
            "JAX is already imported and 64-bit mode is disabled.\n"
            "Please restart the Python kernel and enable it when importing JAX:\n"
            "    import jax\n"
            "    jax.config.update('jax_enable_x64', True)\n"
            "Alternatively import this module first and JAX will be automatically configured correctly."
        )
else:
    import jax
    jax.config.update("jax_enable_x64", True)


from jax import lax, vmap
import jax.numpy as jnp
from jax import jit, device_put
from jax.lib import xla_bridge
from jax.tree_util import Partial
from jax import random
from jax.lax import fori_loop, cond, while_loop
#import Embedded_GP as FoKL_Embedded_GPs
#from GITTloglik_radial_jx6MIG import GITTloglik_radial_jx
from datetime import datetime

from evosax.algorithms.population_based import PSO

cpu = jax.devices("cpu")[0]
gpu = jax.devices("gpu")[0]
    
# jax.config.update("jax_enable_x64", True)
# print(jax.devices())

# # Get the start time
# start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# print("Start Time:", start_time)

# # Define the number of zeros you want to add
# num_zeros = 15 #os.getenv("SLURM_ARRAY_TASK_ID")  # Based on number from slurm file

# # mtx_D = jnp.arange(1, jnp.int32(num_zeros)+1)
# mtx_D = jnp.arange(num_zeros) #array([2.0,3.0,4.0,5.0,6.0,7.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0])
# mtx_D = mtx_D + 1

# modparams_rad = np.loadtxt('./modparams_rad_accel4.txt')
# data = np.loadtxt('data_accel4.txt')
# incs_reg_NCM_charg = np.loadtxt('incs_reg_NCM_charg_accel4.txt')[::10] # Gives every 10th
# # mtx_D = np.loadtxt('1_GITT_Sampling/HMC_Sampling/data/mtx_D.txt', delimiter=',')
# mtx_D = mtx_D.astype(jnp.int32)
# # betas_D_NCM = np.loadtxt('1_GITT_Sampling/HMC_Sampling/data/betas_D_NCM.txt', delimiter=',')

# #phis = jnp.array(FoKL_Embedded_GPs.getKernels.sp500())
# dummy_model = FoKLRoutines.FoKL()
# phis = dummy_model.phis

# # At the beginning of your script, after loading data:
# # modparams_rad = jax.device_put(jnp.array(np.loadtxt('2_HMC_DS_Repo/data/modparams_rad.txt')))
# # data = jax.device_put(jnp.array(np.loadtxt('2_HMC_DS_Repo/data/data.txt', delimiter=',')))
# # incs_reg_NCM_charg = jax.device_put(jnp.array(np.loadtxt('2_HMC_DS_Repo/data/incs_reg_NCM_charg.txt', delimiter=',')[::10]))
# # mtx_D = jax.device_put(mtx_D)
# # phis = jax.device_put(jnp.array(FoKL_Embedded_GPs.getKernels.sp500()))

# modparams_rad_tp = tuple(modparams_rad.flat)

# GITT_NCM = GITTloglik_radial_jx(jnp.array(data), jnp.array((-1e-3)*incs_reg_NCM_charg), jnp.array(phis), modparams_rad_tp, jax_cpus)

# GITT_NCM.modelset(jnp.array(mtx_D))

# GITT_func = jit(GITT_NCM.lpdf_func)

# GITT_fdgrad = jit(GITT_NCM.lpdf_gd_fd)

# GITT_sigfunc = jit(GITT_NCM.sig_func)

#@Partial(jit, static_argnums=(1,2,))
def sample_from_prior(key, meanstds, n_particles):
    return random.normal(key, shape=(n_particles, meanstds.shape[0])) * meanstds[:,1].reshape(-1) + meanstds[:,0].reshape(-1)

#samples = sample_from_prior(key, meanstds, n_particles)

#samples32 = device_put(samples.astype(jnp.float32), device=gpu)
#sig32 = device_put(jnp.sqrt(GITT_NCM.sigb / (1.0 + GITT_NCM.siga)), device=gpu)

#GITT_func_vec = jax.vmap(lambda x: GITT_func(x, sig32))

def run_pso_with_prior(key, obj_fn, meanstds, n_particles, sigma, n_iters, top_k):
    
    key_init, key_sample = random.split(key)
    
    x_init = sample_from_prior(key_init, meanstds, n_particles)
    
    #jax.debug.breakpoint()
    
    x_init = device_put(x_init.astype(jnp.float32), device=gpu)
    sigma = device_put(sigma.astype(jnp.float32), device=gpu)
    
    obj_vmp = jax.vmap(obj_fn, in_axes=(0, None))

    # Initialize PSO
    _, dim = x_init.shape
    pso = PSO(population_size=n_particles, solution=jnp.zeros(dim, dtype=jnp.float32))
    params = pso.default_params
    history = jnp.zeros((n_particles, 50, dim))
    
    init_fitness = obj_vmp(x_init, sigma)
    state = pso.init(key_init, x_init, init_fitness, params)

    # Main loop
    for i in range(n_iters):
        key_it = key_sample#random.fold_in(key_sample, i)
        x, state = pso.ask(key_it, state, params)
        fitness = obj_vmp(x, sigma)
        state, _ = pso.tell(key_it, x, fitness, state, params)
        if n_iters - i <= 50:
            jax.debug.breakpoint()
            history = history.at[:,50 + i - n_iters,:].set(x)
         
        if i % 10 == 0:
            best_idx = jnp.argmin(fitness)
            print(f"Iter {i:03d} | Best fitness: {fitness[best_idx]:.6f}")

    # Sort final population by fitness
    final_fitness = obj_fn(state.population, sigma)
    sorted_idx = jnp.argsort(final_fitness)
    top_params = state.population[sorted_idx[:top_k]]
    top_vals = final_fitness[sorted_idx[:top_k]]
    top_hist = history[sorted_idx[:top_k],:,:]
    return top_params, top_vals, top_hist

# top_particles, top_vals = run_pso_with_prior(
#     GITT_func_vec,
#     samples32,
#     n_iters=200,
#     key=key,
#     top_k=5
# )

# print("Top particles:")
# print(top_particles)

#nlls = vmap(lambda Db: GITT_func(Db, sig32))(samples32)
#start_sample = device_put(top_particles[jnp.argmin(top_vals)].astype(jnp.float64), device=cpu)


# Initialize sigma
#start_sig = GITT_sigfunc(jnp.array(start_sample), key)

# Log Likelihood function
#lpdf = GITT_func(jnp.array(start_sample), start_sig)

# Gradient with respect to the beta values
#lpdf_gd = GITT_fdgrad(jnp.array(start_sample), lpdf, start_sig)

@jit
def HMC(epsilon, L, current_q, sig, M, Cov_Matrix, U, grad_U, key):
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

    # Random Momentum Sampling
    key, subkey = random.split(key)
    mean = jnp.zeros(len(M))
    p = random.multivariate_normal(subkey, mean, M)
    current_p = p

    ### Begin Leapfrog Integration
    # Make half step for momentum at the beginning
    p = p - epsilon * grad_U(current_q, U(current_q, sig), sig) / 2

    def loop_body(i, val):
        q, p, sig = val
        q = q + epsilon * (Cov_Matrix @ p.reshape(-1, 1)).flatten()
        p_update = epsilon * grad_U(q, U(q, sig), sig)
        # jax.debug.print("The value of neg log likelihood is: {}", U(q))
        # jax.debug.print("The value of q is: {}", q)
        last_iter_factor = 1 - (i == L - 1)
        p = p - last_iter_factor * p_update
        return (q, p, sig)

    q, p, _ = fori_loop(0, L, loop_body, (current_q, p, sig))

    # Make half step for momentum at the end
    p = p - epsilon * grad_U(q, U(q, sig), sig) / 2
    ### End Leapfrog Integration

    # Metropolis Hastings Criteria Evaluation
    # Negate momentum for detail balance
    p = -p

    current_U = U(current_q, sig)
    current_K = sum(current_p @ Cov_Matrix @ current_p.reshape(-1, 1)) / 2
    proposed_U = U(q, sig)
    proposed_K = sum(p @ Cov_Matrix @ p.reshape(-1, 1)) / 2

    accept_prob = jnp.exp(current_U - proposed_U + current_K - proposed_K)

    # If statement of Metropolis Hastings Criteria in JAX for optimized performance
    def true_branch(_):
        return q, True

    def false_branch(_):
        return current_q, False

    final, accept = cond(random.uniform(subkey) < accept_prob, true_branch, false_branch, None)

    return final, accept, U(final, sig), key

#jit_HMC = jit(HMC)

def leapfrog(theta, sig, r, grad, epsilon, f, Cov_Matrix):
    """ Perfom a leapfrog jump in the Hamiltonian space
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
    # Something is wrong here I think?? Theta prime is very large. Should M_inv be just M?
    thetaprime = theta + epsilon * (Cov_Matrix @ rprime.reshape(-1, 1)).flatten()
    #compute new gradient
    # Limitation on variance
    logpprime, gradprime = f(thetaprime, sig)
    # make half step in r again
    rprime = rprime + 0.5 * epsilon * gradprime
    return thetaprime, rprime, gradprime, logpprime

@jit
def find_reasonable_epsilon(theta0, sig, key, M, Cov_Matrix, U, grad_U):
    """ 
    Heuristic for choosing an initial value of epsilon.
    Algorithm 4 from original paper 
    """
    def f(theta, sig):
        neg_log_likelihood_temp = U(theta, sig)
        return -neg_log_likelihood_temp, -grad_U(theta, neg_log_likelihood_temp, sig)
    
    logp0, grad0 = f(theta0, sig)
    epsilon = 1.
    # Initial Momentum
    mean = jnp.zeros(len(M))
    key, subkey = random.split(key)
    r0 = random.multivariate_normal(key, mean, M)

    # Figure out what direction we should be moving epsilon.
    _, rprime, gradprime, logpprime = leapfrog(theta0, sig, r0, grad0, epsilon, f, Cov_Matrix)
    # brutal! This trick make sure the step is not huge leading to infinite
    # values of the likelihood. This could also help to make sure theta stays
    # within the prior domain (if any)
    def cond_fun(k):
        _, _, gradprime, logpprime = leapfrog(theta0, sig, r0, grad0, epsilon * k, f, Cov_Matrix)
        is_inf = jnp.isinf(logpprime) | jnp.isinf(gradprime).any()
        return is_inf

    def body_fun(k):
        k *= 0.5
        _, _, gradprime, logpprime = leapfrog(theta0, sig, r0, grad0, epsilon * k, f, Cov_Matrix)
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
        # jax.debug.print("Log Acceptance Probability: {}", logacceptprob )
        return a * logacceptprob > -a * jnp.log(2.)

    def body_fun(carry):
        epsilon, logacceptprob = carry
        epsilon = epsilon * (2. ** a)
        _, rprime, _, logpprime = leapfrog(theta0, sig, r0, grad0, epsilon, f, Cov_Matrix)
        logacceptprob = logpprime - logp0 - 0.5 * ((rprime @ rprime) - (r0 @ r0))
        return epsilon, logacceptprob

    # epsilon = 1.
    epsilon, logacceptprob = lax.while_loop(cond_fun, body_fun, (epsilon, logacceptprob))

    return epsilon

#jit_find_reasonable_epsilon = jit(find_reasonable_epsilon)

def full_sample(key, start_sample, start_sig, U, grad_U, sigfunc, Cov_Matrix, draws):
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
    #Cov_Matrix = jnp.eye((len(start_sample)))
    M = jnp.linalg.inv(Cov_Matrix)
    neg_log_likelihood_array = jnp.zeros(draws+1, dtype=float)
    acceptance_array = jnp.zeros(draws+1, dtype=bool)
    sig_array = jnp.zeros(draws+1, dtype=float)
    samples = jnp.ones((draws+1, (len(start_sample)))) # Starting point always all betas = 1
    # samples = jax.device_put(jnp.ones((draws+1, (len(start_sample)))))
    samples = samples.at[0].set(start_sample)
    sig = start_sig

    # Create relevant functions
    # self.d_neg_log_likelihood_create()
    # self.create_jit_find_reasonable_epsilon()
    # self.create_jit_HMC()

    # Create Initial Epsilon Estimate
    epsilon = find_reasonable_epsilon(samples[0], sig, key, M = M, Cov_Matrix = Cov_Matrix)
    
    print(epsilon)
    print(samples[0])

    # Loop for HMC Sampling
      
    for i in range(draws):
        # Print iteration in loop
        # print(i)
        # Actual HMC Sampling
        sample, accept, neg_log_likelihood_sample, key = HMC(epsilon = epsilon,
                                                                            L = 20, 
                                                                            current_q = samples[i],
                                                                            sig = sig,
                                                                            M = M,
                                                                            Cov_Matrix = Cov_Matrix,
                                                                            U = U,
                                                                            grad_U = grad_U,
                                                                            key = key
                                                                            )
        
        # Save HMC sampling results
        samples = samples.at[i+1].set(sample)
        acceptance_array = acceptance_array.at[i+1].set(accept)
        neg_log_likelihood_array = neg_log_likelihood_array.at[i+1].set(neg_log_likelihood_sample)
        
        #update sigma -- Metropolis-in-Gibbs
        sig = sigfunc(sample, key)
        sig_array = sig_array.at[i+1].set(sig)

        # To make epsilon adaptive, modify based on acceptance rate (ideal 65% per paper)
        if (i+1) % 50 == 0:
            if sum(acceptance_array[i-50:i]) < 15:
                epsilon = epsilon*0.5
                print('Massive Decrease to Epsilon')
            if sum(acceptance_array[i-50:i]) < 25 and sum(acceptance_array[i-50:i]) >= 15:
                epsilon = epsilon*0.8
                print('Decreased Epsilon')
            if sum(acceptance_array[i-50:i]) > 25 and sum(acceptance_array[i-50:i]) <= 45:
                epsilon = epsilon*1.2
                print('Increased Epsilon')
            if sum(acceptance_array[i-50:i]) > 45:
                epsilon = epsilon*1.5
                print('Massive Increase to Epsilon')

        # Update Mass Matrix after warmup (NOTE: breaks detail balance)
        if (i+1) % 100 == 0 and len(jnp.unique(samples[i-100:i],axis=0)) >= 5:
            print('M Update')
            # Take the last 100 values of the vector and create Covariance and Mass Matrixes
            last_100_values = jnp.unique(samples[i-100:i],axis=0)
            cov_matrix = jnp.cov(last_100_values, rowvar=False)
            variances = cov_matrix.diagonal()
            M = jnp.diag(1.0 / variances)
            Cov_Matrix = jnp.diag(variances)
            print(M)

            # Update epsilon
            theta = samples[i]
            epsilon = find_reasonable_epsilon(theta, sig, key, M = M, 
                        Cov_Matrix = Cov_Matrix, U = U, grad_U = grad_U)
    
        print([i, neg_log_likelihood_sample, sig])
        
    return samples, acceptance_array, neg_log_likelihood_array, sig_array

#samples, acceptance_array, neg_log_likelihood_array, sig_array = full_sample(2000, key)

# Get the end time
#end_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#print("End Time:", end_time)

# Convert to a regular NumPy array
#samples_numpy_array = np.array(samples)

#betas_save_name = 'HMCresults/samples_' + str(num_zeros) + '_int_mtx' + end_time + '.npy'

# # Save the array to a .npy file
#np.save(betas_save_name, samples_numpy_array)

#likelihood_save_name = 'HMCresults/neg_log_likelihood_' + str(num_zeros) + '_int_mtx' + end_time + '.npy'

# Convert to a regular NumPy array
#neg_log_likelihood_array_save = np.array(neg_log_likelihood_array)

# Save the array to a .npy file
#np.save(likelihood_save_name, neg_log_likelihood_array_save)

#sig_save_name = 'HMCresults/sig_' + str(num_zeros) + '_int_mtx' + end_time + '.npy'

#sig_array_save = np.array(sig_array)

#np.save(sig_save_name, sig_array_save)
