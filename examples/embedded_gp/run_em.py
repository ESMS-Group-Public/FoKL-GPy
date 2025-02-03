
# Import Relevant Libraries

# FoKL for Model Building
from src.FoKL import Experimental_Embedded_GPs as FoKL_Embedded_GPs

# For Data Manipulation
import pandas as pd
import jax.numpy as jnp
import numpy as np

# For Visualization
import matplotlib.pyplot as plt

data = pd.read_csv('CSTR_data_with_noise.csv')

# Define the phis (basis functions) used
phis = jnp.array(FoKL_Embedded_GPs.getKernels.sp500())

# Data
Inv_temp = data['Temperature_Inv'].to_numpy()
C_CO2 = data['C_CO2'].to_numpy()
C_Sites = data['C_Sites'].to_numpy()
C_CO2_ADS = data['C_CO2_ADS'].to_numpy()

# Outcome variable
r_co2 = data['r_co2'].to_numpy()

def normalize_inputs(inputs, min, max):
    normalized = (inputs - min)/(max - min)
    return normalized

# Define inputs and normalize
inputs = jnp.array([Inv_temp])
inputs_norm = normalize_inputs(inputs, 1/600, 1/300)

# Create object for each individual GP
GPf = FoKL_Embedded_GPs.GP()
GPb = FoKL_Embedded_GPs.GP()

# Create of model and define the number of GP's in it
model = FoKL_Embedded_GPs.Embedded_GP_Model(GPf, GPb)

# Define appropriate parameters to model
model.inputs = inputs_norm.T
model.phis = phis
model.data = r_co2

# Define overall PIML model
def equation(GP_Results):
    r_co2 = -(jnp.exp(-(GP_Results[0]))*C_CO2*C_Sites - jnp.exp(-(GP_Results[1]))*C_CO2_ADS)
    return r_co2

model.set_equation(equation)

samples, matrix, BIC = model.full_routine(draws = 10, tolerance = 0)

#%%
Inv_Temperature = np.array([np.linspace(300,600,50)])**-1
Inv_Temperature_norm = normalize_inputs(Inv_Temperature, 1/600, 1/300)

y_mean, y_bounds = model.evaluate(Inv_Temperature_norm.T, GP_number=0, draws = 10, ReturnBounds = 1)

x = inputs[0]
plt.rcParams['font.size'] = 18
plt.figure(figsize=(8, 6))
# Scatter plot
plt.scatter(x**-1, jnp.exp(-100*x), label='Data Points', c='black')

# Line plot for the mean
plt.plot(jnp.flip(jnp.unique(Inv_Temperature**-1)), jnp.exp(-jnp.unique(y_mean)), label='Mean Predictions', linewidth = 4)

# Line plots for upper and lower bounds, if needed though this example doesn't have much error variance
# plt.plot(jnp.flip(jnp.unique(Inv_Temperature**-1)), jnp.exp(-jnp.unique(y_bounds[:, 0])), label='95% HDI', linewidth = 4, c = 'orange', linestyle='dashed')
# plt.plot(jnp.flip(jnp.unique(Inv_Temperature**-1)), jnp.exp(-jnp.unique(y_bounds[:, 1])), linewidth = 4, c = 'orange', linestyle='dashed')


# Adding labels and legend
plt.xlabel('Temperature (K)')
plt.ylabel('$k_{forward}$ (' + r'$\frac{mol}{m^3s}$)')
plt.title('Comparison of $k_{forward}$ Data with GP Estimation')
plt.legend()

# Show plot
plt.show()
#%%
y_mean, y_bounds = model.evaluate(Inv_Temperature_norm.T, GP_number=1, draws = 10, ReturnBounds = 1)

x = inputs[0]
plt.rcParams['font.size'] = 18
plt.figure(figsize=(8, 6))

# Scatter plot
plt.scatter(x**-1, jnp.exp(-200*x), label='Data Points', c='black')

# Line plot for the mean
plt.plot(jnp.flip(jnp.unique(Inv_Temperature**-1)), jnp.exp(-jnp.unique(y_mean)), label='Mean Predictions', linewidth = 4)

# Line plots for upper and lower bounds, if needed though this example doesn't have much error variance
# plt.plot(jnp.flip(jnp.unique(Inv_Temperature**-1)), jnp.exp(-jnp.unique(y_bounds[:, 0])), label='95% HDI', linewidth = 4, c = 'orange', linestyle='dashed')
# plt.plot(jnp.flip(jnp.unique(Inv_Temperature**-1)), jnp.exp(-jnp.unique(y_bounds[:, 1])), linewidth = 4, c = 'orange', linestyle='dashed')

# Adding labels and legend
plt.xlabel('Temperature (K)')
plt.ylabel('$k_{backwards}$ (' + r'$\frac{1}{s}$)')
plt.title('Comparison of $k_{backwards}$ Data with GP Estimation')
plt.legend()

# Show plot
plt.show()