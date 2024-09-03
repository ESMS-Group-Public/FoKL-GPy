# Import Relevant Libraries

# For Data Processing
import numpy as np
import pandas as pd
from FoKL import FoKLRoutines

# Create FoKL Object
model = FoKLRoutines.FoKL()

# For Graphing
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = "browser"

# Sigmoid Generation

df = pd.DataFrame()

for i in np.arange(0,1,0.01): # x
    for j in np.arange(0,1,0.01): # y

        data = 1/((1+np.exp(-5*i + 2.5))*(1+np.exp(-5*j + 2.5)))
        new_row = pd.DataFrame({'x':[i], 'y':[j], 'data':[data]})
        df = pd.concat([df, new_row], ignore_index = True)

inputs = df[['x','y']].to_numpy().tolist()
data = df['data'].to_numpy()
data = np.expand_dims(data, axis = 0).T

# Initial Conditions

# sigsqd0
model.sigsqd0 = 0.009

# a (inverse gamma distribution shape factor for data observation error variance)
model.a = 9

# b (inverse gamma distribution scale factor for data observation error variance)
model.b = 0.01

# atau (inverse gamma distribution shape factor for beta prior variance)
model.atau = 3

# btau (inverse gamma distribution scale factor for beta prior variance)
model.btau = 4000

# tolerance (number of times needed to run Gibbs Sampling with a higher BIC than previous lowest (best) value)
model.tolerance = 3

# Allows for exclusion of term combinations (in this case none)
model.relats_in = []

# draws from the posterior for each tested model
model.draws = 1000

model.gimmie = False
model.aic = False

model.burnin = 0

# New update hyperparamters
model.update = True # To use update methodology, must be called PRIOR to first fitting
model.built = False # Defines whether fresh model is created or prior model used
model.burn = 500 # burn draws are disregarded prior to update fitting

x = inputs
y = data

# Call clean to format inputs, minmax MUST be specified when updating to avoid double normalization with fit calls

model.clean(x[0:4999],y[0:4999], minmax=[[0,1],[0,1]])

# First fitting
model.fit()

print('First Fit Completed')

mean = model.evaluate(inputs = x)

# # Need to add test-betas[-1] because this is the constant value that doesn't get added in bss_eval
df['prediction'] = (np.array(mean))

fig = px.scatter_3d()

fig.add_trace(go.Scatter3d(
    x=df['x'][0:4999],
    y=df['y'][0:4999],
    z=df['data'][0:4999],
    mode='markers',
    name='Given'
))
fig.add_trace(go.Scatter3d(
    x=df['x'][5000:-1],
    y=df['y'][5000:-1],
    z=df['data'][5000:-1],
    mode='markers',
    name='Withheld'
))
fig.add_trace(go.Scatter3d(
    x=df['x'],
    y=df['y'],
    z=df['prediction'],
    mode='markers',
    name='Predictions'
))

fig.show()

# Model inputs redefined
model.data = y[5000:-1]

# New inputs must be cleaned
model.inputs = model.clean(x[5000:-1])

# Second fitting
model.fit()

# Evaluation of new model
mean = model.evaluate(inputs=x)

df['prediction'] = (np.array(mean))

fig = px.scatter_3d()

fig.add_trace(go.Scatter3d(
    x=df['x'],
    y=df['y'],
    z=df['data'],
    mode='markers',
    name='Given'
))

fig.add_trace(go.Scatter3d(
    x=df['x'],
    y=df['y'],
    z=df['prediction'],
    mode='markers',
    name='Predictions'
))

fig.show()
