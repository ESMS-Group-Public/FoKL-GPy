# the following dataset was obtained from:
#   https://www.nonlinearbenchmark.org/
#       https://www.nonlinearbenchmark.org/benchmarks/f-16-gvt
#           https://data.4tu.nl/articles/_/12954911

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.FoKL import FoKLRoutines
from scipy.interpolate import griddata
from scipy.signal import argrelextrema

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ======================================================================================================================
# Import experimental data from csv files and define custom functions for preparing experimental data:


dataset = [0,1,'max'] # dataset to model . . .
# . . . = [acceleration location (0,1,2), logical if modeling amplitude, 'min' or 'max' amplitude if applicable]
res = 1e3 # number of datapoints to look at (decrease for sake of example to decrease computation time)

dfs = []
for n in [1,3,5,7]:
    df = pd.read_csv(rf'F16GVT_Files\BenchmarkData\F16Data_SineSw_Level{n}.csv')
    dfs.append(df)

def clean_data_for_F16_example(dfs):

    # clean up raw data:
    forceAmp = [4.8,28.8,67.0,95.6] # N, amplitude of force for Levels 1,3,5,7
    inputs = []
    data = []
    for ii in [0,1,2,3]:
        force = np.array(dfs[ii]['Force'])
        accel = np.array((dfs[ii]['Acceleration1'],dfs[ii]['Acceleration2'],dfs[ii]['Acceleration3']))

        # clean data by removing data points before first peak, exceeding force amplitude, and zero's at end
        def find_first_peak_index(force_amp,force_vector):
            start_index = np.argmax(force_vector>0.95*force_amp) # index to start looking for local max (i.e., first peak)
            initial_peak_value = force_vector[start_index]
            # iterate forward from start_index until local max is found
            max_peak_index = start_index
            max_peak_value = initial_peak_value
            for new_index in range(start_index+1, len(force_vector)):
                if force_vector[new_index] > max_peak_value:
                    max_peak_value = force_vector[new_index]
                    max_peak_index = new_index
                elif force_vector[new_index] < initial_peak_value:
                    return max_peak_index+1 # return next index, to ensure local max is not before true peak
        mm = find_first_peak_index(forceAmp[ii], force) # index to begin using data
        mask = abs(force) <= forceAmp[ii] # mask datapoints exceeding nominal force amplitude
        mask[:mm+1] = False # mask datapoints before initial peak
        mm_end = (force != 0).nonzero()[0][-1]
        mask[mm_end+1::] = False  # mask datapoints after force is zero (i.e., vibration turned off)

        accel = accel[:, mask] # apply masking to 'data'
        force = force[mask] # apply masking to 'inputs'
        L = len(force)
        freq = np.linspace(15, 2, L)  # assuming 15Hz at first peak and 2Hz at vibration turned off

        # calculate phase of vibrational force to provide as an input
        phase = np.arccos(force/forceAmp[ii]) # all within first and second quadrant
        def true_phase_adjustment(phase_q12):
            # get forward indices of oscillation flip (assuming continuous changes):
            ff_out = []
            diff0 = phase_q12[1]-phase_q12[0] # should be positive if phase_q12 starts jsut before local peak
            for ff in range(2,len(phase_q12)):
                diff1 = phase_q12[ff]-phase_q12[ff-1]
                if np.sign(diff1) != np.sign(diff0): # if derivative of phase changes signs, then oscillation flips
                    ff_out.append(ff)
                diff0 = diff1
            phase_true = np.copy(phase_q12)
            for ff in range(0,len(ff_out)-1,2):
                phase_true[ff_out[ff]:ff_out[ff+1]] = 2*np.pi-phase_q12[ff_out[ff]:ff_out[ff+1]]
            if not len(ff_out) % 2 == 0: # if not even number of oscillation flips then phase_true ends in q3 or q4
                phase_true[ff_out[-1]::] = 2*np.pi-phase_q12[ff_out[-1]::]
            return phase_true
        phase = true_phase_adjustment(phase) # adjust phase values to all quadrants (assumes continuous changes)

        # inputs.append([freq,force,phase])
        forceAmpVec = np.array([forceAmp[ii]]*L)

        inputs.append([freq,forceAmpVec,phase])
        data.append(accel)

    # Combine multiple different 'inputs' and 'data' into one single 'inputs' and 'data', excluding third dataset
    freq_stacked = np.concatenate([inputs[0][0],inputs[1][0],inputs[3][0]])
    forceAmp_stacked = np.concatenate([inputs[0][1],inputs[1][1],inputs[3][1]])
    phase_stacked = np.concatenate([inputs[0][2],inputs[1][2],inputs[3][2]])
    accel1_stacked = np.concatenate([data[0][0],data[1][0],data[3][0]])
    accel2_stacked = np.concatenate([data[0][1], data[1][1], data[3][1]])
    accel3_stacked = np.concatenate([data[0][2], data[1][2], data[3][2]])

    inputs_stacked = np.array([freq_stacked,forceAmp_stacked,phase_stacked])
    data_stacked = np.array([accel1_stacked,accel2_stacked,accel3_stacked])

    return inputs_stacked, data_stacked

def use_accelAmp_for_F16_example(inputs, data): # return data as Nx2 for [min,max] extrema
    inputs_min = []
    inputs_max = []
    data_min = []
    data_max = []
    for a in [0,1,2]:
        min_id = argrelextrema(data[a], np.less)  # indices of acceleration's local min's
        max_id = argrelextrema(data[a], np.greater) # indices of acceleration's local max's

        # consider only local min's/max's that are negative/positive
        min_id = min_id[0][data[a][min_id] < 0]
        max_id = max_id[0][data[a][max_id] < 0]

        inputs_min.append(inputs[:, min_id])
        inputs_max.append(inputs[:, max_id])
        data_min.append(data[a, min_id])
        data_max.append(data[a, max_id])
    return inputs_min, inputs_max, data_min, data_max


# ======================================================================================================================
# Prepare experimental data for relationships to model:


# each dataset combines data from 4.8, 28.8, and 95.6 N for one-out-of-three independent acceleration measurements

# clean raw data (NOT cleaning for machine learning purposes, but cleaning for focus of experiment/research)
userinputs, userdata = clean_data_for_F16_example(dfs)
inputs_min, inputs_max, data_min, data_max = use_accelAmp_for_F16_example(userinputs, userdata)

# select specific dataset to model
accel_id = dataset[0]
if dataset[1] == 1:
    if dataset[2] == 'min':
        userinputs = inputs_min[accel_id]
        userdata = data_min[accel_id]
    elif dataset[2] == 'max':
        userinputs = inputs_max[accel_id]
        userdata = data_max[accel_id]
elif dataset[1] == 0:
    userdata = userdata[accel_id]

# for sake of example, shrink datasets to reduce computation time
userinputs = userinputs[:, -round(res):]
userdata = userdata[-round(res):]


# ======================================================================================================================
# Fit FoKL model:


# initialize a FoKL class for the dataset being modeled
model = FoKLRoutines.FoKL() # leave argument blank to use default hypers

# customize how model.fit() treats user inputs and data by defining keywords
p_train = 0.75 # use 75% of data for training and the rest for validation testing
CatchOutliers = 'Data' # only look at 'data' for outliers to remove
OutliersMethod = 'Method 1' # apply 'Method 1' for defining what constitutes an outlier

# test all inputs and data (i.e., train and test both included)
betas, mtx, evs = model.fit(userinputs, userdata, p_train=p_train, CatchOutliers=CatchOutliers, OutliersMethod=OutliersMethod)


# NOTES REGARDING ACCESSING MODEL ATTRIBUTES AFTER CALLING 'model.fit()' . . .

# 'inputs' and 'data' have now been automatically formatted, cleaned, reduced to a train set, and stored as attributes.
# > to access all relevant 'inputs' and 'data' after model.fit(), use:
#   > model.inputs         == all normalized inputs w/o outliers (i.e., model.traininputs plus model.testinputs)
#   > model.data           == all data w/o outliers (i.e., model.traindata plus model.testdata)
# > to access other attributes, use:
#   > model.rawinputs      == all normalized inputs w/ outliers == user's 'inputs' but normalized and formatted
#   > model.rawdata        == all data w/ outliers              == user's 'data' but formatted
#   > model.traininputs    == train set of model.inputs
#   > model.traindata      == train set of model.data
#   > model.testinputs     == test set of model.inputs
#   > model.testdata       == test set of model.data
#   > model.normalize      == [min, max] factors used to normalize user's 'inputs' to 0-1 scale of model.rawinputs
#   > model.outliers       == indices removed from model.rawinputs and model.rawdata as outliers
#   > model.trainlog       == indices of model.inputs used for model.traininputs
#   > model.testlog        == indices of model.data used for model.traindata
# > to access numpy versions of attributes related to 'inputs', use:
#   > model.inputs_np      == model.inputs as a numpy array of timestamps x input variables
#   > model.rawinputs_np   == model.rawinputs as a numpy array of timestamps x input variables
#   > model.traininputs_np == model.traininputs as a numpy array of timestamps x input variables
#   > model.testinputs_np  == model.testinputs as a numpy array of timestamps x input variables


# ======================================================================================================================
# Standard post-processing of FoKL model:


meen, bounds, rmse = model.coverage3(model.inputs, model.data, model.draws, 1)


# ======================================================================================================================
# Additional user post-processing of FoKL model:


# user plot controls:
absAccel = 1 # 1 if plotting abs(accel), 0 if +/-



b=1

# SORTED
sort_id = np.argsort(meen1)
meen1_sort = meen1[sort_id]
bounds1_sort = np.transpose(np.array([bounds1[:, 0][sort_id], bounds1[:, 1][sort_id]]))
data_sort = model1.data[sort_id]

# COVERAGE3 PLOT
# plt.plot(model1.data_validn, 'g.') # test
plt.plot(model1.data, 'r.') # train
plt.plot(meen1, 'b', linewidth=2)
plt.plot(bounds1[:, 0], 'k--')
plt.plot(bounds1[:, 1], 'k--')

# WITHOUT ABS(ACCEL):
plt.plot(data_sort, 'ro')
plt.plot(meen1_sort, 'b', linewidth=2)
plt.plot(bounds1_sort[:, 0], 'k--')
plt.plot(bounds1_sort[:, 1], 'k--')
plt.ylabel('Acceleration (m/s^2)')

# WITH ABS(ACCEL):
# plt.plot(np.abs(meen1_sort), 'b', linewidth=2)
# plt.plot(np.abs(bounds1_sort[:, 0]), 'k--')
# plt.plot(np.abs(bounds1_sort[:, 1]), 'k--')
# plt.plot(np.abs(data_sort), 'ro')
# plt.ylabel('|Acceleration| (m/s^2)')

plt.show()

# plt.cla()



b=1


# de-normalize inputs
inputs_scale = model1.inputs_scale # [min,max] used to normalize each input
inputs_np = model1.inputs_np
M = np.shape(inputs_np)[1] # number of inputs
inputs_denorm = []
for ii in range(M):
    inp_min = inputs_scale[ii][0]
    inp_max = inputs_scale[ii][1]
    inputs_denorm.append(np.array(inputs_np[:,ii] * (inp_max-inp_min) + inp_min))

# data for color map
x = inputs_denorm[0] # frequency
y = inputs_denorm[1] # force amplitude
z = model1.data # predicted acceleration at location 1

# define grid of force amp and freq pairs
xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
xi, yi = np.meshgrid(xi, yi)

# interpolate acceleration on the grid
zi = griddata((x, y), z, (xi, yi), method='linear')
zi = np.squeeze(zi)

if absAccel:
    zi = np.abs(zi)
    cbar_title = '|Acceleration| (m/s^2)'
else:
    cbar_title = 'Acceleration (m/s^2)'
plt.contourf(xi, yi, zi, cmap='viridis')
cbar = plt.colorbar()
cbar.set_label(cbar_title)

plt.xlabel('Frequency (Hz)')
plt.ylabel('Force Amplitude (N)')
plt.title('Induced Acceleration at Location 1, Measured Experimentally')
plt.show()




b=1



