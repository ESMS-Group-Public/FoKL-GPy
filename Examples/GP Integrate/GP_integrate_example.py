import csv
import numpy as np
import FoKL
from FoKL import FoKLRoutines
import matplotlib.pyplot as plt

fkl = FoKLRoutines.FoKL()

#BSS-ANOVA GP Integration Example Script. Kyle Hayes and Derek Slack, using code from Dave Mebane.
#This case assumes you are starting your BSS-ANOVA model building from scratch, 
# and have fairly minimal knowledge of the code found in the 
#ESMS-Group-Public/FoKL-GP repository. 

#Our first step is to load and format the data correctly in order to create 
# FoKL model using emulator.
#For the purposes of this example we will use data concerning a set of
# cascaded tanks (2 outputs, 3 inputs)

 #we will create matrices of training
#inputs and outputs. For the purposes of this example we will use half the
#data for creating the model and half for testing
#Differentiation is carried out here by a center difference method since
#the data is smooth. If the data is noisy, using a smoothing method or
#alternate differentiation method. If derivatives are already available,
#skip this step.

#In order to differentiate, we note that the data is sampled at a uniform 4
#second rate

#inputs
traininputs = np.loadtxt('traininputs.txt',dtype=float,delimiter=',')
traindata1 = np.loadtxt('traindata1.txt',dtype=float,delimiter=',')
traindata2 = np.loadtxt('traindata2.txt',dtype=float,delimiter=',')
y = np.loadtxt('y.txt',dtype=float,delimiter=',')

#'u' is our forcing function: an independent variable that varies with
#time. The integrator can handle an arbitrary number of these inputs so
#long as their value is known (or can be approximated) at each time step
#within the integrated range. Here u is our only additional input, but
#for any cases with additional inputs, GP_Integrate would be provided a
#matrix of all these inputs rather than the vector shown here.
#All additional inputs still need to be normalized

utest = np.loadtxt('utest.csv',dtype=float,delimiter=',')

# data
relats_in = [1,1,1,1,1,1]

# generating phis from the spline coefficients text file, getKernels() is a hard coded spline coefficent

phis = getKernels()

# set hyper parameters to build the model

#The BSS-ANOVA model requires the selection hyperparameters (a, b, atau,
#and btau). 4 is a good starting value for a and atau. Increase the value of
#a to decrease the 'spread' of the predictions made by the model. b and btau can be
#calculated from the chosen values of a and atau, as well as an initial
#guess concerning the varience of the noise in the datasets (sigma squared), 
#and for btau the scale of the data


# a
a = 1000

# b
b = 1

# atau
atau = 4

# btau
btau = 0.6091

# tolerance
tolerance = 3
relats_in  = [1,1,1,1,1,1]
# draws
draws = 2000

gimmie = False
way3 = True
threshav = 0
threshstda = 0
threshstdb = 100
aic = False
n,m = np.shape(y)

#Now we create matricies of the training inputs. All inputs need to be
#normalized on a 0 to 1 range. In this case both models use the same sets
#of inputs, so only one input matrix will be created, but if the inputs
#differed for each model seperate matrices would need to be built for each.

#The order of the inputs in the input matrices matters for the integrator.
#Although inconsistantly ordered matrices can be corrected later, it will be
#fastest if input matrices are all set up in the following manner:
#The first columns should be the undifferentiated verisons of the output
#itself (the data for the 1st model, then the 2nd and so on)
#The following columns should contain any other inputs, which can be
#ordered in any manner, so long as that order is consistant for all models.

norms1 = [np.min(y[0,0:int(m/2)]),np.max(y[0,0:int(m/2)])]
norms2 = [np.min(y[1,0:int(m/2)]),np.max(y[1,0:int(m/2)])]
norms = np.transpose([norms1,norms2])

#We can now create our BSS-ANOVA models. Default values are used for
#several inputs to emulator, better results may be obatained with fine
#tuning these variables

betas1, mtx1, evs1 = fkl.emulator(traininputs, traindata1, phis, relats_in, a, b, atau, btau, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)
betas2, mtx2, evs2 = fkl.emulator(traininputs, traindata2, phis, relats_in, a, b, atau, 1, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)

#It's a good idea to throw out the draws from the 'burn in' period; the 1st
#1000 draws should be a safe estimate

betas1 = betas1[1000:]
betas2 = betas2[1000:]

#Models are now created for each of the derivatives we seeked to model. We
#can plot the predictions of the derivatives using coverage, or evaluate
#these predictions for test data using bss_eval. An example of using coverage
#for the first model is shown below:
fkl.coverage(betas1, traininputs, traindata1, phis, mtx1, 50, 1)


# For this example, we will now move into integrating these models.

#Integration

#Integration requires specifying several basics: initial conditions, step
#sizes, start and stop points, and the values of any additional inputs.

start = 4
stop = 3750*4
stepsize = 4
ic = y[:,int(m/2)-1]

#Inputs to the integrator should be placed in the same order as they were
#in the integrator. Since only one set of inputs is provided to the
#integrator (since all models need to be evaluated iteratively) if models
#were built with their inputs in different orders, then the used_inputs
#input to GP_Integrate can be provided additional information to rectify
#the issue (see the documentation of GP_Integrate for more information).

#In the case where all inputs are used for all models (in the correct order),
#used_inputs should be a cell matrix constisting of vectors of ones

used_inputs = [[1,1,1],[1,1,1]]

#All derivative models need to be concatenated into a cell array (in the
#same order as they are contained within the inputs). If the mean
#prediction is all that is required, then the mean beta model should be
#provided to GP_Integrate (accomplished by averaging the columns of the
#beta matricies)


T,Y = fkl.GP_Integrate([np.mean(betas1,axis=0),np.mean(betas2,axis=0)], [mtx1,mtx2], utest, norms, phis, start, stop, ic, stepsize, used_inputs)

#Simple plots comparing the predictions to the actual data

plt.plot(T,Y[0],T,y[0][3750:7500])
plt.plot(T,Y[1],T,y[1][3750:7500])
