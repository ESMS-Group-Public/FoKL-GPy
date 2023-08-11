# FoKL
Karhunen Loève decomposed Gaussian processes with forward variable
selection. Use this package for scalable GP regression and fast
inference on static and dynamic datasets.

## How to Use
To install: 'pip install FoKL', or clone this repo

Once installed, import the package into your environment with:
```
import FoKL
```
and then the proper routines with:
 ```
from FoKL import FoKLRoutines
from FoKL import getKernels
```

Now you are ready to beginning creating your model. FoKL depends on a kernel to do its linear regression, these kernels can be called with the 'getKernel()' function. 
The getKernel() function calls a collection of splines that is good for general use, but future updates can add more kernels.
```
phis = getKernels()
```
Once the kernel is defined, you can initialize your model the required hyper paramters.
```
model = FoKLRoutines.FoKL(phis, relats_in, a, b, atau, btau, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)
```
- The definition of each of these hypers can be found within the function documentation.

With the model defined the training can begin by calling the fit function:
```
model.fit( Normalized Training Inputs, Training Data)
```

The console will display the index and bic of the model being built in real time.
Once completed the model can be validated with the coverage3 function:
```
model.converage3(Normalized Test Inputs, Test Data, draws, plots)
```
Cover

## Integration
FoKL can be used to model state derivatives and thus contains an integration method of these states using an RK4. Due to each state being modeled independently the same functionality cannot be used. The model.fits outputs should be returned to your workspace via:

```
betas, mtx, evs = model.fit(inputs, data)
```
and then used as 

```
T,Y = FoKLRoutines.FoKL.GP_Integrate([np.mean(betas1,axis=0),np.mean(betas2,axis=0)], [mtx1,mtx2], utest, norms, phis, start, stop, ic, stepsize, used_inputs)
```

An example of the integration functionality can be seen in GP_intergrate_example.py

## Citations
Please cite: K. Hayes, M.W. Fouts, A. Baheri and
D.S. Mebane, "Forward variable selection enables fast and accurate
dynamic system identification with Karhunen-Loève decomposed Gaussian
processes", arXiv:2205.13676

Credits: David Mebane (ideas and original code), Kyle Hayes
(integrator), Derek Slack (Python porting)

Funding provided by National Science Foundation, Award No. 2119688


