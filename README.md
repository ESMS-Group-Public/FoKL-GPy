# FoKL
Karhunen Loève decomposed Gaussian processes with forward variable
selection. Use this package for scalable GP regression and fast
inference on static and dynamic datasets.

# How to Use
To install: 'pip install FoKL', or clone this repo

Once installed, import the package into your environment with:
```
'import FoKL'
```
 and then the proper routines with:
 ```
 from FoKL import FoKLRoutines
```
for ease of use it is recommended to call the class with an abreviation 'fkl' s:
```
fkl = FoKLRoutines.FoKL()
```
Now you are ready to beginning creating your model. FoKL depends on a kernel to do its linear regression, these kernels can be called with the 'getKernel()' function. 
The getKernel() function calls a collection of splines that is good for general use, but future updates can add more kernels.
```
phis = getKernel()
```
Once the kernel is defined, you can initialize your model the required hyper paramters.
```
'model = fkl(phis, relats_in, a, b, atau, btau, tolerance, draws, gimmie, way3, threshav, threshstda, threshstdb, aic)'
```
- The definition of each of these hypers can be found within the function documentation.

With the model defined the training can begin by calling the fit function:
```
'model.fit( Normalized Training Inputs, Training Data)'
```
- documentation of fit() inputs and outputs can be found in the function documentation.

The console will display the index and bic of the model being built in real time.
Once completed the model can be validated with the coverage3 function:
```
'model.converage3(Normalized Test Inputs, Test Data, draws, plots)'
```
- documentation of coverage inputs and outputs can be found in function documentation.


Cover

Please cite: K. Hayes, M.W. Fouts, A. Baheri and
D.S. Mebane, "Forward variable selection enables fast and accurate
dynamic system identification with Karhunen-Loève decomposed Gaussian
processes", arXiv:2205.13676

Credits: David Mebane (ideas and original code), Kyle Hayes
(integrator), Derek Slack (Python porting)

Funding provided by National Science Foundation, Award No. 2119688


