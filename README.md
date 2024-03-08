![FoKL-GPy Logo](docs/_static/fokl-gpy_banner.png)

--------------------------------------------------------------------------------

## Contents
<!-- toc -->

- [About FoKL](#about-fokl)
- [Installation and Setup](#installation-and-setup)
- [Use Cases](#use-cases)
- [Package Documentation](#package-documentation)
- [Benchmarks and Papers](#benchmarks-and-papers)
- [Future Development](#future-development)
- [Contact Us](#contact-us)
- [License](#license)
- [Citations](#citations)

<!-- tocstop -->

[//]: # (![FoKL-GPy Logo]&#40;https://github.com/ESMS-Group-Public/FoKL-GPy/docs/_static/esms_logo.png&#41;)
[//]: # (![FoKL-GPy Logo]&#40;https://github.com/jakekrell/FoKL-GPy/tree/debug_bernoulli/docs/_static/test.png&#41;)

## About FoKL

FoKL-GPy, or FoKL, is a Python package intended for use in machine learning. The name comes from a unique implementation 
of **Fo**rward variable selection using **K**arhunen-**L**oève decomposed **G**aussian **P**rocesses (GP's) in 
**Py**thon (i.e., **FoKL-GPy**). 

The primary advantages of FoKL are:
- Fast inference on static and dynamic datasets using scalable GP regression
- Significant accuracy retained despite being fast

Some other advantages of FoKL include:
- Export modeled non-linear dynamics as a symbolic equation
- Take first and second derivatives of model with respect to any input variable (e.g., gradient)
- Multiple kernels available (BSS-ANOVA, Bernoulli Polynomials)
- Automatic normalization and formatting of dataset
- Automatically split specified fraction of dataset into a training set and retain the rest for testing
- Easy adjusting of hyperparameters for sweeping through variations in order to find optimal settings
- Ability to save, share, and load models
- Ability to evaluate a model without known data

To read more about FoKL, please see the [Benchmarks and Papers](#benchmarks-and-papers) section.

## Installation and Setup

FoKL is available through PyPI:

```cmd
pip install FoKL
```

Alternatively, the GitHub repository may be cloned to create a local copy in which the examples and documentation will 
be included:

```cmd
git clone https://github.com/ESMS-Group-Public/FoKL-GPy
```

Once installed, import the FoKL module into your environment with:

```
from FoKL import FoKLRoutines
```

If loading a pre-existing FoKL class object:

```
model = FoKLRoutines.load('filename_of_model')

# Or, if your pre-existing model is in a different folder than your Python script:
model = FoKLRoutines.load('filename_of_model', 'directory_of_model')
```

Else if creating a new FoKL class object:
```
model = FoKLRoutines.FoKL()
```

Now you may call methods on the class and reference its attributes. For help with this, please see [Use Cases](#use-cases).

## Use Cases

Please first refer to the following for tutorials and examples:
- [Training and/or evaluating a model](docs/tutorials/clean.py)
- [Saving and loading a model](docs/tutorials/save_and_load/save_and_load.py)
- [Converting to Pyomo](docs/tutorials/fokl_to_pyomo.py)
- [Plotting and RMSE](examples/sigmoid/sigmoid.py)
- [Integration](examples/gp_integrate/gp_integrate.py)

Then, see [Package Documentation](#package-documentation) as needed.

## Package Documentation

  - [FoKLRoutines.load](#foklroutinesload)
  - [FoKLRoutines.FoKL](#foklroutinesfokl)
    - [clean](#clean)
    - [bss_derivatives](#bss_derivatives)
    - [evaluate_basis](#evaluate_basis)
    - [evaluate](#evaluate)
    - [coverage3](#coverage3)
    - [fit](#fit)
    - [clear](#clear)
    - [to_pyomo](#to_pyomo)
    - [save](#save)
  - [GP_integrate](#gp_integrate)

### FoKLRoutines.load

```
self = FoKLRoutines.load(filename, directory=None)
```

Load a FoKL class from a file.

By default, *directory* is the current working directory that contains the script calling this method. An absolute or 
relative directory may be defined if the model to load is located elsewhere.

For simplicity, enter the returned output from *self.save()* as the argument here, i.e., for *filename*. Do this while 
leaving *directory* blank since *filename* can simply include the directory itself.

### FoKLRoutines.FoKL

```
self = FoKLRoutines.FoKL(**kwargs)
```

This creates a class object that contains all information relevant to and defining a FoKL model.

Upon initialization, hyperparameters and some other settings are defined with default values as attributes of the FoKL 
class. These attributes are as follows, and any or all may be specified as a keyword or later updated by redefining the 
a class attribute.

| Type    | Keyword Argument  | Default Value   | Description                                                                                                               |
|---------|-------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------|
| hyper   | *kernel*          | 'Cubic Splines' | Basis functions (i.e., kernel) to use for training a model                                                                |
| hyper   | *phis*            | *f(kernel)*     | Data structure with coefficients for basis functions                                                                      |
| hyper   | *relats_in*       | []              | Boolean matrix indicating which input variables and/or interactions should be excluded from the model                     |
| hyper   | *a*               | 4               | Shape parameter of the initial-guess distribution for the observation error variance of the data                          |
| hyper   | *b*               | *f(a, data)*    | Scale parameter of the initial-guess distribution for the observation error variance of the data                          |
| hyper   | *atau*            | 4               | Parameter of the initial-guess distribution for the $tau^2$ parameter                                                     |
| hyper   | *btau*            | *f(atau, data)* | Parameter of the initial-guess distribution for the $tau^2$ parameter                                                     |
| hyper   | *tolerance*       | 3               | Influences how long to continue training after additional terms yield diminishing returns                                 |
| hyper   | *draws*           | 1000            | Total number of draws from the posterior for each tested model                                                            |
| hyper   | *gimmie*          | *False*         | Boolean to return the most complex model tried instead of the model with the optimum Bayesian information criterion (BIC) |
| hyper   | *way3*            | *False*         | Boolean to include three-way interactions                                                                                 |
| hyper   | *threshav*        | 0.05            | Threshold to propose terms for elimination. Increase to propose and eliminate more terms                                  |
| hyper   | *threshstda*      | 0.5             | Threshold to eliminate terms based on standard deviation relative to mean                                                 |
| hyper   | *threshstdb*      | 2               | Threshold to eliminate terms based on standard deviation independent of mean                                              |
| hyper   | *aic*             | *False*         | Boolean to use Aikaike information criterion (AIC)                                                                        |
| setting | *UserWarnings*    | *True*          | Boolean to print user-warnings (i.e., FoKL warnings) to command terminal                                                  |
| setting | *ConsoleOutput*   | *True*          | Boolean to print progress of model training to command terminal                                                           |

The following methods are embedded within the class object:

| Method                                | Description                                                                                    |
|---------------------------------------|------------------------------------------------------------------------------------------------|
| [*clean*](#clean)                     | Automatically format, normalize, and create test/train sets from user's provided dataset.      |
| [*bss_derivatives*](#bss_derivatives) | Algebraically calculate partial derivatives of model with respect to input variables.          |
| [*evaluate_basis*](#evaluate_basis)   | Calculate value of specified basis function at single point along normalized domain.           |
| [*evaluate*](#evaluate)               | Calculate values of FoKL model for all requested sets of datapoints.                           |
| [*coverage3*](#coverage3)             | Evaluate FoKL model, calculate confidence bounds, calculate RMSE, and produce plot.            |
| [*fit*](#fit)                         | Train new FoKL model to best-fit training dataset according to hyperparameters.                |
| [*clear*](#clear)                     | Delete attributes from FoKL class so that new models may be trained without new class objects. |
| [*to_pyomo*](#to_pyomo)               | Convert a FoKL model to an expression in a Pyomo model.                                        |
| [*save*](#save)                       | Save FoKL class with all its attributes to retain model and avoid re-training.                 |

Each method has optional inputs that allow for flexibility in how FoKL is used so that you may leverage these methods 
for your specific requirements. Please refer to the [Use Cases](#use-cases) first, then explore the following documentation of 
each method as needed.

#### clean

```
self.clean(self, inputs, data=None, **kwargs)
```

For cleaning and formatting inputs prior to training a FoKL model. Note that data is not required but should be entered 
if available; otherwise, leave blank.

| Input    | Type | Description                                                                                              | Default |
|----------|------|----------------------------------------------------------------------------------------------------------|---------|
| *inputs* | any  | NxM matrix of N experimental measurements, i.e., independent *x* variables for training *f(x1, ..., xM)* | n/a     |
| *data*   | any  | Nx1 vector of N experimental results, i.e., dependent *y* variable where <br/>*y = f(x1, ..., xM)*       | *None*  |

| Keyword                | Type   | Description                                  | Default  |
|------------------------|--------|----------------------------------------------|----------|
| *train*                | scalar | fraction of N datapoints to use for training | 1        |

Other keywords are in development to enable alternate routines of splitting the dataset into test and train sets according to the percentage set by *train*, but currently the split is performed randomly. Also in development are automatic outlier detection and removal routines.

After calling *clean*, several new attributes get defined for the FoKL class. The following is a full list but often only *self.inputs* and *self.data* are needed; however, *self.traininputs* and *self.traindata* as well as the numpy versions *self.inputs_np* and *self.traininputs_np* can be useful. Be sure to use these attributes in place of the original dataset entered as *inputs* and *data* so that normalization and formatting errors are avoided.

| Attribute             | Type  | Description                                                                            |
|-----------------------|-------|----------------------------------------------------------------------------------------|
| *self.inputs*         | list  | all normalized inputs w/o outliers (i.e., *self.traininputs* and *self.testinputs*)    |
| *self.data*           | numpy | all data w/o outliers (i.e., *self.traindata* and *self.testdata*)                     |
| *self.rawinputs*      | list  | all normalized inputs w/ outliers (i.e., user's *inputs* but normalized and formatted) |
| *self.rawdata*        | numpy | all data w/ outliers (i.e., user's *data* but formatted)                               |
| *self.traininputs*    | list  | train set of *self.inputs*                                                             |
| *self.traindata*      | numpy | train set of *self.data*                                                               |
| *self.testinputs*     | list  | test set of *self.inputs*                                                              |
| *self.testdata*       | numpy | test set of *self.data*                                                                |
| *self.normalize*      | list  | [min, max] factors used to normalize user's *inputs* to 0-1 scale of *self.rawinputs*  |
| *self.outliers*       | numpy | indices removed from *self.rawinputs* and *self.rawdata* as outliers                   |
| *self.trainlog*       | numpy | indices of *self.inputs* used for *self.traininputs*                                   |
| *self.testlog*        | numpy | indices of *self.data* used for *self.traindata*                                       |
| *self.inputs_np*      | numpy | *self.inputs* as a numpy array of experiments x input variables                        |
| *self.rawinputs_np*   | numpy | *self.rawinputs* as a numpy array of experiments x input variables                     |
| *self.traininputs_np* | numpy | *self.traininputs* as a numpy array of experiments x input variables                   |
| *self.testinputs_np*  | numpy | *self.testinputs* as a numpy array of experiments x input variables                    |

#### bss_derivatives

```
dy = self.bss_derivatives(self, **kwargs)
```

For returning gradient of modeled function with respect to each, or specified, input variable.
If user overrides default settings, then 1st and 2nd partial derivatives can be returned for any variables.

| Keyword           | Type                    | Description                                                                                                                                                                                                                                    | Default          |
|-------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|
| *inputs*          | numpy                   | NxM matrix of *x* input variables for training *f(x1, ..., xM)*                                                                                                                                                                                | *self.inputs_np* | 
| *kernel*          | string or integer       | basis functions to use for differentiation (i.e., 'Cubic Splines' or 'Bernoulli Polynomials')                                                                                                                                                  | *self.kernel*    |    
| *d1*              | boolean list or integer | index of input variable(s) (i.e., state(s)) to use for first partial derivative; see tip below                                                                                                                                                 | *True*           |           
| *d2*              | boolean list or integer | index of input variable(s) (i.e., state(s)) to use for second partial derivative; see tip below                                                                                                                                                | *False*          |          
| *draws*           | integer                 | total number of draws from the posterior for each tested model                                                                                                                                                                                 | *self.draws*     |     
| *betas*           | numpy                   | draw from the posterior distribution of coefficients                                                                                                                                                                                           | *self.betas*     |     
| *phis*            | list                    | coefficients for the basis functions                                                                                                                                                                                                           | *self.phis*      |      
| *mtx*             | numpy                   | basis function interaction matrix from the best model                                                                                                                                                                                          | *self.mtx*       |       
| *span*            | list                    | [min, max]'s of *inputs* used in *clean* during normalization                                                                                                                                                                                  | *self.normalize* | 
| *IndividualDraws* | boolean                 | for returning derivative(s) at each draw                                                                                                                                                                                                       | *False*          |              
| *ReturnFullArray* | boolean                 | for returning NxMx2 array with zeros for non-requested states such that indexing is preserved; otherwise, only requested states are squeezed into a 2D matrix where columns correspond to increasing input variable index and derivative order | *False*          |              

| Output | Type   | Description                                                                                     | Default  |
|--------|--------|-------------------------------------------------------------------------------------------------|----------|
| *dy*   | numpy  | derivative of model with respect to input variable(s) (i.e., state(s)) defined by *d1* and *d2* | gradient |

Tip:
- To turn off all first-derivatives, set *d1=False* instead of *d1=0*. The reason is *d1* and *d2*, if set to an integer,
 will return the derivative with respect to the input variable indexed by that integer using Python indexing.
 In other words, for a two-input FoKL model, setting *d1=1* and *d2=0* will return the first-derivative with
 respect to the second input (*d1=1*) and the second-derivative with respect to the first input (*d2=0*).
 Alternatively, *d1=[False, True]* and *d2=[True, False]* will function the same so that boolean lists may be used in cases where the derivative with respect to more than one state, but not all states, is required.

#### evaluate_basis

```
basis = self.evaluate_basis(self, c, x, kernel=None, d=0)
```

Evaluate a basis function at a single point by providing coefficients, *x* value(s), and (optionally) the kernel. This method is primarily used internally by other methods and so is not expected to be used by the user, but is available for testing purposes and to provide insight toward how the basis functions get evaluated in the *evaluate* method.

For evaluating a FoKL model, see [*evaluate*](#evaluate).

| Input | Type   | Description                                                           |
|-------|--------|-----------------------------------------------------------------------|
| *c*   | list   | coefficients of the basis function                                    |
| *x*   | scalar | value of independent variable at which to evaluate the basis function |

| Keyword  | Type              | Description                                                             | Default       |
|----------|-------------------|-------------------------------------------------------------------------|---------------|
| *kernel* | string or integer | basis function to evaluate ('Cubic Splines' or 'Bernoulli Polynomials') | *self.kernel* |
| *d*      | integer           | order of derivative (where 0 is no derivative)                          | 0             |

| Output  | Type   | Description                    |
|---------|--------|--------------------------------|
| *basis* | scalar | value of basis function at *x* |

If insightful for understanding how to define *c*, note the kernels yield the following basis function equations (using Python syntax):

| Kernel                  | Order | Basis                                                                         |
|-------------------------|-------|-------------------------------------------------------------------------------|
| 'Cubic Splines'         | *d=0* | <pre>c[0] + c[1] * x + c[2] * (x ** 2) + c[3] * (x ** 3)</pre>                |
| "                       | *d=1* | <pre>c[1] + 2 * c[2] * x + 3 * c[3] * (x ** 2)</pre>                          |
| "                       | *d=2* | <pre>2 * c[2] + 6 * c[3] * x</pre>                                            |
| 'Bernoulli Polynomials' | *d=0* | <pre>sum(c[k] * (x ** k) for k in range(len(c)))</pre>                        |
| "                       | *d=1* | <pre>sum(k * c[k] * (x ** (k - 1)) for k in range(1, len(c)))</pre>           |
| "                       | *d=2* | <pre>sum((k - 1) * k * c[k] * (x ** (k - 2)) for k in range(2, len(c)))</pre> |

#### evaluate

```
meen = self.evaluate(self, inputs=None, betas=None, mtx=None, **kwargs)
```

Evaluate the FoKL model for provided inputs and (optionally) calculate bounds.

| Input    | Type  | Description                                           | Default          |
|----------|-------|-------------------------------------------------------|------------------|
| *inputs* | numpy | input variable(s) at which to evaluate the FoKL model | *self.inputs_np* |
| *betas*  | numpy | coefficients defining FoKL model                      | *self.betas*     |
| *mtx*    | numpy | interaction matrix defining FoKL model                | *self.mtx*       |

| Keyword        | Type    | Description                                                                                                                                                                           | Default      |
|----------------|---------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| *normalize*    | list    | [min, max] factors to normalize *inputs*; leave blank if *inputs* is already normalized or set to *self.normalize* to use the same scale as the original dataset processed by *clean* | *None*       |
| *draws*        | integer | total number of draws from the posterior for each tested model                                                                                                                        | *self.draws* |
| *clean*        | boolean | to automatically normalize and format *inputs*; note this will override *normalize* and result in *inputs* scaled to 0-1                                                              | *False*      |
| *ReturnBounds* | boolean | to return confidence bounds as second output                                                                                                                                          | *False*      |

| Output  | Type  | Description                                                                        |
|---------|-------|------------------------------------------------------------------------------------|
| *meen*  | numpy | model predictions for provided *inputs* (i.e., *y* where *y = f(x0, x1, ..., xM)*) |

Note if attempting to automatically format *inputs* but normalize to different [min, max] values than those of *inputs*, this will have to be done manually. A work around to achieve this is as follows:

```
# Un-normalized and un-formatted 'raw' input variables of dataset:
x = [x0, x1, ..., xM]

# Initialize FoKL class:
f = FoKLRoutines.FoKL(...)

# Automatically normalize and format:
f.clean(x)

# Normalized and formatted 'raw' input variables of dataset:
x = f.inputs_np

# ----------------------------------------------------------------------------------------------------------------------

# Un-normalized and un-formatted 'other' input variables to evaluate:
u = [u0, u1, ..., uM]

# Initialize additional "throw-away" FoKL class in order to access clean without overriding attributes:
dummy = FoKLRoutines.FoKL(...)

# Automatically normalize and format:
dummy.clean(u)

# Normalized and formatted 'other' input variables to evaluate:
u = dummy.inputs_np

# Both x and u are normalized to 0-1 scales, but we need to re-scale u according to x:
u = u * (dummy.normalize[1] - dummy.normalize[0]) + dummy.normalize[0]  # scale of original u
u = (u - f.normalize[0]) / (f.normalize[1] - f.normalize[0])  # scale of normalized x

# ----------------------------------------------------------------------------------------------------------------------

# Both x and u are on the same scale now, normalized according to x, and so long as u is contained within x's extrema:
meen = f.evaluate(u)
```

The following will ***NOT*** achieve the same results:
```
meen != f.evaluate(u, clean=True)
meen != f.evaluate(u, clean=True, normalize=f.normalize)
```

However, if attempting to ***ONLY*** normalize *inputs* to different [min, max] values, implying the format of *inputs* matches exactly with the format of *inputs_np* returned by *clean* (that is, NxM numpy matrix), then the following will indeed achieve the same results:
```
meen = f.evaluate(u, normalize=f.normalize)
```

#### coverage3

```
meen, bounds, rmse = self.coverage3(self, **kwargs)
```

For validation testing of a FoKL model. Default functionality is to evaluate all inputs (i.e., train and test sets) using *evaluate*.
Returned is the predicted output *meen*, confidence bounds *bounds*, and Root Mean Square Error *rmse*. A plot
may be returned by setting *plot=True*; or, for a potentially more meaningful plot in terms of judging
accuracy, *plot='sorted'* will plot the data in increasing value.

To govern what is passed to *evaluate*:

| Keyword  | Type    | Description                                                       | Default       |
|----------|---------|-------------------------------------------------------------------|---------------|
| *inputs* | list    | normalized and properly formatted inputs to evaluate              | *self.inputs* |
| *data*   | numpy   | properly formatted data outputs to use for validating predictions | *self.data*   |
| *draws*  | integer | total number of draws from the posterior for each tested model    | *self.draws*  |

To govern basic plot controls:

| Keyword             | Type              | Description                                                   | Default  |
|---------------------|-------------------|---------------------------------------------------------------|----------|
| *plot*              | boolean or string | for generating plot; set to 'sorted' for plot of ordered data | *False*  |
| *bounds*            | boolean           | for plotting bounds                                           | *True*   |
| *xaxis*             | integer           | index of the input variable to plot along the x-axis          | indices  |
| *labels*            | boolean           | for adding labels to plot                                     | *True*   |
| *xlabel*            | string            | x-axis label                                                  | 'Index'  |
| *ylabel*            | string            | y-axis label                                                  | 'Data'   |
| *title*             | string            | plot title                                                    | 'FoKL    |
| *legend*            | boolean           | for adding legend to plot                                     | *True*   |
| *LegendLabelFoKL*   | string            | FoKL's label in legend                                        | 'FoKL'   |
| *LegendLabelData*   | string            | Data's label in legend                                        | 'Data'   |
| *LegendLabelBounds* | string            | Bounds's label in legend                                      | 'Bounds' |

To govern detailed plot controls:

| Keyword          | Type   | Description                 | Default |
|------------------|--------|-----------------------------|---------|
| *PlotTypeFoKL*   | string | FoKL's color and line type  | 'b'     |
| *PlotSizeFoKL*   | scalar | FoKL's line size            | 2       |
| *PlotTypeBounds* | string | Bounds' color and line type | 'k--'   |
| *PlotSizeBounds* | scalar | Bounds' line size           | 2       |
| *PlotTypeData*   | string | Data's color and line type  | 'ro'    |
| *PlotSizeData*   | scalar | Data's line size            | 2       |

| Output   | Type   | Description                                                           |
|----------|--------|-----------------------------------------------------------------------|
| *meen*   | numpy  | predicted output values for each indexed input (from *evaluate*)      |
| *bounds* | numpy  | confidence interval for each predicted output value (from *evaluate*) |
| *rmse*   | scalar | Root Mean Squared Error (RMSE) of prediction versus known data        |

#### fit

```
betas, mtx, evs = self.fit(self, inputs=None, data=None, **kwargs)
```

Training routine for fitting model to known inputs and data.

| Input    | Type  | Description                                                                                           | Default            |
|----------|-------|-------------------------------------------------------------------------------------------------------|--------------------|
| *inputs* | list  | NxM matrix of independent *x* variables for fitting <br/>*f(x1, ..., xM)*                             | *self.traininputs* |
| *data*   | numpy | Nx1 vector of dependent variable *y* to create model for predicting the value of *y = f(x1, ..., xM)* | *self.traindata*   |

| Keyword         | Type    | Description                                                                     | Default |
|-----------------|---------|---------------------------------------------------------------------------------|---------|
| *clean*         | boolean | to perform automatic normalization and formatting by internally calling *clean* | *False* |
| *ConsoleOutput* | boolean | to print [ind, ev] to console during FoKL model generation                      | *True*  |

If *clean=True*, then any keywords documented for the *clean* method may be used here in *fit*. See documentation for [*clean*](#clean), and note the class attributes that get defined for future reference.

| Output  | Type  | Description                                                                                                                                                                                                                                                 |
|---------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *betas* | numpy | draws from the posterior distribution of coefficients, with rows corresponding to single draws and columns corresponding to terms in the model                                                                                                              |
| *mtx*   | numpy | interaction matrix of the best model, with rows corresponding to terms in the model (i.e., columns of *betas* beyond the first column) and columns corresponding to input variables (i.e., columns of *self.inputs_np*); values are order of basis function |
| *ev*    | numpy | vector of BIC values from all the models evaluated                                                                                                                                                                                                          |

#### clear

```
self.clear(self, keep=None, clear=None, all=False)
```

Delete all attributes from the FoKL class except for hyperparameters and settings, unless otherwise specified by the *clear* keyword.
If an attribute is listed in both the *clear* and *keep* keywords, then the attribute is cleared.

| Input   | Type            | Description                                                                                       | Default     |
|---------|-----------------|---------------------------------------------------------------------------------------------------|-------------|
| *keep*  | list of strings | attributes to keep in addition to hyperparameters and settings, e.g., *keep=['inputs_np', 'mtx']* | *self.keep* |
| *clear* | list of strings | hyperparameters to delete, e.g., *clear=['kernel', 'phis']*                                       | *None*      |
| *all*   | boolean         | if *True* then all attributes (including hyperparameters) get deleted regardless                  | *False*     |

Note when the FoKL class was initialized, the names of the hyperparameters and settings which got defined as attributes were stored in a list of strings. This list of strings was defined as *self.keep* when the class was initialized, which here preserves those attributes by default. See documentation for [*FoKLRoutines.FoKL*](#foklroutinesfokl) to see a list of these attributes (i.e., hyperparameters and settings).

To remove all attributes from the class, simply call:
```
self.clear(all=True)
```

#### to_pyomo

```
m = self.to_pyomo(self, m=None, y=None, x=None, ReturnObjective=False)
```

Automatically convert a FoKL model trained with or defined by the 'Bernoulli Polynomials' kernel to a symbolic expression of a Pyomo model. By default, the symbolic FoKL expression is defined as a Pyomo constraint; however, it may be defined as the Pyomo model's objective with *ReturnObjective=True*.

| Input             | Type                               | Description                                                                                             | Default             |
|-------------------|------------------------------------|---------------------------------------------------------------------------------------------------------|---------------------|
| *m*               | Pyomo model                        | pre-existing Pyomo model if already defined                                                             | *None*              |
| *y*               | scalar                             | if known, value of FoKL model output variable to include in Pyomo model                                 | *None*              |
| *x*               | list of scalar(s) and/or *None*(s) | if known, value(s) of FoKL model input variables to include in Pyomo model (e.g., *x=[0.7, None, 0.4]*) | *[None, ..., None]* |
| *ReturnObjective* | boolean                            | to set the FoKL model as the Pyomo model's objective (i.e., minimize error in FoKL model evaluation)    | *False*             |

| Output | Type        | Description                          |
|--------|-------------|--------------------------------------|
| *m*    | Pyomo model | Pyomo model with FoKL model included |

| Objects of Pyomo Model | Type             | Description                                             |
|------------------------|------------------|---------------------------------------------------------|
| *m.y*                  | variable         | FoKL model output variable                              |
| *m.x*                  | set of variables | FoKL model input variables                              |
| *m.fokl*               | expression       | FoKL model equation *f(x0, x1, ... xM)*                 |
| *m.con*                | constraint       | if *ReturnObjective=False*, *m.y = m.fokl*              |
| *m.obj*                | objective        | if *ReturnObjective=True*, minimize *abs(m.fokl - m.y)* |

#### save

```
filepath = self.save(self, filename=None, directory=None)
```

Save a FoKL class as a file. By default, *filename* is of the form 'model_yyyymmddhhmmss.fokl' and is saved to the
directory of the Python script calling this method. Use *directory* to change the directory saved to, or simply
embed the directory manually within *filename*. Note that the directory must exist prior to calling this method.

Returned is *filepath*. Enter this as the argument to [*load*](#foklroutinesload) to later reload the model. Explicitly, that is:
```
FoKLRoutines.load(filepath)  # == FoKLRoutines.load(filename, directory)
```

| Input       | Type   | Description                                                                                                                      |
|-------------|--------|----------------------------------------------------------------------------------------------------------------------------------|
| *filename*  | string | name of file to save model as (note '.fokl' extension can be automatically or manually appended)                                 |
| *directory* | string | absolute path to pre-existing folder in which to contain *filename*, or path relative to directory of script calling this method |

| Output     | Type   | Description                               |
|------------|--------|-------------------------------------------|
| *filepath* | string | absolute path to where the file was saved |

### GP_integrate

```
T, Y = GP_Integrate(betas, matrix, b, norms, phis, start, stop, y0, h, used_inputs)
```

| Input         | Description                                                                                                                                                                                                                                                                                                                                                                      |
|---------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *betas*       | *betas* is a list of arrays in which each entry to the list contains a specific row of the betas matrix, or the mean of the betas matrix for each model being integrated.                                                                                                                                                                                                        |
| *matrix*      | *matrix* is a list of arrays containing the interaction matrix of each model.                                                                                                                                                                                                                                                                                                    |
| *b*           | *b* is an array of the values of all the other inputs to the model(s) (including any forcing functions) over the time period we integrate over. The length of b should be equal to the number of points in the final time series (end-start)/h. All values in b need to be normalized with respect to the min and max values of their respective values in the training dataset. |
| *norms*       | *norms* is a matrix of the min and max values of all the inputs being integrated (in the same order as y0). Min values are in the top row, max values in the bottom.                                                                                                                                                                                                             |
| *phis*        | *phis* is a data structure with coefficients for basis functions.                                                                                                                                                                                                                                                                                                                |
| *start*       | *start* is the time at which integration begins.                                                                                                                                                                                                                                                                                                                                 |
| *stop*        | *stop* is the time to end integration.                                                                                                                                                                                                                                                                                                                                           |
| *y0*          | *y0* is an array of the inital conditions for the models being integrated.                                                                                                                                                                                                                                                                                                       |
| *h*           | *h* is the step size with respect to time.                                                                                                                                                                                                                                                                                                                                       |
| *used_inputs* | *used_inputs* is a list of arrays containing the information as to what inputs are used in what model. Each array should contain a vector corresponding to a different model. Inputs should be referred to as those being integrated first, followed by those contained in b (in the same order as they appear in y0 and b respectively).                                        |

| Output | Description                                                                                |
|--------|--------------------------------------------------------------------------------------------|
| *T*    | *T* is an array of the time steps the models are integrated at.                            |
| *Y*    | *Y* is an array of the models that have been integrated, at the time steps contained in T. |

For example, if two models were being integrated, with 3 other inputs total
and the 1st model used both models outputs as inputs and the 1st and 3rd additional
inputs, while the 2nd model used its own output as an input and the 2nd
and 3rd additional inputs,
```
used_inputs = [[1, 1, 1, 0, 1], [0, 1, 0, 1, 0]]
```
If the models created do not follow this ordering scheme for their inputs,
the inputs can be rearranged based upon an alternate
numbering scheme provided to *used_inputs*. E.g., if the inputs need to be reordered then the 1st input should have a '1' in its place in the
*used_inputs* vector, the 2nd input should have a '2' and so on. Using the
same example as before, if the 1st models inputs needed rearranged so that
the 3rd additional input came first, followed by the two model outputs in
the same order as they are in *y0*, and ends with the 1st additional input,
then the 1st cell in *used_inputs* would have the form [2, 3, 4, 0, 1].

## Benchmarks and Papers

As mentioned in [About FoKL](#about-fokl), the primary advantage offered by FoKL in comparison to other machine learning packages 
is a significant decrease in computation time for training a model while not experiencing a significant decrease in 
accuracy. This holds true for most datasets but especially for those with an underlying static or dynamic relationship
as is often the case in any physical science experiment.

The following paper outlines the methodology of FoKL and includes two example problems.
- [Fast variable selection makes Karhunen-Loève
decomposed Gaussian process BSS-ANOVA a speedy
and accurate choice for dynamic systems
identification](docs/_static/arXiv.2205.13676v2.pdf)

The two example problems are:
- ‘Susceptible, Infected, Recovered’ (SIR) toy problem
- ‘Cascaded Tanks’ experimental dataset for a benchmark

## Future Development

FoKL-GPy is actively in development. Current focus is on:
- Pyomo
- optimization of code and integration with faster C++ routines
- adding examples for better comparisons and benchmarks
- more robust tutorials

Please reach out via the information in the [Contact Us](#contact-us) section with any suggestions for development.

## Contact Us

| Topic                                            | Point of Contact | Email                                                  |
|--------------------------------------------------|------------------|--------------------------------------------------------|
| Installation<br/>Troubleshooting<br/>Development | Jacob Krell      | [jpk0024@mix.wvu.edu](jpk0024@mix.wvu.edu)             |
| Research<br/>Theory<br/>Other                    | David Mebane     | [david.mebane@mail.wvu.edu](david.mebane@mail.wvu.edu) |

## License

FoKL-GPy has an MIT license. Please see the [LICENSE](LICENSE) file.

## Citations

Please cite: K. Hayes, M.W. Fouts, A. Baheri and
D.S. Mebane, "Forward variable selection enables fast and accurate
dynamic system identification with Karhunen-Loève decomposed Gaussian
processes", arXiv:2205.13676

Credits: David Mebane (ideas and original code), Kyle Hayes
(integrator), Derek Slack (Python porting), Jacob Krell (Python v3 dev.)

Funding provided by National Science Foundation, Award No. 2119688
