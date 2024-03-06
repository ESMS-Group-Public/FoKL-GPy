[//]: # (![FoKL-GPy Logo]&#40;https://github.com/ESMS-Group-Public/FoKL-GPy/docs/_static/esms_logo.png&#41;)
![FoKL-GPy Logo](https://github.com/jakekrell/FoKL-GPy/tree/debug_bernoulli/docs/_static/esms_logo.png)

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
- [Citations](#citations)

<!-- tocstop -->

## About FoKL

FoKL-GPy, or FoKL, is a Python package intended for use in machine learning. The name comes from a unique implementation 
of **Fo**rward variable selection using **K**arhunen-**L**oève decomposed **G**aussian **P**rocesses (GP's) in 
**Py**thon (i.e., **FoKL-GPy**). 

The primary advantages of FoKL are:
- Fast inference on static and dynamic datasets using scalable GP regression
- Significant accuracy retained compared to neural networks and other GP's despite being faster

Some other advantages of FoKL include:
- Export modeled non-linear dynamics as a symbolic equation
- Take first and second derivatives of model with respect to any input variable (e.g., gradient)
- Multiple kernels available (BSS-ANOVA, Orthonormal Bernoulli Polynomials)
- Automatic normalization and formatting of dataset
- Automatically split specified fraction of dataset into a training set and retain the rest for testing
- Easy adjusting of hyperparameters for sweeping through variations in order to find optimal settings
- Ability to save, share, and load models
- Ability to evaluate a model without known data

## Installation and Setup

FoKL is available through PyPI.

```cmd
pip install FoKL
```

Alternatively, the GitHub repository may be cloned to create a local copy in which the examples and documentation will 
be included.

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

Now you may call methods on the class and reference its attributes! Please see [Use Cases](#use-cases) for examples.

## Use Cases

Please refer to within the below examples for more detailed documentation:
- [Training and/or evaluating a model](docs/tutorials/clean.py)
- [Saving and loading a model](docs/tutorials/save_and_load/save_and_load.py)
- [Converting to Pyomo](docs/tutorials/fokl_to_pyomo.py)
- [Plotting and RMSE](examples/Sigmoid/sigmoid.py)
- [Integration](examples/GP Integrate/gp_integrate.py)

## Package Documentation

Below is a collection of the functions and methods that comprise the FoKL-GPy package.

Included are:
  - [load](#load)
  - [FoKL](#fokl)
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

### load

```
self = FoKLRoutines.load(filename, directory=None)
```

Load a FoKL class from a file.

By default, *directory* is the current working directory that contains the script calling this method. An absolute or 
relative directory may be defined if the model to load is located elsewhere.

For simplicity, enter the returned output from *self.save()* as the argument here, i.e., for *filename*. Do this while 
leaving *directory* blank since *filename* can simply include the directory itself.

### FoKL

```
self = FoKLRoutines.FoKL(**kwargs)
```

This creates a class object that contains all information relevant to and defining a FoKL model.

Upon initialization, hyperparameters and some other settings are defined with default values as attributes of the FoKL 
class. These attributes are as follows, and any or all may be specified as a keyword or later updated by redefining the 
class's attribute(s).

| Type    | Keyword Argument | Default Value   | Description                                                                                                               |
|---------|------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------|
| hyper   | kernel           | 'Cubic Splines' | Basis functions (i.e., kernel) to use for training a model                                                                |
| hyper   | phis             | f(kernel)       | Data structure with coefficients for basis functions                                                                      |
| hyper   | relats_in        | []              | Boolean matrix indicating which input variables and/or interactions should be excluded from the model                     |
| hyper   | a                | 4               | Shape parameter of the initial-guess distribution for the observation error variance of the data                          |
| hyper   | b                | f(a, data)      | Scale parameter of the initial-guess distribution for the observation error variance of the data                          |
| hyper   | atau             | 4               | Parameter of the initial-guess distribution for the $tau^2$ parameter                                                     |
| hyper   | btau             | f(atau, data)   | Parameter of the initial-guess distribution for the $tau^2$ parameter                                                     |
| hyper   | tolerance        | 3               | Influences how long to continue training after additional terms yield diminishing returns                                 |
| hyper   | draws            | 1000            | Total number of draws from the posterior for each tested model                                                            |
| hyper   | gimmie           | False           | Boolean to return the most complex model tried instead of the model with the optimum Bayesian information criterion (BIC) |
| hyper   | way3             | False           | Boolean to include three-way interactions                                                                                 |
| hyper   | threshav         | 0.05            | Threshold to propose terms for elimination. Increase to propose and eliminate more terms                                  |
| hyper   | threshstda       | 0.5             | Threshold to eliminate terms based on standard deviation relative to mean                                                 |
| hyper   | threshstdb       | 2               | Threshold to eliminate terms based on standard deviation independent of mean                                              |
| hyper   | aic              | False           | Boolean to use Aikaike information criterion (AIC)                                                                        |
| setting | UserWarnings     | True            | Boolean to print user-warnings (i.e., FoKL warnings) to command terminal                                                  |
| setting | ConsoleOutput    | True            | Boolean to print progress of model training to command terminal                                                           |

The following methods are embedded within the class object:

| Method                                  | Description                                                                                    |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
| [**clean**](#clean)                     | Automatically format, normalize, and create test/train sets from user's provided dataset.      |
| [**bss_derivatives**](#bss_derivatives) | Algebraically calculate partial derivatives of model with respect to input variables.          |
| [**evaluate_basis**](#evaluate_basis)   | Calculate value of specified basis function at single point along normalized domain.           |
| [**evaluate**](#evaluate)               | Calculate values of FoKL model for all requested sets of datapoints.                           |
| [**coverage3**](#coverage3)             | Evaluate FoKL model, calculate confidence bounds, calculate RMSE, and produce plot.            |
| [**fit**](#fit)                         | Train new FoKL model to best-fit training dataset according to hyperparameters.                |
| [**clear**](#clear)                     | Delete attributes from FoKL class so that new models may be trained without new class objects. |
| [**to_pyomo**](#to_pyomo)               | Convert a FoKL model to an expression in a Pyomo model.                                        |
| [**save**](#save)                       | Save FoKL class with all its attributes to retain model and avoid re-training.                 |

Each method has optional inputs that allow for flexibility in how FoKL is used so that you may leverage these methods 
for your specific requirements. Please refer to the [Use Cases](#use-cases) first, then explore the following documentation of 
each method as needed.

#### clean

```
self.clean(self, inputs, data=None, **kwargs)
```

For cleaning and formatting inputs prior to training a FoKL model. Note that data is not required but should be entered 
if available; otherwise, leave blank.

Inputs:
    inputs == NxM matrix of independent (or non-linearly dependent) 'x' variables for fitting f(x1, ..., xM)
    data   == Nx1 vector of dependent variable to create model for predicting the value of f(x1, ..., xM)

Keyword Inputs:
    train == percentage (0-1) of N datapoints to use for training == 1 (default)

Added Attributes:
    > 'inputs' and 'data' get automatically formatted, cleaned, reduced to a train set, and stored as:
        > model.inputs         == all normalized inputs w/o outliers (i.e., model.traininputs plus
                                  model.testinputs)
        > model.data           == all data w/o outliers (i.e., model.traindata plus model.testdata)
    > other useful info related to 'inputs' and 'data' get stored as:
        > model.rawinputs      == all normalized inputs w/ outliers == user's 'inputs' but normalized and
                                                                       formatted
        > model.rawdata        == all data w/ outliers              == user's 'data' but formatted
        > model.traininputs    == train set of model.inputs
        > model.traindata      == train set of model.data
        > model.testinputs     == test set of model.inputs
        > model.testdata       == test set of model.data
        > model.normalize      == [min, max] factors used to normalize user's 'inputs' to 0-1 scale of
                                  model.rawinputs
        > model.outliers       == indices removed from model.rawinputs and model.rawdata as outliers
        > model.trainlog       == indices of model.inputs used for model.traininputs
        > model.testlog        == indices of model.data used for model.traindata
    > to access numpy versions of the above attributes related to 'inputs', use:
        > model.inputs_np      == model.inputs as a numpy array of timestamps x input variables
        > model.rawinputs_np   == model.rawinputs as a numpy array of timestamps x input variables
        > model.traininputs_np == model.traininputs as a numpy array of timestamps x input variables
        > model.testinputs_np  == model.testinputs as a numpy array of timestamps x input variables

#### bss_derivatives

```
dState = self.bss_derivatives(self, **kwargs)
```

For returning gradient of modeled function with respect to each, or specified, input variable.
If user overrides default settings, then 1st and 2nd partial derivatives can be returned for any variables.

Keyword Inputs:
    inputs == NxM matrix of 'x' input variables for fitting f(x1, ..., xM)    == self.inputs_np (default)
    kernel == function to use for differentiation (i.e., cubic or Bernoulli)  == self.kernel (default)
    d1     == index of input variable(s) to use for first partial derivative  == True (default)
    d2     == index of input variable(s) to use for second partial derivative == False (default)
    draws  == number of beta terms used                                       == self.draws (default)
    betas  == draw from the posterior distribution of coefficients            == self.betas (default)
    phis   == spline coefficients for the basis functions                     == self.phis (default)
    mtx    == basis function interaction matrix from the best model           == self.mtx (default)
    span   == list of [min, max]'s of input data used in the normalization    == self.normalize (default)
    IndividualDraws == boolean for returning derivative(s) at each draw       == 0 (default)
    ReturnFullArray == boolean for returning NxMx2 array instead of Nx2M      == 0 (default)

Return Outputs:
    dState == derivative of input variables (i.e., states)

Notes:
    - To turn off all the first-derivatives, set d1=False instead of d1=0. 'd1' and 'd2', if set to an integer,
    will return the derivative with respect to the input variable indexed by that integer using Python indexing.
    In other words, for a two-input FoKL model, setting d1=1 and d2=0 will return the first-derivative with
    respect to the second input (d1=1) and the second-derivative with respect to the first input (d2=0).
    Alternatively, d1=[False, True] and d2=[True, False] will function the same.

#### evaluate_basis

```
basis = self.evaluate_basis(self, c, x, kernel=None, d=0)
```

Evaluate a basis function at a single point by providing coefficients, x value(s), and (optionally) the kernel.

Inputs:
    > c == coefficients of a single basis functions
    > x == value(s) of independent variable at which to evaluate the basis function

Optional Input:
    > kernel == 'Cubic Splines' or 'Bernoulli Polynomials' == self.kernel (default)
    > d      == integer representing order of derivative   == 0 (default)

Output (in Python syntax, for d=0):
    > if kernel == 'Cubic Splines':
        > basis = c[0] + c[1]*x + c[2]*(x**2) + c[3]*(x**3)
    > if kernel == 'Bernoulli Polynomials':
        > basis = sum(c[k]*(x**k) for k in range(len(c)))

#### evaluate

```
meen = self.evaluate(self, inputs=None, betas=None, mtx=None, **kwargs)
```

Evaluate the FoKL model for provided inputs and (optionally) calculate bounds. Note 'evaluate_fokl' may be a
more accurate name so as not to confuse this function with 'evaluate_basis', but it is unexpected for a user to
call 'evaluate_basis' so this function is simply named 'evaluate'.

Input:
    inputs == input variable(s) at which to evaluate the FoKL model == self.inputs_np (default)

Optional Inputs:
    betas        == coefficients defining FoKL model                       == self.betas (default)
    mtx          == interaction matrix defining FoKL model                 == self.mtx (default)
    normalize    == [min, max] of inputs used for normalization            == self.normalize (default)
    draws        == number of beta terms used                              == self.draws (default)
    clean        == boolean to automatically normalize and format 'inputs' == False (default)
    ReturnBounds == boolean to return confidence bounds as second output   == False (default)

#### coverage3

```
meen, bounds, rmse = self.coverage3(self, **kwargs)
```

For validation testing of FoKL model. Default functionality is to evaluate all inputs (i.e., train+test sets).
Returned is the predicted output 'meen', confidence bounds 'bounds', and root mean square error 'rmse'. A plot
may be returned by calling 'coverage3(plot=1)'; or, for a potentially more meaningful plot in terms of judging
accuracy, 'coverage3(plot='sorted')' plots the data in increasing value.

Optional inputs for numerical evaluation of model:
    inputs == normalized and properly formatted inputs to evaluate              == self.inputs (default)
    data   == properly formatted data outputs to use for validating predictions == self.data (default)
    draws  == number of beta terms used                                         == self.draws (default)

Optional inputs for basic plot controls:
    plot              == binary for generating plot, or 'sorted' for plot of ordered data == False (default)
    bounds            == binary for plotting bounds                                       == True (default)
    xaxis             == integer indexing the input variable to plot along the x-axis     == indices (default)
    labels            == binary for adding labels to plot                                 == True (default)
    xlabel            == string for x-axis label                                          == 'Index' (default)
    ylabel            == string for y-axis label                                          == 'Data' (default)
    title             == string for plot title                                            == 'FoKL' (default)
    legend            == binary for adding legend to plot                                 == True (default)
    LegendLabelFoKL   == string for FoKL's label in legend                                == 'FoKL' (default)
    LegendLabelData   == string for Data's label in legend                                == 'Data' (default)
    LegendLabelBounds == string for Bounds's label in legend                              == 'Bounds' (default)

Optional inputs for detailed plot controls:
    PlotTypeFoKL   == string for FoKL's color and line type  == 'b' (default)
    PlotSizeFoKL   == scalar for FoKL's line size            == 2 (default)
    PlotTypeBounds == string for Bounds' color and line type == 'k--' (default)
    PlotSizeBounds == scalar for Bounds' line size           == 2 (default)
    PlotTypeData   == string for Data's color and line type  == 'ro' (default)
    PlotSizeData   == scalar for Data's line size            == 2 (default)

Return Outputs:
    meen   == predicted output values for each indexed input
    bounds == confidence interval for each predicted output value
    rmse   == root mean squared deviation (RMSE) of prediction versus known data

#### fit

```
betas, mtx, evs = self.fit(self, inputs=None, data=None, **kwargs)
```

For fitting model to known inputs and data (i.e., training of model).

Inputs:
    inputs == NxM matrix of independent (or non-linearly dependent) 'x' variables for fitting f(x1, ..., xM)
    data   == Nx1 vector of dependent variable to create model for predicting the value of f(x1, ..., xM)

Keyword Inputs (for fit):
    clean         == boolean to perform automatic cleaning and formatting               == False (default)
    ConsoleOutput == boolean to print [ind, ev] to console during FoKL model generation == True (default)

Keyword Inputs (for clean):
    train                == percentage (0-1) of N datapoints to use for training  == 1 (default)
    TrainMethod          == method for splitting test/train set for train < 1     == 'random' (default)
    CatchOutliers        == boolean for removing outliers from inputs and/or data == False (default)
    OutliersMethod       == string defining the method to use for removing outliers (e.g., 'Z-Score)
    OutliersMethodParams == parameters to modify OutliersMethod (format varies per method)

Return Outputs:
    'betas' are a draw from the posterior distribution of coefficients: matrix, with
    rows corresponding to draws and columns corresponding to terms in the GP

    'mtx' is the basis function interaction matrix from the
    best model: matrix, with rows corresponding to terms in the GP (and thus to the
    columns of 'betas' and columns corresponding to inputs. a given entry in the
    matrix gives the order of the basis function appearing in a given term in the GP.
    all basis functions indicated on a given row are multiplied together.
    a zero indicates no basis function from a given input is present in a given term

    'ev' is a vector of BIC values from all of the models
    evaluated

Added Attributes:
    - Various ... please see description of 'clean()'

#### clear

```
self.clear(self, keep=None, clear=None, all=False)
```

Delete all attributes from the FoKL class except for hyperparameters by default, but user may specify otherwise.
If an attribute is listed in both 'clear' and 'keep', then the attribute is cleared.

Optional Inputs:
    keep (list of strings)  == attributes to keep in addition to hyperparameters, e.g., ['inputs_np', 'mtx']
    clear (list of strings) == hyperparameters to delete, e.g., ['kernel', 'phis']
    all (boolean)           == if True then all attributes (including hyperparameters) get deleted regardless

Tip: To remove all attributes, simply call 'self.clear(all=1)'.

#### to_pyomo

```
m = self.to_pyomo(self, m=None, y=None, x=None, ReturnObjective=False)
```

Automatically convert a pre-trained FoKL model to expressions and constraints for a symbolic Pyomo model. Note
that by default, the Pyomo model's objective does not get defined here but can be overridden with
ReturnObjective=True.

Optional Inputs:
    - m               == Pyomo model (if already defined)
    - y               == FoKL output to include in Pyomo model (if known)
    - x               == FoKL input variables to include in Pyomo model (if known), e.g., x=[0.7, None, 0.4]
    - ReturnObjective == boolean to set the FoKL model as the Pyomo model's objective      == 0 (default)
    - TruncateBasis   == integer (3, lp-1) to decrease the resolution of the cubic splines == 0 (default)

Output:
    - m == Pyomo model with FoKL model included
        - m.y    == evaluated output corresponding to FoKL model
        - m.x[j] == input variable corresponding to FoKL model

Note:
    - It is highly recomme nded to use a FoKL model trained on the 'Bernoulli Polynomials' kernel. Otherwise, with
    'Cubic Splines', the solution time is extremely impractical even for the simplest of models.

#### save

```
filepath = self.save(self, filename=None, directory=None)
```

Save a FoKL class as a file. By default, the 'filename' is 'model_yyyymmddhhmmss.fokl' and is saved to the
directory of the Python script calling this method. Use 'directory' to change the directory saved to, or simply
embed the directory manually within 'filename'.

Returned is the 'filepath'. Enter this as the argument of 'load' to later reload the model. Explicitly, that is
'FoKLRoutines.load(filepath)' or 'FoKLRoutines.load(filename, directory)'.

Note the directory must exist prior to calling this method.

### GP_integrate

```
T, Y = GP_Integrate(betas, matrix, b, norms, phis, start, stop, y0, h, used_inputs)
```

betas is a list of arrays in which each entry to the list contains a specific row of the betas matrix,
or the mean of the the betas matrix for each model being integrated

matrix is a list of arrays containing the interaction matrix of each model

b is an array of of the values of all the other inputs to the model(s) (including
any forcing functions) over the time period we integrate over. The length of b
should be equal to the number of points in the final time series (end-start)/h
All values in b need to be normalized with respect to the min and max values
of their respective values in the training dataset

h is the step size with respect to time

norms is a matrix of the min and max values of all the inputs being
integrated (in the same order as y0). min values are in the top row, max values in the bottom.

Start is the time at which integration begins. Stop is the time to
end integration.

y0 is an array of the inital conditions for the models being integrated

Used inputs is a list of arrays containing the information as to what inputs
are used in what model. Each array should contain a vector corresponding to a different model.
Inputs should be referred to as those being integrated first, followed by
those contained in b (in the same order as they appear in y0 and b
respectively)
For example, if two models were being integrated, with 3 other inputs total
and the 1st model used both models outputs as inputs and the 1st and 3rd additional
inputs, while the 2nd model used its own output as an input and the 2nd
and 3rd additional inputs, used_inputs would be equal to
[[1,1,1,0,1],[0,1,0,1,0]].
If the models created do not follow this ordering scheme for their inputs
the inputs can be rearranged based upon an alternate
numbering scheme provided to used_inputs. E.g. if the inputs need to breordered the the 1st input should have a '1' in its place in the
used_inputs vector, the 2nd input should have a '2' and so on. Using the
same example as before, if the 1st models inputs needed rearranged so that
the 3rd additional input came first, followed by the two model outputs in
the same order as they are in y0, and ends with the 1st additional input,
then the 1st cell in used_inputs would have the form [2,3,4,0,1]

T an array of the time steps the models are integrated at.

Y is an array of the models that have been integrated, at the time steps
contained in T.

## Benchmarks and Papers

As mentioned in [About FoKL](#about-fokl), the primary advantage offered by FoKL in comparison to other machine learning packages 
is a significant decrease in computation time for training a model while not experiencing a significant decrease in 
accuracy. This holds true for most datasets but especially for those with an underlying static or dynamic pattern 
typically encountered in the physical sciences.

The following paper outlines the methodology of FoKL and includes two example problems.
- [https://arxiv.org/pdf/2205.13676v2.pdf](https://arxiv.org/pdf/2205.13676v2.pdf)

The two example problems are:
- ‘Susceptible, Infected, Recovered’ (SIR) toy problem
- ‘Cascaded Tanks’ experimental dataset for a benchmark

## Future Development

FoKL-GPy is actively in development. Current focus is on:
- Pyomo
- optimization of code and integration with faster C++ routines
- adding examples for better comparisons and benchmarks

Please reach out via the information in the [Contact Us](#contact-us) section with any suggestions for development.

## Contact Us

Please reach out with any inquiries!

| Topic                                            | Point of Contact | Email                     |
|--------------------------------------------------|------------------|---------------------------|
| Installation<br/>Troubleshooting<br/>Development | Jacob Krell      | jpk0024@mix.wvu.edu       |
| Research<br/>Theory<br/>Other                    | David Mebane     | david.mebane@mail.wvu.edu |

## Citations

Please cite: K. Hayes, M.W. Fouts, A. Baheri and
D.S. Mebane, "Forward variable selection enables fast and accurate
dynamic system identification with Karhunen-Loève decomposed Gaussian
processes", arXiv:2205.13676

Credits: David Mebane (ideas and original code), Kyle Hayes
(integrator), Derek Slack (Python porting), Jacob Krell (Python v3 dev.)

Funding provided by National Science Foundation, Award No. 2119688
