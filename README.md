![FoKL-GPy Logo](https://github.com/ESMS-Group-Public/FoKL-GPy/docs/_static/esms_logo.png)

--------------------------------------------------------------------------------

## Contents
<!-- toc -->

- [About FoKL](#about-fokl)
- [Getting Started]()
  - [Installation](#)
  - [Setup]()
- [Use Cases]()
  - [Saving and loading a model]()
  - [Evaluating a model]()
  - [Taking partial derivatives of a model]()
  - [Integration]()
  - [Training a new model]()
  - [Training several new models]()
  - [Sweeping over hyperparameters]()
  - [Coverting to Pyomo]()
- [Documentation]()
  - [FoKLRoutines.load]()
  - [FoKLRoutines.FoKL]()
    - [self.clean]()
    - [self.bss_derivatives]()
    - [self.evaluate_basis]()
    - [self.evaluate]()
    - [self.coverage3]()
    - [self.fit]()
    - [self.clear]()
    - [self.to_pyomo]()
    - [self.save]()
  - [GP_integrate]()
- [Comparisons and Benchmarks]()
- [Future Development]()
- [Citations]()
- [License]()

<!-- tocstop -->

## About FoKL

FoKL-GPy, or FoKL, is a Python package intended for use in machine learning. The name comes from a unique implementation 
of **Fo**rward variable selection using **K**arhunen-**L**oève decomposed **G**aussian **P**rocesses (GP's) in 
**Py**thon. 

Advantages of FoKL are:
- Fast inference on static and dynamic datasets using scalable GP regression
- Accurate modeling of non-linear dynamics
- Get model as a symbolic equation
- Multiple kernels available
- User-friendliness allows for automatic normalization, test/train splits, etc.
- Easy adjusting of hyperparameters for sweeping during optimization
- Ability to save, share, and load models
- Ability to evaluate a user-proposed model without requiring training data

### Pyomo and Symbolic Models

For applications where it is useful to deal analytically with a model, FoKL may return a model as an algebraic equation 
through the use of Pyomo. Further, additional Pyomo constraints may be added and a nonlinear optimizer such as IPOPT may
be used to solve for variables in the FoKL model expression.

## Getting Started

## Installation

FoKL is available through PyPI.

```cmd
pip install FoKL
```

Alternatively, the GitHub repository may be cloned to create a local copy in which the examples and documentation will 
be included.

```cmd
git clone https://github.com/ESMS-Group-Public/FoKL-GPy
```

### Setup

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

Now you may call methods on the class and reference its attributes! Please see [Use Cases]() for examples.

## Use Cases

Please refer to within the below examples for more detailed documentation:
- [Saving and loading a model]()
- [Evaluating a model]()
- [Taking partial derivatives of a model]()
- [Integration]()
- [Training a new model]()
- [Training several new models]()
- [Sweeping over hyperparameters]()
- [Coverting to Pyomo]()

## Documentation

### FoKLRoutines.load(filename, directory=None)

### FoKLRoutines.FoKL(**kwargs)

This creates a class object that contains all information relevant to and defining a FoKL model.

Upon initialization, hyperparameters and some other settings are defined with default values as attributes of the FoKL 
class. These attributes are as follows, and any or all may be specified as a keyword or later updated by redefining the 
class's attribute(s).

| Type             | Keyword Argument | Default Value   | Description                                                                                                               |
|------------------|------------------|-----------------|---------------------------------------------------------------------------------------------------------------------------|
| Hyperparameter   | kernel           | 'Cubic Splines' | Basis functions (i.e., kernel) to use for training a model                                                                |
| Hyperparameter   | phis             | f(kernel)       | Data structure with coefficients for basis functions                                                                      |
| Hyperparameter   | relats_in        | []              | Boolean matrix indicating which input variables and/or interactions should be excluded from the model                     |
| Hyperparameter   | a                | 4               | Shape parameter of the initial-guess distribution for the observation error variance of the data                          |
| Hyperparameter   | b                | f(a, data)      | Scale parameter of the initial-guess distribution for the observation error variance of the data                          |
| Hyperparameter   | atau             | 4               | Parameter of the initial-guess distribution for the $tau^2$ parameter                                                     |
| Hyperparameter   | btau             | f(atau, data)   | Parameter of the initial-guess distribution for the $tau^2$ parameter                                                     |
| Hyperparameter   | tolerance        | 3               | Influences how long to continue training after additional terms yield diminishing returns                                 |
| Hyperparameter   | draws            | 1000            | Total number of draws from the posterior for each tested model                                                            |
| Hyperparameter   | gimmie           | False           | Boolean to return the most complex model tried instead of the model with the optimum Bayesian information criterion (BIC) |
| Hyperparameter   | way3             | False           | Boolean to include three-way interactions                                                                                 |
| Hyperparameter   | threshav         | 0.05            | Threshold to propose terms for elimination. Increase to propose and eliminate more terms                                  |
| Hyperparameter   | threshstda       | 0.5             | Threshold to eliminate terms based on standard deviation relative to mean                                                 |
| Hyperparameter   | threshstdb       | 2               | Threshold to eliminate terms based on standard deviation independent of mean                                              |
| Hyperparameter   | aic              | False           | Boolean to use Aikaike information criterion (AIC)                                                                        |
| User-Setting     | UserWarnings     | True            | Boolean to print user-warnings (i.e., FoKL warnings) to command terminal                                                  |
| User-Setting     | ConsoleOutput    | True            | Boolean to print progress of model training to command terminal                                                           |

The following methods are embedded within the class object:

| Method                                                                | Description                                                                                    |
|-----------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| [**clean**](https://pytorch.org/docs/stable/torch.html)               | Automatically format, normalize, and create test/train sets from user's provided dataset.      |
| [**bss_derivatives**](https://pytorch.org/docs/stable/autograd.html)  | Algebraically calculate partial derivatives of model with respect to input variables.          |
| [**evaluate_basis**](https://pytorch.org/docs/stable/jit.html)        | Calculate value of specified basis function at single point along normalized domain.           |
| [**evaluate**](https://pytorch.org/docs/stable/nn.html)               | Calculate values of FoKL model for all requested sets of datapoints.                           |
| [**coverage3**](https://pytorch.org/docs/stable/multiprocessing.html) | Evaluate FoKL model, calculate confidence bounds, calculate RMSE, and produce plot.            |
| [**fit**](https://pytorch.org/docs/stable/data.html)                  | Train new FoKL model to best-fit training dataset according to hyperparameters.                |
| [**clear**]()                                                         | Delete attributes from FoKL class so that new models may be trained without new class objects. |
| [**to_pyomo**]()                                                      | Convert a FoKL model to an expression in a Pyomo model.                                        |
| [**save**]()                                                          | Save FoKL class with all its attributes to retain model and avoid re-training.                 |

Each method has optional inputs that allow for flexibility in how FoKL is used so that you may leverage these methods 
for your specific requirements. Please refer to the [Use Cases]() first, then explore the following documentation of 
each method as needed.

#### self.clean

#### self.bss_derivatives

#### self.evaluate_basis

#### self.evaluate

#### self.coverage3

#### self.fit

#### self.clear

#### self.to_pyomo

#### self.save

### Further Documentation: GP_integrate

## Comparisons and Benchmarks

FoKL outperforms neural nets on dynamic datasets which is expected for a GP, but also tends to train faster than other 
GP's by orders of magnitude. FoKL also tends to fit more accurately than other GP's.

Some papers on FoKL include:
- [link to paper]()
- [link to paper]()

NOTES TO SELF:
- https://arxiv.org/pdf/2205.13676v2.pdf
  - ‘Susceptible, Infected, Recovered’ (SIR) toy problem 
  - experimental ‘Cascaded Tanks’ benchmark dataset

## Future Development

More sophisticated outlier removal methods are currently in development, but for demonstration purposes the following 
will search through 'data' and remove any points with a z-score greater than 4:

```
model.fit(inputs, data, CatchOutliers='Data', OutliersMethod='Z-Score', OutliersMethodParams=4)
```

Also in development are additional methods for splitting 'data' into test/train sets, beyond the current method which is 
limited to a random split.

## Citations

Please cite: K. Hayes, M.W. Fouts, A. Baheri and
D.S. Mebane, "Forward variable selection enables fast and accurate
dynamic system identification with Karhunen-Loève decomposed Gaussian
processes", arXiv:2205.13676

Credits: David Mebane (ideas and original code), Kyle Hayes
(integrator), Derek Slack (Python porting), Jacob Krell (Python v3 dev.)

Funding provided by National Science Foundation, Award No. 2119688

## License


