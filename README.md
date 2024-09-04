x![FoKL-GPy Logo](docs/_static/fokl-gpy_banner.png)

--------------------------------------------------------------------------------

## Contents
<!-- toc -->

- [About FoKL](#about-fokl)
- [Installation and Setup](#installation-and-setup)
- [Use Cases](#use-cases)
- [User Documentation](#user-documentation)
- [Benchmarks and Papers](#benchmarks-and-papers)
- [Future Development](#future-development)
- [Contact Us](#contact-us)
- [License](#license)
- [Citations](#citations)

<!-- tocstop -->

## About FoKL

FoKL-GPy, or FoKL, is a Python package intended for use in machine learning. The name comes from a unique implementation 
of **Fo**rward variable selection using **K**arhunen-**L**oève decomposed **G**aussian **P**rocesses (GP's) in 
**Py**thon (i.e., **FoKL-GPy**). 

The primary advantages of FoKL are:
- Fast inference on static and dynamic datasets using scalable GP regression
- Significant accuracy retained despite being fast

Some other advantages of FoKL include:
- Export modeled non-linear dynamics as a symbolic equation (i.e., use a GP model in Pyomo)
- Take first and second derivatives of model with respect to any input variable (e.g., gradient)
- User-friendly (e.g., automatic handling of various dataset formats, automatic creation of training set, etc.)
- Easy adjusting of hyperparameters for sweeping through variations in order to find optimal settings
- Ability to save, share, and load models
- Ability to import and evaluate a model without known data (i.e., without training)

To read more about FoKL, please see the [Benchmarks and Papers](#benchmarks-and-papers) section.

## Installation and Setup

From your command-line terminal, FoKL is available through PyPI:

```cmd
pip install FoKL
```

Alternatively, the GitHub repository may be cloned to create a local copy in which the examples and documentation will 
be included:

```cmd
git clone https://github.com/ESMS-Group-Public/FoKL-GPy
```

Once installed, import the FoKL module in Python with:
```python
from FoKL import FoKLRoutines
```

From here, the FoKL class object may be created and its methods accessed. Please see [Use Cases](#use-cases) to learn more about working with FoKL models.

## Use Cases

Please first refer to the following for tutorials and examples:
- [Automatically formatting and normalizing datasets](docs/tutorials/clean.ipynb)
- [Training and/or evaluating a model](docs/tutorials/fit_and_evaluate.py)
- [Saving and/or loading a model](docs/tutorials/save_and_load/save_and_load.py)
- [Validating model via plot and RMSE](examples/sigmoid/sigmoid.py)
- [Integrating models of derivatives](examples/gp_integrate/gp_integrate.py)
- [Converting multiple models to Pyomo](examples/pyomo_multiple_models/pyomo_multiple_models.py)
- [Solving model in Pyomo with non-linear optimization](examples/pyomo_maximize/pyomo_maximize.py)

Then, see [User Documentation](#user-documentation) as needed.

## User Documentation

  - [FoKLRoutines](#foklroutines)
    - [load](#load)
    - [FoKL](#fokl)
      - [clean](#clean)
      - [generate_trainlog](#generate_trainlog)
      - [trainset](#trainset)
      - [bss_derivatives](#bss_derivatives)
      - [evaluate_basis](#evaluate_basis)
      - [evaluate](#evaluate)
      - [coverage3](#coverage3)
      - [fit](#fit)
      - [clear](#clear)
      - [to_pyomo](#to_pyomo)
      - [save](#save)
  - [fokl_to_pyomo](#fokl_to_pyomo)
  - [getKernels](#getkernels)
  - [GP_integrate](#gp_integrate)

### FoKLRoutines

The [FoKLRoutines](#foklroutines) module houses the primary routines for a FoKL model. Namely, these are the [load](#load) function and [FoKL](#fokl) class object.

#### load

```python
model = FoKLRoutines.load(filename, directory=None)
```

Load a FoKL class from a file. If failing to load a file and/or directory relative to the run script, ensure the terminal directory is set to that of the run script.

By default, ```directory``` is the current working directory that contains the script calling this method. An absolute or 
relative directory may be defined if the model to load is located elsewhere.

For simplicity, enter the returned output from [save](#save) as the argument here, i.e., for ```filename```. Do this while 
leaving ```directory``` blank since ```filename``` can simply include the directory itself.

#### FoKL

```python
model = FoKLRoutines.FoKL(**kwargs)
```

This creates a class object that contains all information relevant to and defining a FoKL model.

Upon initialization, hyperparameters and some other settings are defined with default values as attributes of the FoKL 
class. These attributes are as follows, and any or all may be specified as a keyword or later updated by redefining the 
a class attribute.

| Type           | Keyword Argument    | Default Value                   | Description                                                                                                                              |
|----------------|---------------------|---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| hyperparameter | ```kernel```        | ```'Cubic Splines'```           | Format of basis functions from the BSS-ANOVA kernel to use for training a model (```'Cubic Splines'``` or ```'Bernoulli Polynomials'```) |
| "              | ```phis```          | $f($ ```kernel``` $)$           | Data structure with coefficients for basis functions                                                                                     |
| "              | ```relats_in```     | ```[]```                        | Boolean matrix indicating which input variables and/or interactions should be excluded from the model                                    |
| "              | ```a```             | ```4```                         | Shape parameter of the initial-guess distribution for the observation error variance of the data                                         |
| "              | ```b```             | $f($ ```a```, ```data``` $)$    | Scale parameter of the initial-guess distribution for the observation error variance of the data                                         |
| "              | ```atau```          | ```4```                         | Parameter of the initial-guess distribution for the $\tau^2$ parameter                                                                   |
| "              | ```btau```          | $f($ ```atau```, ```data``` $)$ | Parameter of the initial-guess distribution for the $\tau^2$ parameter                                                                   |
| "              | ```tolerance```     | ```3```                         | Influences how long to continue training after additional terms yield diminishing returns                                                |
| "              | ```burnin```        | ```1000```                      | Total number of draws from the posterior for each tested model before the ```draws``` draws                                              |
| "              | ```draws```         | ```1000```                      | Total number of draws from the posterior for each tested model after the ```burnin``` draws                                              |
| "              | ```gimmie```        | ```False```                     | Boolean to return the most complex model tried instead of the model with the optimum Bayesian information criterion (BIC)                |
| "              | ```way3```          | ```False```                     | Boolean to include three-way interactions                                                                                                |
| "              | ```threshav```      | ```0.05```                      | Threshold to propose terms for elimination. Increase to propose and eliminate more terms                                                 |
| "              | ```threshstda```    | ```0.5```                       | Threshold to eliminate terms based on standard deviation relative to mean                                                                |
| "              | ```threshstdb```    | ```2```                         | Threshold to eliminate terms based on standard deviation independent of mean                                                             |
| "              | ```aic```           | ```False```                     | Boolean to use Aikaike information criterion (AIC)                                                                                       |
| setting        | ```UserWarnings```  | ```True```                      | Boolean to print user-warnings (i.e., FoKL warnings) to command terminal                                                                 |
| "              | ```ConsoleOutput``` | ```True```                      | Boolean to print progress of model training to command terminal                                                                          |

The following methods are embedded within the class object:

| Method                                  | Description                                                                                    |
|-----------------------------------------|------------------------------------------------------------------------------------------------|
| [clean](#clean)                         | Automatically format and normalize user-provided dataset.                                      |
| [generate_trainlog](#generate_trainlog) | Generate random indices of the dataset to use as a training set.                               |
| [trainset](#trainset)                   | Return the training set.                                                                       |
| [bss_derivatives](#bss_derivatives)     | Algebraically calculate partial derivatives of model with respect to input variables.          |
| [evaluate_basis](#evaluate_basis)       | Calculate value of specified basis function at single point along normalized domain.           |
| [evaluate](#evaluate)                   | Calculate values of FoKL model for all requested sets of datapoints.                           |
| [coverage3](#coverage3)                 | Evaluate FoKL model, calculate confidence bounds, calculate RMSE, and produce plot.            |
| [fit](#fit)                             | Train new FoKL model to best-fit training dataset according to hyperparameters.                |
| [clear](#clear)                         | Delete attributes from FoKL class so that new models may be trained without new class objects. |
| [to_pyomo](#to_pyomo)                   | Convert a FoKL model to an expression in a Pyomo model.                                        |
| [save](#save)                           | Save FoKL class with all its attributes to retain model and avoid re-training.                 |

Each method has optional inputs that allow for flexibility in how FoKL is used so that you may leverage these methods 
for your specific requirements. Please refer to the [Use Cases](#use-cases) first, then explore the following documentation of 
each method as needed.

##### clean

```python
model.clean(inputs, data=None, **kwargs)
```

Automatically format and normalize datasets. Note that data is not required but should be entered 
if available; otherwise, leave blank. Multiple options are available to govern the normalization of inputs. See [Automatically formatting and normalizing datasets](#use-cases) for example usage.

| Input        | Type | Description                                                                                                                          | Default    |
|--------------|------|--------------------------------------------------------------------------------------------------------------------------------------|------------|
| ```inputs``` | any  | $n \times m$ input matrix $\mathbf{x}$ of $n$ instances by $m$ features in model $\overline{y}=f(\overline{x}_1,...,\overline{x}_m)$ | n/a        |
| ```data```   | any  | $n \times 1$ output vector $\overline{y}$ of $n$ instances in model $\overline{y}=f(\overline{x}_1,...,\overline{x}_m)$              | ```None``` |

| Keyword             | Type    | Description                                         | Default    |
|---------------------|---------|-----------------------------------------------------|------------|
| ```train```         | scalar  | (0,1] fraction of $n$ instances to use for training | ```1```    |
| ```AutoTranspose``` | boolean | assumes $n > m$ and transposes dataset accordingly  | ```True``` |
| ```bit```           | integer | (16, 32, 64) floating point bits to save dataset as | ```64```   |
| ```normalize```     | boolean | to pass formatted dataset to ```_normalize()```     | ```True``` |
| ```minmax```        | list of [min, max] lists | upper/lower bounds of each input variable | model.minmax |
| ```pillow```        | list of [lower, upper] lists | fraction of span by which to expand [min, max]; or, values on 0-1 scale that [min, max] should map to | ```0``` |

After calling [clean](#clean), the now normalized and formatted dataset gets saved as attributes of the FoKL class. Be sure to use these attributes in place of the originally entered ```inputs``` and ```data``` so that normalization and formatting errors are avoided. The attributes are as follows:

| Attribute             | Type                 | Description                                                             |
|-----------------------|----------------------|-------------------------------------------------------------------------|
| ```model.inputs```    | $n \times m$ ndarray | normalized and formatted ```inputs```                                   |
| ```model.data```      | $n \times 1$ ndarray | formatted ```data```                                                    |                                                            |
| ```model.minmax``` | list of $m$ lists    | [min, max] factors used to normalize ```inputs``` to ```model.inputs``` |
| ```model.trainlog```  | $n \times 1$ ndarray | logical index of instances from dataset to use as training set          |

To then access the training set ```[traininputs, traindata]```, see [trainset](#trainset).

##### generate_trainlog

```python
model.trainlog = model.generate_trainlog(train, n=None)
```

Generate random logical vector of length $n$ with ```train``` percent as ```True```. It is expected that [generate_trainlog](#generate_trainlog) will be called internally by [clean](#clean) and not by the user, though this method is available if sweeping through values of ```train``` in order to compare the accuracy of models fitted to training sets of different sizes.

##### trainset

```python
traininputs, traindata = model.trainset()
```

Run this line to access the training set, which is simply ```model.inputs``` and ```model.data``` indexed by ```model.trainlog```. See [clean](#clean) for how ```model.inputs``` and ```model.data``` get defined and/or [generate_trainlog](#generate_trainlog) for how ```model.trainlog``` gets defined.

##### bss_derivatives

```python
dy = model.bss_derivatives(**kwargs)
```

For returning gradient of modeled function with respect to each, or specified, input variable.
If user overrides default settings, then 1st and 2nd partial derivatives can be returned for any variables.

| Keyword               | Type                                                    | Description                                                                                                                                                                                                                                                    | Default               |
|-----------------------|---------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------|
| ```inputs```          | -                                                       | see ```model.inputs``` of [clean](#clean)                                                                                                                                                                                                                      | ```model.inputs```    | 
| ```kernel```          | -                                                       | see ```kernel``` of [FoKL](#fokl)                                                                                                                                                                                                                              | ```model.kernel```    |    
| ```d1```              | integer (for single) or list of booleans (for multiple) | index of input variable(s) (i.e., state(s)) to use for first partial derivative; see tip below                                                                                                                                                                 | ```True```            |           
| ```d2```              | integer (for single) or list of booleans (for multiple) | index of input variable(s) (i.e., state(s)) to use for second partial derivative; see tip below                                                                                                                                                                | ```False```           |          
| ```draws```           | -                                                       | see ```draws``` of [FoKL](#fokl)                                                                                                                                                                                                                               | ```model.draws```     |     
| ```betas```           | -                                                       | see ```betas``` of [FoKL](#fokl)                                                                                                                                                                                                                               | ```model.betas```     |     
| ```phis```            | -                                                       | see ```phis``` of [FoKL](#fokl)                                                                                                                                                                                                                                | ```model.phis```      |      
| ```mtx```             | $(terms-1) \times m$ ndarray                            | interaction matrix defining terms in FoKL model by indexing basis function order for each term and input variable combination                                                                                                                                  | ```model.mtx```       |       
| ```minmax```       | -                                                       | see ```minmax``` of [clean](#clean)                                                                                                                                                                                                                         | ```model.minmax``` | 
| ```IndividualDraws``` | boolean                                                 | for returning derivative(s) at each draw                                                                                                                                                                                                                       | ```False```           |              
| ```ReturnFullArray``` | boolean                                                 | for returning $n \times m \times 2$ array with zeros for non-requested states such that indexing is preserved; otherwise, only requested states are squeezed into a 2D matrix where columns correspond to increasing input variable index and derivative order | ```False```           |              

| Output   | Type                                                                                                                                                      | Description                                                                                             | Default                                                                                             |
|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| ```dy``` | $n \times m \times 2$ ndarray if ```ReturnFullArray=True```, else $n \times m_{\delta}$ where $m_{\delta}$ is the number of partial derivatives requested | derivative of model with respect to input variable(s) (i.e., state(s)) defined by ```d1``` and ```d2``` | gradient (i.e., $n \times m_{\delta}$ ndarray where $m_{\delta} =m$ because ```d1=True, d2=False``` |

Tip:
- To turn off all first-derivatives, set ```d1=False``` instead of ```d1=0```. The reason is ```d1``` and ```d2```, if set to an integer,
 will return the derivative with respect to the input variable indexed by that integer using Python indexing.
 In other words, for a two-input FoKL model, setting ```d1=1``` and ```d2=0``` will return the first-derivative with
 respect to the second input (```d1=1```) and the second-derivative with respect to the first input (```d2=0```).
 Alternatively, ```d1=[False, True]``` and ```d2=[True, False]``` will function the same so that boolean lists may be used in cases where the derivative with respect to more than one state, but not all states, is required.

##### evaluate_basis

```python
basis = model.evaluate_basis(c, x, kernel=None, d=0)
```

Evaluate a basis function at a single point by providing coefficients, $x$ value(s), and (optionally) the kernel. This method is primarily used internally by other methods and so is not expected to be used by the user, but is available for testing purposes and to provide insight toward how the basis functions get evaluated.

For evaluating a FoKL model, see [evaluate](#evaluate).

| Input   | Type            | Description                                                           |
|---------|-----------------|-----------------------------------------------------------------------|
| ```c``` | list of scalars | coefficients of the basis function or its derivative                                    |
| ```x``` | scalar          | value of independent variable at which to evaluate the basis function or its derivative |

| Keyword      | Type    | Description                                    | Default            |
|--------------|---------|------------------------------------------------|--------------------|
| ```kernel``` | -       | see ```kernel``` of [FoKL](#fokl)              | ```model.kernel``` |
| ```d```      | integer | order of derivative (where 0 is no derivative) | ```0```            |

| Output      | Type   | Description                        |
|-------------|--------|------------------------------------|
| ```basis``` | scalar | evaluation of basis function or its derivative at ```x``` |

If insightful for understanding how to define ```c```, the values of ```kernel``` and order ```d``` correspond to the following equations at which ```basis``` is evaluated:

| Kernel                        | Order    | Basis Function $B_i$ or its Derivative </sup>                                                                                                                                        |
|-------------------------------|-----------|----------------------------------------------------------------------------------------------------------------------------------------------|
| ```'Cubic Splines'```         | ```d=0``` | $B_i=c_0+c_1 \cdot x+c_2 \cdot x^2+c_3 \cdot x^3 \implies$ <pre>```c[0] + c[1] * x + c[2] * (x ** 2) + c[3] * (x ** 3)```</pre>                  |
| "                             | ```d=1``` | $\frac{\partial}{\partial x}(B_i)=c_1+2\cdot c_2\cdot x+3\cdot c_3\cdot x^2 \implies$ <pre>```c[1] + 2 * c[2] * x + 3 * c[3] * (x ** 2)```</pre>                              |
| "                             | ```d=2``` | $\frac{\partial^2}{\partial x^2}(B_i)=2\cdot c_2+6\cdot c_3\cdot x \implies$ <pre>```2 * c[2] + 6 * c[3] * x```</pre>                                                             |
| ```'Bernoulli Polynomials'``` | ```d=0``` | $B_i=\sum_{k=0}^{i} (c_k \cdot x^k)\implies$ <pre>```c[0] + sum(c[k] * (x ** k) for k in range(1, len(c)))```</pre>                                      |
| "                             | ```d=1``` | $\frac{\partial}{\partial x}(B_i)=\sum_{k=1}^{i} (k \cdot c_k \cdot x^{k-1})\implies$ <pre>```c[1] + sum(k * c[k] * (x ** (k - 1)) for k in range(2, len(c)))```</pre>                |
| "                             | ```d=2``` | $\frac{\partial^2}{\partial x^2}(B_i)=\sum_{k=2}^{i} (k \cdot (k-1) \cdot c_k \cdot x^{k-2})\implies$ <pre>```sum((k - 1) * k * c[k] * (x ** (k - 2)) for k in range(2, len(c)))```</pre> |

When called internally by [evaluate](#evaluate), the coefficients ```c``` (i.e., $\overline{c}_i$) automatically correspond to $i$ such that $B_i=f(\overline{c}_i)$. For  ```'Cubic Splines'```, this is achieved by ```c = list(model.phis[i - 1][k][phind] for k in range(4))``` where ```phind``` $=f(x)$. For ```'Bernoulli Polynomials'```, this is achieved by ```c = model.phis[i - 1]```.

##### evaluate

```python
mean = model.evaluate(inputs=None, betas=None, mtx=None, **kwargs)
```

Evaluate the FoKL model for provided inputs and (optionally) calculate bounds.

| Input        | Type | Description                               | Default            |
|--------------|------|-------------------------------------------|--------------------|
| ```inputs``` | -    | see ```model.inputs``` of [clean](#clean) | ```model.inputs``` |
| ```betas```  | -    | see ```betas``` of [fit](#fit)            | ```model.betas```  |
| ```mtx```    | -    | see ```mtx``` of [fit](#fit)              | ```model.mtx```    |

| Keyword            | Type    | Description                                                                                                                    | Default           |
|--------------------|---------|--------------------------------------------------------------------------------------------------------------------------------|-------------------|
| ```minmax```    | -       | see ```minmax``` of [clean](#clean)                                                                                         | ```None```        |
| ```draws```        | -       | see ```draws``` of [FoKL](#fokl)                                                                                               | ```model.draws``` |
| ```clean```        | boolean | pass ```inputs``` to [clean](#clean) if true; note this will override ```minmax``` and result in ```inputs``` scaled to 0-1 | ```False```       |
| ```ReturnBounds``` | boolean | return 95% confidence bounds as second output if true                                                                          | ```False```       |

If ```clean=True```, then any keywords documented for [clean](#clean) may be used here.

| Output                  | Type                 | Description                                                                                                                                                                                                           |
|-------------------------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```mean```              | $n \times 1$ ndarray | prediction of $\overline{y}$ in $\overline{y}=f(\overline{x}_1,...,\overline{x}_m)$ for provided ```inputs```; prediction of ```model.data``` defined in [clean](#clean) by default (i.e., ```inputs=model.inputs```) |
| ```bounds``` (optional) | $n \times 2$ ndarray | upper and lower bounds for 95% confidence interval of predicting; returned if ```ReturnBounds=True```                                                                                                                 |

##### coverage3

```python
mean, bounds, rmse = model.coverage3(**kwargs)
```

For validation testing of a FoKL model. Default functionality is to evaluate all inputs (i.e., train and test sets) using [evaluate](#evaluate).
Returned is the predicted output ```mean```, 95% confidence bounds ```bounds```, and Root Mean Square Error ```rmse```. A plot
may be returned by setting ```plot=True```; or, for a potentially more meaningful plot in terms of judging
accuracy, ```plot='sorted'``` will plot the data in increasing value.

To govern what is passed to [evaluate](#evaluate):

| Keyword      | Type | Description                               | Default            |
|--------------|------|-------------------------------------------|--------------------|
| ```inputs``` | -    | see ```model.inputs``` of [clean](#clean) | ```model.inputs``` |
| ```data```   | -    | see ```model.data``` of [clean](#clean)   | ```model.data```   |
| ```draws```  | -    | see ```draws``` of [FoKL](#fokl)          | ```model.draws```  |
|``` nrmse```  | -    | normalized root mean square error         | False              |

To govern basic plot controls:

| Keyword                 | Type              | Description                                                         | Default        |
|-------------------------|-------------------|---------------------------------------------------------------------|----------------|
| ```plot```              | boolean or string | for generating plot; set to ```'sorted'``` for plot of ordered data | ```False```    |
| ```bounds```            | boolean           | for plotting bounds                                                 | ```True```     |
| ```xaxis```             | integer           | index of the input variable to plot along the x-axis                | indices        |
| ```labels```            | boolean           | for adding labels to plot                                           | ```True```     |
| ```xlabel```            | string            | x-axis label                                                        | ```'Index'```  |
| ```ylabel```            | string            | y-axis label                                                        | ```'Data'```   |
| ```title```             | string            | plot title                                                          | ```'FoKL'```   |
| ```legend```            | boolean           | for adding legend to plot                                           | ```True```     |
| ```LegendLabelFoKL```   | string            | FoKL's label in legend                                              | ```'FoKL'```   |
| ```LegendLabelData```   | string            | Data's label in legend                                              | ```'Data'```   |
| ```LegendLabelBounds``` | string            | Bounds's label in legend                                            | ```'Bounds'``` |

To govern detailed plot controls:

| Keyword              | Type   | Description                 | Default     |
|----------------------|--------|-----------------------------|-------------|
| ```PlotTypeFoKL```   | string | FoKL's color and line type  | ```'b'```   |
| ```PlotSizeFoKL```   | scalar | FoKL's line size            | ```2```     |
| ```PlotTypeBounds``` | string | Bounds' color and line type | ```'k--'``` |
| ```PlotSizeBounds``` | scalar | Bounds' line size           | ```2```     |
| ```PlotTypeData```   | string | Data's color and line type  | ```'ro'```  |
| ```PlotSizeData```   | scalar | Data's line size            | ```2```     |

| Output       | Type   | Description                                                            |
|--------------|--------|------------------------------------------------------------------------|
| ```mean```   | -      | see ```mean``` of [evaluate](#evaluate)                                |
| ```bounds``` | -      | see ```bounds``` of [evaluate](#evaluate)                              |
| ```rmse```   | scalar | Root Mean Squared Error (RMSE) of prediction in relation to ```data``` |

##### fit

```python
betas, mtx, evs = model.fit(inputs=None, data=None, **kwargs)
```

Training routine for fitting model to known inputs and data.

| Input        | Type | Description                                    | Default                                 |
|--------------|------|------------------------------------------------|-----------------------------------------|
| ```inputs``` | -    | see ```traininputs``` of [trainset](#trainset) | ```traininputs, _ = model.trainset()``` |
| ```data```   | -    | see ```traindata``` of [trainset](#trainset)   | ```_, traindata = model.trainset()```   |

| Keyword             | Type    | Description                                                                                                                                                                                                       | Default     |
|---------------------|---------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------|
| ```clean```         | boolean | pass ```inputs``` and ```data``` to [clean](#clean) if true                                                                                                                                                       | ```False``` |
| ```ConsoleOutput``` | boolean | print [ind, ev] to console during FoKL model generation; will print percent completed of each Gibbs sampler call prior to [ind, ev] if large dataset (i.e., if less than 64-bit was requested in [clean](#clean)) | ```True```  |

If ```clean=True```, then any keywords documented for [clean](#clean) may be used here.

| Output      | Type                         | Description                                                                                                                                                                                                                                                                  |
|-------------|------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```betas``` | $draws \times terms$ ndarray | draws from the posterior distribution of coefficients, with rows corresponding to draws (i.e., a single set of coefficients) and columns corresponding to terms in the model (i.e., $\beta_0, \beta_1, \dots $)                                                              |
| ```mtx```   | $(terms-1) \times m$ ndarray | interaction matrix defining order of basis function for term/variable combinations in FoKL model, with rows corresponding to terms (i.e., columns of ```betas``` beyond the first column) and columns corresponding to input variables (i.e., columns of ```model.inputs```) |
| ```evs```   | ndarray                      | vector of BIC values corresponding to each proposed model during training                                                                                                                                                                                                    |

##### clear

```python
model.clear(keep=None, clear=None, all=False)
```

Delete all attributes from the FoKL class except for hyperparameters and settings, unless otherwise specified by the ```clear``` keyword.
If an attribute is listed in both the ```clear``` and ```keep``` keywords, then the attribute is cleared.

| Input       | Type            | Description                                                                                        | Default          |
|-------------|-----------------|----------------------------------------------------------------------------------------------------|------------------|
| ```keep```  | list of strings | attributes to keep in addition to hyperparameters and settings, e.g., ```keep=['inputs', 'mtx']``` | ```model.keep``` |
| ```clear``` | list of strings | hyperparameters to delete, e.g., ```clear=['kernel', 'phis']```                                    | ```None```       |
| ```all```   | boolean         | if ```True``` then all attributes (including hyperparameters) get deleted regardless               | ```False```      |

Note when the FoKL class was initialized, ```model.keep``` got defined by default as a list of strings including the names of all hyperparameters and settings. These then get preserved here by default.

To remove all attributes from the class, simply call:
```python
model.clear(all=True)
```

##### to_pyomo

```python
m = model.to_pyomo(xvars, yvars, m=None, xfix=None, yfix=None, truescale=True, std=True, draws=None)
```

Pass arguments to [fokl_to_pyomo](#fokl_to_pyomo). If embedding a single GP in Pyomo rather than multiple, it is recommended to use this method to avoid importing an additional module in the run script.

##### save

```python
filepath = model.save(filename=None, directory=None)
```

Save a FoKL class as a file with extension '*.fokl*'. If not saving where expected relative to the run script, ensure the terminal directory is set to that of the run script.

Both inputs are optional. By default, ```filename``` is of the form '*model_yyyymmddhhmmss.fokl*' and is saved to the
current directory. To change the directory, embed within ```filename``` or assign to ```directory``` if using the default ```filename``` format.

Returned is ```filepath```. Enter this as the argument to [load](#foklroutinesload) to later reload the model. Explicitly, that is:
```python
FoKLRoutines.load(filepath)
```

| Input           | Type   | Description                                                                                        |
|-----------------|--------|----------------------------------------------------------------------------------------------------|
| ```filename```  | string | name of file to save model as (note '*.fokl*' extension can be automatically or manually appended) |
| ```directory``` | string | absolute or relative path to pre-existing folder in which to save ```filename```                   |

| Output         | Type   | Description                               |
|----------------|--------|-------------------------------------------|
| ```filepath``` | string | absolute path to where the file was saved |

### fokl_to_pyomo

```python
from FoKL.fokl_to_pyomo import fokl_to_pyomo
m = fokl_to_pyomo(models, xvars, yvars, m=None, xfix=None, yfix=None, truescale=True, std=True, draws=None)
```

Embed GP's in Pyomo by automatically converting ```draws``` from FoKL models trained with or defined by the ```'Bernoulli Polynomials'``` kernel to symbolic expressions in a Pyomo model.

Defining the Pyomo model's objective and any other constraints must be done outside of ```fokl_to_pyomo```. The user must then define an appropriate solver for the Pyomo model. The following is known to work for global optimization problems, which will likely be required for a problem using a GP model.
```python
solver = pyo.SolverFactory('multistart')
solver.solve(m, solver='ipopt')
```

For documentation on the components of the Pyomo model automatically generated, see [*nomenclature_of_fokl_to_pyomo.ipynb*](docs/_dev/in_dev__nomenclature_of_fokl_to_pyomo.ipynb). The function arguments are as follows.

| Input | Type | Description | Default |
|---|---|---|---|
| ```models``` | list of FoKL class objects | multiple FoKL models to be embedded in single Pyomo model | - |
| ```xvars``` | list of lists of strings | strings are input variable names, and lists correspond to models | - |
| ```yvars``` | list of strings | strings are output variable names corresponding to models | - |
| ```m``` | Pyomo model | pre-existing Pyomo model if existing | ```pyo.ConcreteModel()``` |
| ```xfix``` | list of lists of floats | floats are input variable values if known and to be fixed), and lists correspond to models | - |
| ```yfix``` | list of floats | floats are output variable values (if known and to be fixed) corresponding to models | - |
| ```truescale``` | list of lists of booleans | corresponding to the variables created by ```xvars```, set ```True``` to use true scale (i.e., un-normalized) values and set ```False``` to use normalized values; unless ```xvars``` is to correspond to the normalized input variables, leave blank | ```[[True, ..., True], ..., [True, ..., True]]``` |
| ```std``` | list of booleans | set ```False``` if standard deviation of FoKL model (corresponding to position in list) is not needed so Pyomo model only defines mean | ```[[True, ..., True], ..., [True, ..., True]]``` |
| ```draws``` | int | number of most recent draws to embed in Pyomo | ```model.draws``` |

### getKernels

*For internal use.*

This package is used by [FoKL](#fokl) during initialization to return the data structure ```phis```, containing coefficients for the basis functions specified by ```kernel```.
```python
import FoKL.getKernels
phis = getKernels.sp500()  # kernel == 'Cubic Splines'
phis = getKernels.bernoulli()  # kernel == 'Bernoulli Polynomials'
```

### GP_integrate

```python
from FoKL.GP_Integrate import GP_Integrate
T, Y = GP_Integrate(betas, matrix, b, norms, phis, start, stop, y0, h, used_inputs)
```

Integrate FoKL models that were fitted to derivatives. Multiple models are able to be integrated simulatneously. Currently, only models trained on the "Cubic Splines" basis functions are supported.

For example, training ```model1``` on $x = f(\dot{x}, b_1)$ and ```model2``` on $y = f(\dot{y}, b_2)$ is as usual. Then, to integrate the models with constants $(b_1, b_2)$ set to ```b``` and initial conditions $(x_0, y_0)$ set to ```y0```,

```python
betas1, mtx1, _ = model1.fit([xdot, b1], x)
betas2, mtx2, _ = model2.fit([ydot, b2], y)

T, Y = GP_integrate([np.mean(betas1, axis=0), np.mean(betas2, axis=0)], 
                    [mtx1, mtx2], 
                    [b1, b2], 
                    ..., 
                    [x0, y0], 
                    ...)
```

| Input             | Description                                                                                                                                                                                                                                                                                                                                                                                                 |
|-------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```betas```       | ```betas``` is a list of arrays in which each entry to the list contains a specific row of the betas matrix, or the mean of the betas matrix for each model being integrated.                                                                                                                                                                                                                               |
| ```matrix```      | ```matrix``` is a list of arrays containing the interaction matrix of each model.                                                                                                                                                                                                                                                                                                                           |
| ```b```           | ```b``` is an array of the values of all the other inputs to the model(s) (including any forcing functions) over the time period we integrate over. The length of ```b``` should be equal to the number of points in the final time series ```(stop - start) / h```. All values in ```b``` need to be normalized with respect to the min and max values of their respective values in the training dataset. |
| ```norms```       | ```norms``` is a matrix of the min and max values of all the inputs being integrated (in the same order as ```y0```). Min values are in the top row, max values in the bottom.                                                                                                                                                                                                                              |
| ```phis```        | ```phis``` is a data structure with coefficients for basis functions.                                                                                                                                                                                                                                                                                                                                       |
| ```start```       | ```start``` is the time at which integration begins.                                                                                                                                                                                                                                                                                                                                                        |
| ```stop```        | ```stop``` is the time to end integration.                                                                                                                                                                                                                                                                                                                                                                  |
| ```y0```          | ```y0``` is an array of the inital conditions for the models being integrated.                                                                                                                                                                                                                                                                                                                              |
| ```h```           | ```h``` is the step size with respect to time.                                                                                                                                                                                                                                                                                                                                                              |
| ```used_inputs``` | ```used_inputs``` is a list of arrays containing the information as to what inputs are used in what model. Each array should contain a vector corresponding to a different model. Inputs should be referred to as those being integrated first, followed by those contained in ```b``` (in the same order as they appear in ```y0``` and ```b``` respectively).                                             |

| Output  | Description                                                                                          |
|---------|------------------------------------------------------------------------------------------------------|
| ```T``` | ```T``` is an array of the time steps the models are integrated at.                                  |
| ```Y``` | ```Y``` is an array of the models that have been integrated, at the time steps contained in ```T```. |

To demonstrate ```used_inputs```, suppose two models were being integrated with 3 other inputs total.
The 1st model uses the output of both models as inputs; and, the 1st and 3rd additional
inputs. The 2nd model uses its own output as an input; and, the 2nd
and 3rd additional inputs. This yields
```python
used_inputs = [[1, 1, 1, 0, 1], [0, 1, 0, 1, 0]]
```
If the models created do not follow this ordering scheme for their inputs,
the inputs can be rearranged based upon an alternate
numbering scheme provided to ```used_inputs```. E.g., if the inputs need to be reordered then the 1st input should have a '1' in its place in the
```used_inputs``` vector, the 2nd input should have a '2', and so on. Using the
same example as before, if the 1st model's inputs need to be rearranged so that
the 3rd additional input comes first, followed by the two model outputs in
the same order as they are in ```y0```, and ends with the 1st additional input,
then the 1st list in ```used_inputs``` would be ```[2, 3, 4, 0, 1]```.

# Updating Models
BSS ANOVA models can be updated as new data comes available. To perform this capability a few different hyperparameters can be defined for model updating methods
| Hyper Paramter | Description | Necessary to define? |
|----------|------------------------------------------------------------------------------------|--------------|
| update | Removes variable selection functionality to allow for future updates of models | Yes |
| sigsqd0 | initial sigma squared guess | Yes |
| burn | How many draws to remove from prior betas model before new fitting | No, sets to 500 |
| built | Boolean for if model has been previously built | Yes |

Once the proper parameters are in place, models can be updated with each successive calling of ```model.fit``` and redefining of the inputs and data. See [update Sigmoid Example Problem](https://github.com/ESMS-Group-Public/FoKL-GPy/blob/Update/examples/sigmoid/updateSig.py) for an example

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

| Topic                                            | Point of Contact | Email                                                         |
|--------------------------------------------------|------------------|---------------------------------------------------------------|
| Installation<br/>Troubleshooting<br/>Development | Jacob Krell      | [jpk0024@mix.wvu.edu](mailto:jpk0024@mix.wvu.edu)             |
| Research<br/>Theory<br/>Other                    | David Mebane     | [david.mebane@mail.wvu.edu](mailto:david.mebane@mail.wvu.edu) |

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
