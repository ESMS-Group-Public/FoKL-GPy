import FoKL
import pandas as pd
import numpy as np
import random
from FoKL import FoKLRoutines
from FoKL import getKernels

data_load = pd.read_csv("testdatatest.csv", dtype=float)
inputs = data_load[['x','y']]
data = data_load['data']

# Loading data that used default fit parameters

betas_tested = np.loadtxt("betas.csv")

mtx_tested = np.loadtxt("mtx.csv")

evs_tested = np.loadtxt("evs.csv")

meen_tested = np.loadtxt("meen.csv")

# Loading data that used changed fit parameters (as well as different seed)

betas_testedChange = np.loadtxt("betasChange.csv")

mtx_testedChange = np.loadtxt("mtxChange.csv")

evs_testedChange = np.loadtxt("evsChange.csv")

meen_tested2 = np.loadtxt("meenChange.csv")

# The data above is to ensure that fit and coverage produce the proper results when run again

def test_fit_for_shaping(): #Ensures the values used in fit become numpy arrays, even if something like Pandas is used, this is necessary to ensure they are properly converted to lists later in the code
    model = FoKLRoutines.FoKL()
   
    betas, mtx, evs = model.fit(inputs, data, clean = True) 
    assert isinstance(betas, np.ndarray)
    assert isinstance(mtx, np.ndarray)
    assert isinstance(evs, np.ndarray)

def test_fit_with_data_and_default_params():
    random.seed(102823, version = 1)
    np.random.seed(102823)

    model = FoKLRoutines.FoKL()

    #seeds above for defualt parameter testing for fit

    betas, mtx, evs = model.fit(inputs, data, clean = True)

    assert betas.shape == (1000, len(data_load)) #The shape of betas is based on the rows of data used in the loaded data as well as draws (the 1000)

    assert np.allclose(evs, evs_tested) # evs evaluates the model which was set with a seed so their arrays should be the same
    assert np.allclose(betas, betas_tested) # While betas is stochastic, using a seed should lead to the array being the same, this tests that
    assert np.allclose(mtx, mtx_tested) # This produces a matrix from the best model set using GP integrate, it should be the same with a seed set

def test_fit_with_data_and_changed_params(): #This does the same thing as previous test but used a different seed and different fit parameters
    random.seed(102923, version = 1)
    np.random.seed(102923)

    model = FoKLRoutines.FoKL()
    
    #seeds above for changed parameter testing for fit

    betas, mtx, evs = model.fit(inputs, data, aic=True, a=3, b=1.8, atau=17, btau=2100.5, tolerance=3, clean = True)

    assert betas.shape == (1000, len(data_load))

    assert np.allclose(evs, evs_testedChange) # evs evaluates the model which was set with a seed so their arrays should be the same
    assert np.allclose(betas, betas_testedChange) # While betas is stochastic, using a seed should lead to the array being the same, this tests that
    assert np.allclose(mtx, mtx_testedChange) # This produces a matrix from the best model set using GP integrate, it should be the same with a seed set

def test_coverage3():
    random.seed(102823, version = 1)
    np.random.seed(102823) #defualt parameters

    model = FoKLRoutines.FoKL()
    
    betas, mtx, evs = model.fit(inputs, data, clean = True)
    meen, bounds, rmse = model.coverage3()

    assert len(meen) == 10 # ensures the meen array is the expected length (based on rows of data used)
    assert bounds.shape == (10, 2) # ensures bounds match the expected size (based on rows and columns of data used)
    assert rmse.shape == () # ensures rmse is an empty tuple and a scalar value rather than an array

    assert np.allclose(meen, meen_tested) # compares the meen created from makingdata.py using default parameters to the newly created meen in this test, since seeds are set these should be the same.

def test_routines_works_defaultparams(): #this is an integration test, it is the previous default parameter tests combined
    random.seed(102823, version = 1)
    np.random.seed(102823)

    model = FoKLRoutines.FoKL()
    
    #seeds above for defualt parameter testing for fit

    betas, mtx, evs = model.fit(inputs, data, clean = True)

    assert betas.shape == (1000, len(data_load)) #The shape of betas is based on the rows of data used in the loaded data as well as draws (the 1000)

    assert np.allclose(evs, evs_tested) # evs evaluates the model which was set with a seed so their arrays should be the same
    assert np.allclose(betas, betas_tested) # While betas is stochastic, using a seed should lead to the array being the same, this tests that
    assert np.allclose(mtx, mtx_tested) # This produces a matrix from the best model set using GP integrate, it should be the same with a seed set

    #tests for coverage3

    meen, bounds, rmse = model.coverage3()

    assert np.allclose(meen, meen_tested)
    assert len(meen) == 10
    assert bounds.shape == (10, 2)
    assert rmse.shape == ()

def test_routines_works_changedparams(): #this is an integration test, it is the previous changed parameter tests combined
    random.seed(102923, version = 1)
    np.random.seed(102923)

    model = FoKLRoutines.FoKL()\
    
    betas, mtx, evs = model.fit(inputs, data, aic=True, a=3, b=1.8, atau=17, btau=2100.5, tolerance=3, clean = True)
    meen, bounds, rmse = model.coverage3()

    #tests for fit

    assert betas.shape == (1000, 10)
    assert np.allclose(evs, evs_testedChange) # evs evaluates the model which was set with a seed so their arrays should be the same
    assert np.allclose(betas, betas_testedChange) # While betas is stochastic, using a seed should lead to the array being the same, this tests that
    assert np.allclose(mtx, mtx_testedChange) # This produces a matrix from the best model set using GP integrate, it should be the same with a seed set


    #tests for coverage3

    assert np.allclose(meen, meen_tested2)
    assert len(meen) == 10
    assert bounds.shape == (10, 2)
    assert rmse.shape == ()

# get all info on how to download tox, writing tests, where the tests are and maybe how tox works with environments 
# teach how to do tox testing over again, assume we know what FoKL
