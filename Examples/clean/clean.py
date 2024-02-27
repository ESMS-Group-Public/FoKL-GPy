"""
The following


"""
from src.FoKL import FoKLRoutines
import numpy as np


f = FoKLRoutines.FoKL()

# If you have known inputs (i.e., x0, x1, ..., xN), FoKL can automatically normalize and format them using 'clean':

x0 = np.linspace(0, 1, 10)
x1 = np.random.rand(10)
x2 = np.exp(x0)

f.clean([x0, x1, x2])

# If you also have a known model (i.e., betas and mtx), 'clean' can be performed inside of 'evaluate':

betas = [1, 1, 1]
mtx = [[0, 0, 1], [0, 1, 0]]

y = f.evaluate([x0, x1, x2], betas, mtx, clean=1)

# If you have known data (i.e., y) but not a model, 'clean' can be performed inside of 'fit' by default. However, this
# is the default functionality so 'clean=1' does not need to be specified. Instead, try using the 'train' keyword for
# the 'clean' method so that the model only trains on 60% of the inputs. The two methods below provide the same
# functionality except for any stochastic influences. The first method is recommended for its simplicity:

y = x0 + x1 + x2

f.fit([x0, x1, x2], y, train=0.6)

f.clean([x0, x1, x2], train=0.6)
f.fit(f.inputs, y)

# After fitting, the model may be evaluated:

y_theory = f.evaluate()
percent_error = abs(y_theory - y)/y

breakpoint()

