from src.FoKL import FoKLRoutines
import numpy as np
import matplotlib.pyplot as plt


X = np.linspace(0,1,10001)
Y = X

model = FoKLRoutines.FoKL()
_, _, _ = model.fit(X, Y) # just to auto-set attr's of model

betas = np.transpose(np.array([[0],[1]])) # draws by betas ... includes beta0
for id in range(6): # max is 25
    mtx = np.array([id+1])[:, np.newaxis]  # betas by input vars ... excludes beta0

    f_dX, basis = model.bss_derivatives(d1=1, betas=betas, mtx=mtx, draws=1) # first derv
    f_d2X, _ = model.bss_derivatives(d1='off', d2 = 1, betas=betas, mtx=mtx, draws=1) # second derv

    plt.figure(1)
    plt.title('mtx = %i, (b0,b1)=[0,1]' %(id+1))
    plt.xlabel('x')
    plt.plot(X, basis)
    plt.plot(X, f_dX)
    plt.plot(X, f_d2X)
    plt.legend(['basis','d(basis)/dX','d2(basis)/dX^2'])
    plt.show()

