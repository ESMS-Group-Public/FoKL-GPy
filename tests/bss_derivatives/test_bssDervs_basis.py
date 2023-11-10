from src.FoKL import FoKLRoutines
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,1,10001)

model = FoKLRoutines.FoKL()

for id in range(6): # max is 25

    d_basis, basis = model.bss_derivatives(inputs=x, span=[0,1], d2=1, betas=[0,1], mtx=id+1, draws=1, ReturnBasis=1)

    plt.figure(id+1)
    plt.title('mtx = %i, (b0,b1)=[0,1]' %(id+1))
    plt.xlabel('x')
    plt.plot(x, basis)
    plt.plot(x, d_basis[:,0]) # 1st derivative
    plt.plot(x, d_basis[:,1]) # 2nd derivative
    plt.legend(['basis','d(basis)/dx','d^2(basis)/dx^2'])
    plt.show()


breakpoint()