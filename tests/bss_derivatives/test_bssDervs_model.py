from src.FoKL import FoKLRoutines
import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(0,1,10001)
y = np.exp(x)

model = FoKLRoutines.FoKL()

model.fit(x, y)
y_model = model.evaluate(x)

d_model = model.bss_derivatives(d2=1)

plt.figure(1)
plt.title('FoKL model of e^x')
plt.xlabel('x')
plt.plot(x, y_model)
plt.plot(x, d_model[:,0]) # 1st derivative
plt.plot(x, d_model[:,1]) # 2nd derivative
plt.legend(['model','d(model)/dx','d^2(model)/dx^2'])
plt.show()


breakpoint()