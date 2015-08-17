#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
from m2l2.regression import polynomial, OLS

# Generate fit data: sin wave with noise
N = 20
noise = 0.1
x = uniform.rvs(size=N)
eps = norm.rvs(scale=noise, size=N)

y = np.sin(2*np.pi*(x+1/12)) + eps

# plot original and data points
xx = np.linspace(0, 1, 100)
ax = plt.axes()
ax.plot(xx, np.sin(2*np.pi*(xx+1/12)), 'k--', label='source')
ax.plot(x, y, 'b+', label='data', ms=10, mew=1.2)
ax.set_ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Regression Testing')


# Model

X = polynomial(x, degree=5)

reg = OLS()
reg.fit(X, y)


# plot results
XX = polynomial(xx, degree=5)

ax.plot(xx, reg.predict(XX), 'r-', label='model')

plt.show()
