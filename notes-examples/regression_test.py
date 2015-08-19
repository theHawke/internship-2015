#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
from m2l2.regression import polynomial, gaussian, sigmoidal, OLS, RidgeRegression, Lasso, ARD

# Generate fit data: sin wave with noise
N = 20
noise = 0.1
x = uniform.rvs(size=N)
eps = norm.rvs(scale=noise, size=N)

y = np.sin(2*np.pi*(x+1/12.0)) + eps

# plot original and data points
xx = np.linspace(0, 1)
ax = plt.axes()
ax.plot(xx, np.sin(2*np.pi*(xx+1/12.0)), 'k--', label='source')
ax.plot(x, y, 'b+', label='data', ms=10, mew=1.2)
ax.set_ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Regression Testing')


# Model

X = gaussian(x, low=0, high=1, num=10)

reg = ARD()
reg.fit(X, y)


# plot results
XX = gaussian(xx, low=0, high=1, num=10)

# for ARD only: plot the standard deviation of the predictive distribution
yy, vv = reg.predict(XX, return_variance=True)
sd = np.sqrt(vv)
ax.fill_between(xx, yy-sd, yy+sd, color='r', alpha=0.3)

# plot the predicted curve
ax.plot(xx, reg.predict(XX), 'r-', label='model')

plt.show()
