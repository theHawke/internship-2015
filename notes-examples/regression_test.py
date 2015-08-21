#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import uniform, norm
import matplotlib.pyplot as plt
from m2l2.regression import Polynomial, Gaussian, Sigmoidal
from m2l2.regression import OLS, RidgeRegression, Lasso, Bayesian, PLS
from m2l2.RVM import RVM

# Generate fit data: sin wave with noise
N = 20
noise = 0.1
x = uniform.rvs(size=N)
eps = norm.rvs(scale=noise, size=N)

y = np.sin(2*np.pi*x) + eps

# plot original and data points
xx = np.linspace(0, 1)
ax = plt.axes()
ax.plot(xx, np.sin(2*np.pi*xx), 'k--', label='source')
ax.plot(x, y, 'b+', label='data', ms=10, mew=1.2)
ax.set_ylim(-1.5, 1.5)
plt.xlabel('x')
plt.ylabel('y(x)')
plt.title('Regression Testing')


# Model
reg = RVM(Gaussian(low=0, high=1, num=10, scale=0.3))
reg.fit(x, y)


# plot results

# for Bayesian models only: plot the standard deviation of the predictive distribution
yy, vv = reg.predict(xx, return_variance=True)
sd = np.sqrt(vv)
ax.fill_between(xx, yy-sd, yy+sd, color='r', alpha=0.3)

# plot the predicted curve
ax.plot(xx, reg.predict(xx), 'r-', label='model')

plt.show()
