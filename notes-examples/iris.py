#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
from m2l2.classification import DA, NaiveBayes, kNN, SVM, AdaBoost

## Data generation
data = load_iris()
X = data.data
y = data.target
X = X[:,[0,1]]
X = X[y != 2]
y = y[y != 2]

## set up plot
x_min, x_max = X[:, 0].min(), X[:, 0].max()
hsp = (x_max - x_min)/5
x_min, x_max = x_min - hsp, x_max + hsp
y_min, y_max = X[:, 1].min(), X[:, 1].max()
vsp = (y_max - y_min)/5
y_min, y_max = y_min - vsp, y_max + vsp

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.axes()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.title('Fisher Iris Data')


## model
classy = NaiveBayes()
classy.train(X, y)


## plot results
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = np.apply_along_axis(classy.classify,
                        1, np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.4)
ax.contour(xx, yy, Z, [0], colors='k', alpha=.8)

# only for SVM: indicate support vectors
#ax.scatter(classy._SV[:,0], classy._SV[:,1], 30, 'k', 'x', alpha=.7)

plt.show()
