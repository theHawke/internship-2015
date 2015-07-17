#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from m2l2.classification import DA, NaiveBayes, kNN, SVM
import csv

## Data generation
with open('iris.data', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    Xp,yp = zip(*[(row[0:4], row[4]) for row in reader])

classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

y = np.array([classes[name] for name in yp])
X = np.array([np.array([float(f[0]), float(f[1])]) for f in Xp])
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
# Plot the training points
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.title('Fisher Iris Data')


## model
classy = SVM(kernel='rbf')
classy.train(X, y)


## plot results
Z = np.apply_along_axis(lambda x: np.sign(classy.classify(x)),
                        1, np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
ax.contourf(xx, yy, Z, cmap=cm, alpha=.4)

plt.show()
