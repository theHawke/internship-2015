#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from matplotlib.colors import ListedColormap
from numpy.random import shuffle
from m2l2.clustering import kMeans, GaussianMixtureEM
import csv

## Data generation
with open('iris.data', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    Xy = [(row[0:4], row[4]) for row in reader]
    shuffle(Xy)
    Xp,yp = zip(*Xy)

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
cm_bright = ListedColormap(['#FF0000', '#0000FF', '#00FF00'])
fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.title('Fisher Iris Data (clustering)')


## model
km = GaussianMixtureEM(X, n=2)

scatter = ax.scatter(X[:, 0], X[:, 1], c=km.cl, cmap=cm_bright)
means = ax.scatter(km.mu[:,0], km.mu[:,1], c=np.arange(km._ncl),
                   cmap = cm_bright, marker='+', s=60)

## animation
def update_plot(i):
    scatter.set_array(km.cl)
    means.set_offsets(km.mu)
    km.E_step()
    km.M_step()
    return scatter, means

an = ani.FuncAnimation(fig, update_plot, frames=100, blit=True,
                       interval=500, repeat=False)

an.save("cluster_iris.mp4")
