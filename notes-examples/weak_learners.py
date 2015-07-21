#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import csv

## Data generation
with open('iris.data', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    X,y = zip(*[(row[0:4], row[4]) for row in reader])

classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

y = np.array([classes[name] for name in y])
X = np.array([np.array([float(f[0]), float(f[1]), float(f[2]), float(f[3])]) for f in X])
#X = X[y != 2]
#y = y[y != 2]

## set up plot
x_min, x_max = X[:, 0].min(), X[:, 0].max()
hsp = (x_max - x_min)/5
x_min, x_max = x_min - hsp, x_max + hsp
y_min, y_max = X[:, 1].min(), X[:, 1].max()
vsp = (y_max - y_min)/5
y_min, y_max = y_min - vsp, y_max + vsp

# just plot the dataset first
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


class OneClass1D:
    def __init__(self, cl, d):
        self._cl = cl
        self._d = d

    def train(self, X, y):
        Xd = X[:,self._d]
        Xp = Xd[y == self._cl]
        Xo = Xd[y != self._cl]

        self.mu1 = np.mean(Xp)
        self.var1 = np.var(Xp)
        self.n1 = Xp.size
        self.mu0 = np.mean(Xo)
        self.var0 = np.var(Xo)
        self.n0 = Xo.size

    def classify(self, x):
        if x.ndim == 1:
            xx = x[self._d]
        elif x.ndim == 2:
            xx = x[:,self._d]
        return np.sign(np.sqrt(self.var0/self.var1) * self.n1/self.n0 *
                       np.exp((xx-self.mu0)**2/(2*self.var0) -
                              (xx-self.mu1)**2/(2*self.var1)) - 1)

wl = [[OneClass1D(c, d) for d in range(4)] for c in range(3)]
for cl in wl:
    for l in cl:
        l.train(X, y)

# Class 1
for k in range(len(wl)):
    classk = wl[k]

    ci = (y == k) * 2 - 1
    cls = np.array([l.classify(X) for l in classk])
    nmcl = np.sum(ci != cls, axis=1)
    n = y.size

    order = np.argsort(nmcl)

    if nmcl[order[0]] == 0:
        print "perfect classifier"
        continue

    alpha = np.zeros_like(order, dtype=np.float)

    alpha[0] = 0.5 * np.log((n-nmcl[order[0]])/float(nmcl[order[0]]))

    for i in range(1,alpha.size):
        wi = np.exp(-ci * np.sum([alpha[j]*cls[order[j]]
                                  for j in range(alpha.size)], axis=0))
        alpha[i] = 0.5 * np.log(np.sum(wi[cls[order[i]] == ci])/
                                np.sum(wi[cls[order[i]] != ci]))

    cls_final = np.sign(np.sum([alpha[j]*cls[order[j]]
                                for j in range(alpha.size)], axis=0))

    print alpha
    print nmcl[order[0]]
    print np.sum(cls_final != ci)
