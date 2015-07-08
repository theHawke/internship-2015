#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from scipy.optimize import fmin_slsqp
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from random import shuffle
import csv

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

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
fig, ax = plt.subplots()
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)


## my implementation: passive-aggressive algorithm
X = np.column_stack((np.ones(y.size),X))
ci = np.array([-1 if x == 0 else 1 for x in y])

w = np.array([0,1,-1])

ws = np.empty((y.size,3))

for i in range(y.size):
    if ci[i]*np.dot(w,X[i]) < 1:
        f = lambda x: np.dot(x-w,x-w)
        fp = lambda x: 2*(x-w)
        cons = lambda x: ci[i]*np.dot(x,X[i]) - 1
        w = fmin_slsqp(f, X[i]/np.dot(X[i],X[i]), eqcons=[cons], fprime=fp)
    ws[i] = w

linef = lambda x, st: -x*st[0]/st[1] + (st[0]**2 + st[1]**2)/st[1]

ax.grid()
line, = ax.plot([],[])
plt.xlabel('Sepal Width')
plt.ylabel('Sepal Length')
plt.title('Fisher Iris Data (Passive-Aggressive)')

def init_func():
    line.set_data([],[])
    scatter = ax.scatter([],[], c=[], cmap=cm_bright)
    return line,scatter

def update_plot(i):
    scatter = ax.scatter(X[:i+1,1], X[:i+1,2], c=y[:i+1], cmap=cm_bright)

    st = -ws[i,0]*ws[i,1:3]/np.dot(ws[i,1:3],ws[i,1:3])
    line.set_data([x_min, x_max],[linef(x_min, st), linef(x_max, st)])

    return line,scatter

an = ani.FuncAnimation(fig, update_plot, frames=range(y.size), blit=True,
                       init_func=init_func, interval=500, repeat=False)

#fig.show()

an.save("pa_iris.mp4")
