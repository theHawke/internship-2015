
#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from sklearn.datasets import load_iris
from matplotlib.colors import ListedColormap
from numpy.random import shuffle
from m2l2.clustering import kMeans, GaussianMixtureEM
from matplotlib.patches import Ellipse

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
def EllipseFromSigma(Sigma, mu):
    w, v = np.linalg.eigh(Sigma)
    angle = np.arctan2(v[1,0], v[0,0])
    return Ellipse(xy=mu, width = w[0]*4, height = w[1]*4,
                   angle = angle/np.pi*180, fill=False, color='black')

km = GaussianMixtureEM(X, n=2)

# set a nice starting position
km.mu = np.array([[4.5,2.5],[6.5,4]])
km.E_step()

scatter = ax.scatter(X[:, 0], X[:, 1], c=km.cl, cmap=cm_bright)
means = ax.scatter(km.mu[:,0], km.mu[:,1], c=np.arange(km._ncl),
                   cmap = cm_bright, marker='+', s=60)
ells = [EllipseFromSigma(S, m) for (S, m) in zip(km.Sigma, km.mu)]
for e in ells: ax.add_artist(e)

## animation
def update_plot(i):
    scatter.set_array(km.cl)
    means.set_offsets(km.mu)
    for i in range(len(ells)):
        ells[i].remove()
        ells[i] = EllipseFromSigma(km.Sigma[i], km.mu[i])
        ax.add_artist(ells[i])
    km.E_step()
    km.M_step()
    return scatter, means, ells

an = ani.FuncAnimation(fig, update_plot, frames=100, blit=True,
                       interval=500, repeat=False)

an.save("cluster_iris.mp4")
