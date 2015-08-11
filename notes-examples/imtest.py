#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.misc import face, imshow
from m2l2.clustering import kMeans, GaussianMixtureEM

X = np.array(face().reshape((-1,3)), dtype=np.float)

cluster = GaussianMixtureEM(X, 4)
cluster.run(threshold=0.01)

Xp = np.array(cluster.mu[cluster.cl,:], dtype=np.uint8)

imshow(Xp.reshape((768,1024,3)))
