# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import svd


def normalise(X, return_params=False):
    N = X.shape[0]

    mu = np.sum(X, axis=1)/N

    var = np.sum((X - mu)**2, axis=1)/N

    newX = (X - mu)/np.sqrt(var)

    if return_params:
        return newX, mu, var
    else:
        return newX


def PCA(X):
    N, D = X.shape

    mu = np.sum(X, axis=1)/N

    _, s, V = svd(X-mu)

    return s, V
