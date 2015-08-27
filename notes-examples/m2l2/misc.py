# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import svd, eigh


def normalise(X, return_params=False):
    N, _ = X.shape

    mu = np.sum(X, axis=0)/N

    var = np.sum((X - mu)**2, axis=0)/N

    newX = (X - mu)/np.sqrt(var)

    if return_params:
        return newX, mu, var
    else:
        return newX


def PCA(X, m, return_params=False):
    N, D = X.shape

    assert(m <= D)

    mu = np.sum(X, axis=0)/N

    _, s, V = svd(X-mu)

    newX = np.dot(X-mu, V[:m,:].T)

    if return_params:
        return newX, V[:m,:], s[:m]
    else:
        return newX


def LDA(X, y, m, return_params=False):
    N, D = X.shape

    assert(m < D)

    mu = np.sum(X, axis=0)/N

    ys = np.unique(y)
    Xp = np.array([X[y == c] for c in ys])

    mus = np.array([np.sum(Xi, axis=0)/Xi.shape[0] for Xi in Xp])

    # average within-class covariance
    Sigma = np.sum([np.dot((Xi-mui).T, Xi-mui)/Xi.shape[0]
                    for (Xi, mui) in zip(Xp, mus)], axis=0)/ys.size

    # between-class covariance
    Sigma_B = np.dot((mus-mu).T, mus-mu)/ys.size

    w, v = eigh(Sigma_B, Sigma, eigvals = (D-m, D-1))

    newX = np.dot(X-mu, v[:,::-1])

    if return_params:
        return newX, v[:,::-1], w[::-1]
    else:
        return newX


#
# Stuff to maybe add:
# - kernel PCA
# - Independent Component Analysis (ICA)
# - Factor Analysis (FA)
#
