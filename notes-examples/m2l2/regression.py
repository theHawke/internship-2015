# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import inv, solve, qr, solve_triangular, svd
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize

class OLS:
    def __init__(self, method='svd'):
        self._m = method

    def fit(self, X, y):
        # We need to solve (X^T.X)w == X^T y for w.
        # This can be done with various methods:
        if self._m == 'svd':
            # commonly used method, most numerically stable,
            # but a bit slower than qr.
            # using the Singuar Value Decomposition X = U . Sigma . V^T,
            # the equation can be rewritten as: w = (V . Sigma^-1 . U^T) y
            U, s, Vh = svd(X, full_matrices=False)
            self.w = Vh.T.dot(np.diag(1/s).dot(U.T.dot(y)))

        elif self._m == 'qr':
            # also a common method
            # using the QR Decomposition, the equation can be rewritten as:
            # R w = Q^T y
            Q, R = qr(X, mode='economic')
            self.w = solve_triangular(R, np.dot(Q.T, y))

        elif self._m == 'solve':
            # uses the scipy implementation of solve
            self.w = solve(np.dot(X.T, X), np.dot(X.T, y), sym_pos=True)

        elif self._m == 'inverse':
            # very inefficient, for comparison of numerical stability
            # solves by direct computation of the inverse of (X^T . X)
            self.w = np.dot(inv(np.dot(X.T, X)), np.dot(X.T, y))

        else:
            error("Method not implemented")


    def predict(self, x):
        return np.dot(self.w, x.T)

class RidgeRegression:
    def __init__(self, alpha, method='svd'):
        self._m = method
        self.alpha = alpha

    def fit(self, X, y):
        # For regularisation, we need to solve the equation
        # (X^T.X + alpha*I) w == X^T y for w
        N, M = X.shape

        if self._m == 'svd':
            # Using SVD and the Woodbury matrix identity, we get the
            # following equation:
            U, s, Vh = svd(X, full_matrices=False)
            invpart = np.diag(1 / (1 + self.alpha * s**(-2)))
            self.w = np.dot(np.eye(M) - np.dot(Vh.T, np.dot(invpart, Vh)),
                            np.dot(X.T, y)) / self.alpha

        elif self._m == 'solve':
            # uses the scipy implementation of solve
            self.w = solve(np.dot(X.T, X) + self.alpha*np.eye(M),
                           np.dot(X.T, y), sym_pos=True)

        elif self._m == 'inverse':
            # very inefficient, for comparison of numerical stability
            # solves by direct computation of the inverse of (X^T . X)
            self.w = np.dot(inv(np.dot(X.T, X) + self.alpha*np.eye(M)),
                            np.dot(X.T, y))

        # alternatively, there are iterative methods to 'solve'
        # X w == y directly
        elif self._m == 'lsqr':
            self.w, = lsqr(X, y, damp=self.alpha)

        else:
            error("Method not implemented")

    def predict(self, x):
        return np.dot(self.w, x.T)


class Lasso:
    def __init__(self, alpha):
        self.alpha=alpha

    def fit(self, X, y):
        # L1 (Lasso) regularisation doesn't have a nice closed form like L2
        # does, so we have to minimise |y - Xw|^2 + α|w|
        # we do this by reverting to an iterative algorithm
        N, M = X.shape

        def f(w):
            diff = y - np.dot(X, w)
            return np.sum(diff**2) + self.alpha*np.sum(w)

        def df(w):
            return self.alpha*np.sign(w) - 2*np.dot(X.T, y - np.dot(X, w))

        def ddf(w):
            return -2*np.dot(X.T, X) + np.diag(self.alpha*(w == 0)*100000000)

        res = minimize(f, np.zeros(M), method='Newton-CG', jac=df, hess=ddf)
        self.w = res.x

    def predict(self, x):
        return np.dot(self.w, x.T)


def polynomial(x, degree=3):
    """generate a design matrix X for polynomial regression"""
    return np.column_stack([x**n for n in range(degree+1)])

def gaussian(x, low=-1, high=1, num=10, scale=None):
    """generate a design matrix X from gaussians"""
    if scale is None:
        # this setting ensures a decent amount of overlap between
        # gaussians while still being distinct
        s = (high-low)/float(num)
    else:
        s = scale
    return np.column_stack([np.exp(-(x-mu)**2/(2*s**2))
                 for mu in np.linspace(low, high, num)])

def sigmoidal(x, low=-1, high=1, num=10, scale=None):
    """generate a design matrix from sigmoids"""
    if scale is None:
        s = (high-low)/float(num)
    else:
        s = scale
    return np.column_stack([1/(1 + np.exp(-(x-mu)/s))
                 for mu in np.linspace(low, high, num)])
