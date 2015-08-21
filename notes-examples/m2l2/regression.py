# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import inv, solve, qr, solve_triangular, svd
from scipy.sparse.linalg import lsqr
from scipy.optimize import minimize


class LinearModel:
    def __init__(self, basis, **kwargs):
        self._basis = basis
        self._init(**kwargs)

    def _init(self):
        pass

    def fit(self, X, y):
        # generate the design matrix
        Xp = self._basis.generate(X)
        self._fit(Xp, y)

    def _fit(self, X, y):
        self.w = np.zeros(X.shape[1])

    def predict(self, x, **kwargs):
        # generate the predictor matrix
        xp = self._basis.generate(x)
        return self._predict(xp, **kwargs)

    def _predict(self, x):
        # default implementation
        # fit() must have defined self.w
        return np.dot(x, self.w)

class OLS(LinearModel):
    def _init(self, method='svd'):
        self._m = method

    def _fit(self, X, y):
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


class RidgeRegression(LinearModel):
    def _init(self, alpha, method='svd'):
        self._m = method
        self.alpha = alpha

    def _fit(self, X, y):
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


class Lasso(LinearModel):
    def _init(self, alpha):
        self.alpha=alpha

    def _fit(self, X, y):
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


class Bayesian(LinearModel):
    def _fit(self, X, y):
        # preliminary calculations
        N, M = X.shape

        assert(M <= N) #TODO: make sure it works for M > N

        # SVD, X = U.Σ.V^T, s = diag(Σ)
        _, s, VT = svd(X, full_matrices=False)

        # X^T.y
        XT_y = np.dot(X.T, y)

        # eigenvalues of X^T.X, λ' = λ/β
        lambdap = s**2

        # initial values
        alpha = 1
        beta = 1

        # count iterations
        i = 0

        while True:
            i = i + 1
            # unfortunatley, lambda is a reserved keyword no utf-8 variables
            lambda_i = beta*lambdap
            # inverse of the diagonal matrix:
            # (I + α/β Σ^-2)^-1 = diag(1/(1 + α/λ))
            invpart = np.diag(1 / (1 + alpha / lambda_i))
            m_N = beta/alpha * np.dot(np.eye(M) -
                                      np.dot(VT.T, np.dot(invpart, VT)), XT_y)
            gamma = np.sum(lambda_i/(alpha + lambda_i))

            # new values for α, β
            alpha_old = alpha
            beta_old = beta
            alpha = gamma/np.dot(m_N, m_N)
            diff = y - np.dot(X, m_N)
            beta = (N-gamma)/np.sum(diff**2)

            if (np.abs(beta-beta_old)/beta_old < 1e-6 and
                np.abs(alpha-alpha_old)/alpha_old < 1e-6):
                break

        self.it = i

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        lambda_i = beta*lambdap
        invpart = np.diag(1 / (1 + alpha / lambda_i))
        self.S_N = 1/alpha * (np.eye(M) - np.dot(VT.T, np.dot(invpart, VT)))
        self.m_N = beta * np.dot(self.S_N, XT_y)

    def _predict(self, x, return_variance=False):
        res = np.dot(x, self.m_N)
        if return_variance:
            var = 1/self.beta + np.einsum('ij,ik,jk->i', x, x, self.S_N)
            return res, var
        else:
            return res


class PLS(LinearModel):
    def _fit(self, X, y, l = None):
        N, M = X.shape

        if l is None:
            l = M

        # preallocate arrays
        w = np.empty((M,M))
        t = np.empty((N,M))
        p = np.empty((M,M))
        q = np.empty(M)

        w0 = np.dot(X.T, y)
        w[:,0] = w0/np.linalg.norm(w0)
        t[:,0] = np.dot(X, w[:,0])

        for k in range(l):
            t_k = np.dot(t[:,k], t[:,k])
            t[:,k] = t[:,k]/t_k
            p[:,k] = np.dot(X.T, t[:,k])
            q[k] = np.dot(y, t[:,k])
            if q[k] == 0:
                # go until rank(X) is 0, i.e. all its entries are zero
                l = k
                break
            if k < l-1:
                X = X - t_k * np.outer(t[:,k], p[:,k])
                w[:,k+1] = np.dot(X.T, y)
                t[:,k+1] = np.dot(X, w[:,k+1])

        # shrink the arrays
        if l < M:
            w = w.resize((M,l))
            t = t.resize((N,l))
            p = p.resize((M,l))
            q = q.resize(l)

        self.B = np.dot(w, np.dot(np.linalg.inv(np.dot(p.T, w)), q))
        self.B_0 = q[0] - np.dot(p[:,0], self.B)

    def _predict(self, x):
        return np.dot(x, self.B) + self.B_0


#
# Basis generators to be used with the LinearModels
#
class Polynomial:
    def __init__(self, degree=5):
        """generate a design matrix X for polynomial regression"""
        self.d = degree

    def generate(self, x):
        return np.column_stack([x**n for n in range(self.d+1)])


class Gaussian:
    def __init__(self, low=-1, high=1, num=10, scale=None):
        """generate a design matrix X from gaussians"""
        self.l = low
        self.h = high
        self.n = num
        if scale is None:
            # this setting ensures a decent amount of overlap between
            # gaussians while still being distinct
            self.s = (high-low)/float(num)
        else:
            self.s = scale

    def generate(self, x):
        return np.column_stack([np.exp(-(x-mu)**2/(2*self.s**2))
                                for mu in np.linspace(self.l, self.h, self.n)])


class Sigmoidal:
    def __init__(self, low=-1, high=1, num=10, scale=None):
        """generate a design matrix X from gaussians"""
        self.l = low
        self.h = high
        self.n = num
        if scale is None:
            # this setting ensures a decent amount of overlap between
            # sigmoids while still being distinct
            self.s = (high-low)/float(num)
        else:
            self.s = scale

    def generate(self, x):
        """generate a design matrix from sigmoids"""
        if scale is None:
            s = (high-low)/float(num)
        else:
            s = scale
            return np.column_stack([1/(1 + np.exp(-(x-mu)/self.s))
                                    for mu in np.linspace(self.l, self.h, self.n)])
