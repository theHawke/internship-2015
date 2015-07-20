# -*- coding: utf-8 -*-
import numpy as np

class kMeans:
    def __init__(self, X, n = 2):
        self._ncl = n
        self._dim = X.shape[1]
        self._X = X
        maxs = np.amax(X, axis=0)
        mins = np.amin(X, axis=0)
        self.mu = np.column_stack([np.random.uniform(mn, mx, n)
                                    for (mn, mx) in zip(mins, maxs)])
        self.E_step()

    def E_step(self):
        self.cl = np.array([np.argmin(np.sum((xn - self.mu)**2, axis=1))
                            for xn in self._X])

    def M_step(self):
        self.mu = np.array([np.mean(self._X[self.cl==i], axis=0)
                            for i in range(self._ncl)])

    def cost(self):
        return np.sum([(xn-self.mu[cln])**2 for (xn, cln) in zip(self._X, self.cl)])

    def run(self):
        cost = self.cost()
        while True:
            self.E_step()
            self.M_step()
            old_cost = cost
            cost = self.cost()
            if old_cost == cost:
                break

def GaussLikely(xy, mu, icov, dcov):
    """Inverse and determinant of the covariance matrix are
    passed seperately since they can be precomputed"""
    offs = xy - mu
    arg = - np.dot(offs, np.dot(icov, offs)) / 2
    return np.exp(arg) / np.sqrt(2*np.pi*dcov)

class GaussianMixtureEM:
    def __init__(self, X, n = 2):
        self._ncl = n
        self._dim = X.shape[1]
        self._N = X.shape[0]
        self._X = X
        maxs = np.amax(X, axis=0)
        mins = np.amin(X, axis=0)
        self.mu = np.column_stack([np.random.uniform(mn, mx, n)
                                    for (mn, mx) in zip(mins, maxs)])
        self.Sigma = np.repeat([np.diag((maxs-mins)**2/16)], n, axis=0)
        self._icov = np.array([np.linalg.inv(cov) for cov in self.Sigma])
        self._dcov = np.array([np.linalg.det(cov) for cov in self.Sigma])
        self.pi = np.ones(n)/n

        self.E_step()

    def E_step(self):
        unnorm = np.array([[self.pi[k]*GaussLikely(xn, self.mu[k],
                                                   self._icov[k], self._dcov[k])
                            for k in range(self._ncl)]
                           for xn in self._X])
        normf = np.sum(unnorm, axis=1)
        self.gamma = np.array([row/fact for (row, fact) in zip(unnorm, normf)])
        self.cl = np.argmax(self.gamma, axis=1)

    def M_step(self):
        N_k = np.sum(self.gamma, axis=0)

        self.mu = np.array([np.sum([self.gamma[n,k]*self._X[n]
                                    for n in range(self._N)], axis=0) / N_k[k]
                            for k in range(self._ncl)])

        self.Sigma = np.array([np.sum(
            [self.gamma[n,k]*np.outer(self._X[n]-self.mu[k], self._X[n]-self.mu[k])
             for n in range(self._N)], axis=0) / N_k[k]
                               for k in range(self._ncl)])
        self._icov = np.array([np.linalg.inv(cov) for cov in self.Sigma])
        self._dcov = np.array([np.linalg.det(cov) for cov in self.Sigma])

        self.pi = N_k / self._N

    def logLikelihood(self):
        return np.sum([np.log(
            np.sum([self.pi[k]*GaussLikely(xn, self.mu[k],
                                           self._icov[k], self._dcov[k])
                    for k in range(self._ncl)])) for xn in self._X])

    def run(self, threshold=0.0001):
        ll = self.logLikelihood()
        while True:
            self.E_step()
            self.M_step()
            old_ll = ll
            ll = self.logLikelihood()
            if (ll - old_ll)/ll < threshold:
                break
