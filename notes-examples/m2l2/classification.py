#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from copy import copy
from heapq import heapify, heappop
from scipy.optimize import fmin_tnc

def GaussLikely(xy, mu, icov, dcov):
    """Inverse and determinant of the covariance matrix are
    passed seperately since they can be precomputed"""
    offs = xy - mu
    arg = - np.dot(offs, np.dot(icov, offs)) / 2
    return np.exp(arg) / np.sqrt(2*np.pi*dcov)

class NaiveBayes:
    """A Gaussian Naive Bayes classifier,
    can do k-class classification"""

    def __init__(self, K=2):
        self._K = K

    def train(self, X, y):
        data = [X[y == i] for i in range(self._K)]
        self._mu = [np.mean(Xp, axis = 0) for Xp in data]
        cov = [np.cov(Xp, rowvar=0) for Xp in data]
        self._dcov = map(np.linalg.det, cov)
        self._icov = map(np.linalg.inv, cov)
        tot = float(len(X))
        self._wght = [float(len(Xp))/tot for Xp in data]

    def classify(self, x):
        scores = [w*GaussLikely(x, mu, icov, dcov) for (mu, icov, dcov, w)
                  in zip(self._mu, self._icov, self._dcov, self._wght)]
        if self._K == 2:
            return np.log(scores[1]/scores[0])
        else:
            return np.argmax(scores)


class DA:
    """A Discriminant Analysis classifier, can be used for linear
    (default) or quadratic discriminant analysis."""

    def __init__(self, datype='linear'):
        if datype == 'quadratic':
            self._linear = False
        else:
            self._linear = True

    def train(self, X, y):
        X0 = X[y == 0]
        X1 = X[y == 1]

        mu0 = np.apply_along_axis(np.mean, 0, X0)
        mu1 = np.apply_along_axis(np.mean, 0, X1)

        cov0 = np.cov(X0-mu0, rowvar=0)
        cov1 = np.cov(X1-mu1, rowvar=0)

        if self._linear:
            cov = (cov0 + cov1)/2
            cov0, cov1 = cov, cov

        icov0 = np.linalg.inv(cov0)
        icov1 = np.linalg.inv(cov1)

        self.A = (icov1 - icov0) / 2

        self.b = np.dot(icov0, mu0) - np.dot(icov1, mu1)

        self.c = (np.log(np.linalg.det(cov1)/np.linalg.det(cov0))
                  + np.dot(mu1, np.dot(icov1, mu1))
                  - np.dot(mu0, np.dot(icov0, mu0))) / 2

    def classify(self, x):
        return -(np.dot(x, np.dot(self.A, x)) + np.dot(self.b, x) + self.c)


def abssq(x):
    """returns the squared norm ⟨x,x⟩ of the input vector"""
    return np.dot(x,x)

class SVM:
    """A two-class Support Vector Machine Classifier,
    can use a custom kernel function (default is linear)."""

    def _rbf(self, x1, x2):
        diff = x1-x2
        if diff.ndim > 1:
            asq = np.apply_along_axis(abssq, 1, diff)
        else:
            asq = abssq(diff)
        return np.exp(self._nitssq*asq)

    def __init__(self, kernel='linear', sigma=1, customK=None):
        if kernel == 'rbf':
            self._kernel = self._rbf
            self._nitssq = - 1 / (2 * sigma**2)
        elif kernel == 'custom':
            if customK == None:
                raise ValueError("A custom kernel function has to be provided when using the 'custom' option for the kernel.")
            self._kernel = customK
        else:
            self._kernel = np.dot

    def train(self, X, y):
        X = np.column_stack((np.ones(y.size), X))
        ci = np.array([-1 if x == 0 else 1 for x in y])

        # the function L(α) and its derivative are negated to use minimize
        kfX = np.array([self._kernel(X,X[j]) for j in range(y.size)])

        def L(a):
            aci = a * ci
            return 0.5*np.sum(np.outer(aci,aci) * kfX) - np.sum(a)

        def dL(a):
            return ci * np.sum(np.reshape(a*ci,(-1,1))*kfX, axis=0) - 1

        bs = [(0, 1)] * y.size
        self.alpha,_,_ = fmin_tnc(L, np.ones(y.size)/2, fprime=dL, bounds=bs, disp=0)

        self._aci = (self.alpha*ci)[self.alpha != 0]
        self._X = X[self.alpha != 0]

    def classify(self, x):
        xh = np.array([1, x[0], x[1]])
        return np.sum(self._aci*self._kernel(self._X,xh))


class kNN:
    """A k-Nearest-Neighbours classifier (implementation not (yet) very efficient)"""

    def __init__(self, k = 7):
        self._k = k

    def train(self, X, y):
        self._X = X
        self._y = y
        self._K = np.unique(y).size

    def classify(self, x):
        # implementation is very inefficient, has complexity
        # O(|X| + k log|X|) for every call to classify
        h = [(np.linalg.norm(p-x), c) for (p, c) in zip(self._X, self._y)]
        heapify(h)
        c = np.zeros(self._K)
        i = 0
        while i < self._k or np.size(c[c == np.max(c)]) > 1:
            c[heappop(h)[1]] += 1
            i += 1
        return np.argmax(c)


class OVA:
    """Use a two-class Classifier to do multi-class classification with
    the One-versus-All (One-versus-Many, One-versus-Rest) strategy"""

    def __init__(self, K, classifier):
        """The classifier should be an object with the train(X,y) and
        classify(point) methods. The classifier should work on two
        classes labeled '0' and '1' and classify(point) should give a
        positive score for class '1' and a negative score for class '0'."""
        self._clsf = [copy(classifier) for i in range(K)]
        self._K = K

    def train(self, X, y):
        """Classes should be labeled 0 to num_classes-1"""
        # 1 indicates the single class, 0 the rest
        labels = [np.array([1 if c == i else 0 for c in y])
                  for i in range(self._K)]

        for (cls, lab) in zip(self._clsf, labels):
            cls.train(X, lab)

    def classify(self, x):
        """Will return the class with the maximum vs-Rest score"""
        scores = [cls.classify(x) for cls in self._clsf]
        return np.argmax(scores)


class OVO:
    """Use a two-class Classifier to do multi-class classification with
    the One-versus-One strategy"""

    def __init__(self, K, classifier):
        """The classifier should be an object with the train(X,y) and
        classify(point) methods. The classifier should work on two
        classes labeled '0' and '1' and classify(point) should give a
        positive score for class '1' and a negative score for class '0'."""
        self._clsf = [[copy(classifier) for j in range(i)]
                      for i in range(K)]
        self._K = K

    def train(self, X, y):
        """Classes should be labeled 0 to num_classes-1"""
        for i in range(self._K):
            for j in range(i):
                sel = np.array(map(lambda c: c == i or c == j, y))
                yp = np.array([1 if c == i else 0 for c in y[sel]])
                self._clsf[i][j].train(X[sel], yp)

    def classify(self, x):
        """Will return the class with the most votes from vs-One classifiers"""
        res = [[cls.classify(x) > 0 for cls in row] for row in self._clsf]
        scores = np.array([np.sum(np.array(
            [int(res[i][j]) if j < i else int(not res[j][i]) if i < j else 0
             for j in range(self._K)])) for i in range(self._K)])
        return np.argmax(scores)
