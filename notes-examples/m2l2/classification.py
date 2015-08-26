# -*- coding: utf-8 -*-
import numpy as np
from copy import copy
from heapq import heapify, heappop
from scipy.optimize import fmin_tnc, fmin_slsqp
from scipy.spatial import KDTree

def GaussLikely(xy, mu, var):
    """Inverse and determinant of the covariance matrix are
    passed seperately since they can be precomputed
    """
    offs = xy - mu
    arg = - np.sum(offs**2/var) / 2
    return np.exp(arg) / np.sqrt(2*np.pi*np.product(var))

class NaiveBayes:
    def train(self, X, y):
        # determine the number of classes
        self._cl = np.unique(y)
        self._K = self._cl.size

        # separate into the different classes
        data = [X[y == i] for i in self._cl]

        # calculate mean
        self._mu = [np.mean(Xp, axis = 0) for Xp in data]

        # calculate variances
        self._var = [np.var(Xp, axis=0) for Xp in data]

        # calculate prior class probabilities
        tot = float(len(X))
        self._prior = [float(len(Xp))/tot for Xp in data]

    def classify(self, x):
        scores = [pp*GaussLikely(x, mu, var) for (mu, var, pp)
                  in zip(self._mu, self._var, self._prior)]
        if self._K == 2:
            return np.log(scores[1]/scores[0])
        else:
            return self._cl[np.argmax(scores)]


class DA:
    """A Discriminant Analysis classifier, can be used for linear
    (default) or quadratic discriminant analysis.
    """

    def __init__(self, datype='linear'):
        if datype == 'quadratic':
            self._linear = False
        else:
            self._linear = True

    def train(self, X, y):
        # separate into the two classes
        X0 = X[y == 0]
        X1 = X[y == 1]

        # find means
        mu0 = np.apply_along_axis(np.mean, 0, X0)
        mu1 = np.apply_along_axis(np.mean, 0, X1)

        # find covariance matrices
        cov0 = np.cov(X0-mu0, rowvar=0)
        cov1 = np.cov(X1-mu1, rowvar=0)

        # for LDA, assume equal covariance for both classes
        if self._linear:
            cov = (cov0 + cov1)/2
            cov0, cov1 = cov, cov

        icov0 = np.linalg.inv(cov0)
        icov1 = np.linalg.inv(cov1)

        # classification is done according to the sign of
        # xAx + bx + c
        self.A = (icov1 - icov0) / 2

        self.b = np.dot(icov0, mu0) - np.dot(icov1, mu1)

        self.c = (np.log(np.linalg.det(cov1)/np.linalg.det(cov0))
                  + np.dot(mu1, np.dot(icov1, mu1))
                  - np.dot(mu0, np.dot(icov0, mu0))) / 2

    def classify(self, x):
        return -(np.dot(x, np.dot(self.A, x)) + np.dot(self.b, x) + self.c)


class kNN:
    """A k-Nearest-Neighbours classifier (implemented using KDTrees)"""

    def __init__(self, k = 3):
        self._k = k # number of nearest neighbours

    def train(self, X, y):
        # A KDTree is a datastructure that is aware of the spatial
        # distribution of its elements and can be used for efficient
        # nearest neighbour queries
        self._T = KDTree(X)
        self._y = y
        self._K = np.max(y) + 1 # number of classes

    def classify(self, x):
        # query the KDTree
        dist, ind = self._T.query(x, k=self._k)

        # for k == 1, KDTree.query returns only a value, not a list
        if self._k == 1:
            return self._y[ind]

        # count occurences of each class
        count = np.zeros(self._K)
        for i in ind:
            count[self._y[i]] += 1
        if np.size(count[count == np.max(count)]) == 1:
            # if there is a single class with the most nearest
            # neighbours, return it
            return np.argmax(count)
        else:
            # if there are multiple classes with nearest neighbours,
            # return the one with smallest mean distance
            cls = np.argwhere(count == np.max(count)).flatten()
            return cls[np.argmin([np.mean(dist[ind == cl]) for cl in cls])]


class PA:
    """Passive-Aggressive online-learning algorithm"""

    def train(self, X, y):
        # add bias component to each vector
        X = np.column_stack((np.ones(y.size),X))

        # label classes as -1, 1 instead of 0, 1
        ci = y * 2 - 1

        # initialise w
        w = np.zeros(X.shape[1])

        # create array to keep track of history
        self.ws = np.empty(X.shape)

        for i in range(y.size):
            if ci[i]*np.dot(w,X[i]) < 1:
                f = lambda x: np.dot(x-w,x-w)
                dfdx = lambda x: 2*(x-w)
                cons = lambda x: ci[i]*np.dot(x,X[i]) - 1
                # perform the constrained minimisation
                w = fmin_slsqp(f, X[i]/np.dot(X[i],X[i]),
                               eqcons=[cons], fprime=dfdx, disp=0)
            self.ws[i] = w

        self._w = w

    def classify(self, x):
        return self._w[0] + np.dot(x, self._w[1:])

    def getIterationData(self, i):
        return self.ws[i]


def abssq(x):
    """returns the squared norm ⟨x,x⟩ of the input vector"""
    return np.dot(x,x)

class SVM:
    """A two-class Support Vector Machine Classifier,
    can use a custom kernel function (default is linear).
    """

    def _rbf(self, x1, x2):
        """A Gaussian Radial Basis Function for use as a kernel"""
        diff = x1-x2
        if diff.ndim > 1:
            asq = np.apply_along_axis(abssq, 1, diff)
        else:
            asq = abssq(diff)
        return np.exp(self._nitssq*asq)

    def _biasDot(self, x1, x2):
        return np.dot(x1, x2) + 1

    def __init__(self, kernel='linear', sigma=1, customK=None):
        if kernel == 'rbf':
            self._kernel = self._rbf
            self._nitssq = - 1 / (2 * sigma**2)
        elif kernel == 'custom':
            if customK == None:
                raise ValueError(
                    "A custom kernel function has to be provided when using \
                    the 'custom' option for the kernel.")
            self._kernel = customK
        else: # if kernel == 'linear'
            self._kernel = self._biasDot

    def train(self, X, y):
        # label classes as -1, 1 instead of 0, 1
        ci = y * 2 - 1

        # the outer-product-via-kernel-function of the data,
        # n² invocations of the kernel function
        kfX = np.array([self._kernel(X,X[j]) for j in range(y.size)])

        # L(α) = 1/2 Sum_i,j c_i α_i c_j α_j k(x_i,x_j) - Sum_i α_i
        def L(a):
            aci = a * ci
            return 0.5*np.sum(np.outer(aci,aci) * kfX) - np.sum(a)

        def dLda(a):
            return ci * np.sum(np.reshape(a*ci,(-1,1))*kfX, axis=0) - 1

        bs = [(0, None)] * y.size # limit α to [0,inf]

        # minimise the cost function L(α)
        self.alpha,_,_ = fmin_tnc(L, np.ones(y.size), fprime=dLda, bounds=bs, disp=0)

        # only keep α, c_i and X for support vectors (α != 0)
        self._aci = (self.alpha*ci)[self.alpha != 0]
        self._SV = X[self.alpha != 0]

    def _class(self, x):
        return np.sum(self._aci*self._kernel(self._SV,x))

    def classify(self, x):
        if x.ndim > 1:
            return np.apply_along_axis(self._class, 1, x)
        else:
            return self._class(x)


class DecisionBranch:
    """A one-level decision tree used as a weak classifier for AdaBoost"""

    def train(self, X, y, w=None):
        if w is None:
            w = np.ones_like(y, dtype=np.float)

        # no of dimensions in the input points
        dim = X.shape[1]
        # number of training points
        n = X.shape[0] # == y.size

        error = np.zeros(dim, dtype=np.float)
        splitp = np.zeros(dim, dtype=np.float)
        gtlt = np.zeros(dim, dtype=np.bool)

        # for each dimension, choose the best splitting point
        for d in range(dim):
            Xp = X[:,d]
            # we need the points ordered in the current dimension
            order = np.argsort(Xp)

            # we want to find the position in the array which gives the
            # smallest weighted classification error. There are n+1 possible
            # points, since: i -> class boundary between X[i-1] and X[i].
            # Assuming class 1 is left of i and class 0 right,
            # left[i] is the sum of weighted errors up to i from the left,
            # right[i] is the sum of weighted errors down to i from the right,
            left = np.insert(np.cumsum((w*np.logical_not(y))[order]), 0, 0)
            right = np.append(np.cumsum((w*y)[order][::-1])[::-1], 0)
            # errsum[i] is the total error if the class boundary is at i
            errsum = left + right

            # mn is the point with minimum misclassification error 'mcle'
            mn = np.argmin(errsum)
            mcle = errsum[mn]
            # mx is the point with maximum misclassification error.
            # it is also the point with mimimum error if the classes were
            # reversed, i.e. class 1 right of boundary , class 0 left
            mx = np.argmax(errsum)
            revmcle = np.sum(w) - errsum[mx]

            # see which way around has better error
            # and store results for this dimension
            if mcle <= revmcle:
                error[d] = mcle
                gtlt[d] = True
                p = mn
            else:
                error[d] = revmcle
                gtlt[d] = False
                p = mx

            splitp[d] = Xp[order[0]] - 1 if p == 0 else ( # left of the leftmost
                        Xp[order[-1]] + 1 if p == n else ( # right of the rightmost
                            0.5 * (Xp[order[p-1]] + Xp[order[p]]) # between two points
                        ))

        # see which dimesion has the best error and set the parameters
        self._d = np.argmin(error)
        self._gtlt = gtlt[self._d]
        self._p = splitp[self._d]

    def classify(self, x):
        # for handeling array inputs
        if x.ndim == 1:
            xx = x[self._d]
        elif x.ndim == 2:
            xx = x[:,self._d]

        # depending on the chosen direction, label classes 1->1, 0->-1
        if self._gtlt:
            return (xx < self._p) * 2 - 1
        else:
            return (xx > self._p) * 2 - 1

class AdaBoost:
    """A classifier obtained by combining several weak classifiers"""

    def __init__(self, m, baseClassifier = DecisionBranch()):
        # construct m copies of the base classifier
        self._clsf = [copy(baseClassifier) for _ in range(m)]
        self._m = m

    def train(self, X, y):
        # initialise w_0 for the iterative process
        wm = np.ones_like(y, dtype=np.float)
        # label classes as -1, 1 instead of 0, 1
        ci = y * 2 - 1
        # create array to store alpha_m values
        a = np.zeros(self._m, dtype=np.float)

        for m in range(self._m):
            # train classifier m on weighted training samples
            self._clsf[m].train(X, y, wm)

            # see how much is misclassified
            mcl = self._clsf[m].classify(X) != ci

            # zero misclassification rate breaks the logarithm
            if np.sum(mcl) == 0:
                a[m] = 1
                continue

            # weighted misclassification rate
            em = np.sum(wm * mcl)/np.sum(wm)
            # calculate alpha for this classifier
            a[m] = np.log((1-em)/em)
            # calculate w_{m+1} for the next step
            wm *= np.exp(a[m] * mcl)

        self._alpha = a

    def classify(self, x):
        # find individual classifications
        ym = np.array([c.classify(x) for c in self._clsf])
        # return weighted sum
        return np.dot(ym, self._alpha)



class OVA:
    """Use a two-class Classifier to do multi-class classification with
    the One-versus-All (One-versus-Many, One-versus-Rest) strategy
    """

    def __init__(self, K, classifier):
        """The classifier should be an object with the train(X,y) and
        classify(point) methods. The classifier should work on two
        classes labeled '0' and '1' and classify(point) should give a
        positive score for class '1' and a negative score for class '0'.
        """
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
    the One-versus-One strategy
    """

    def __init__(self, K, classifier):
        """The classifier should be an object with the train(X,y) and
        classify(point) methods. The classifier should work on two
        classes labeled '0' and '1' and classify(point) should give a
        positive score for class '1' and a negative score for class '0'.
        """
        # builds a 'triangle' of one v one classifiers
        #         j
        #      0 1 2 3 4
        #   0  _
        #   1 |_|_
        # i 2 |_|_|_
        #   3 |_|_|_|_
        #   4 |_|_|_|_|
        #
        self._clsf = [[copy(classifier) for j in range(i)]
                      for i in range(K)]
        self._K = K # number of classes

    def train(self, X, y):
        """Classes should be labeled 0 to num_classes-1"""
        for i in range(self._K):
            for j in range(i):
                # fill the i,j cell of the triangle
                sel = np.array(map(lambda c: c == i or c == j, y))
                yp = np.array([1 if c == i else 0 for c in y[sel]])
                self._clsf[i][j].train(X[sel], yp)

    def classify(self, x):
        """Will return the class with the most votes from vs-One classifiers"""
        # get the result for each classifier in the triangle
        res = [[cls.classify(x) > 0 for cls in row] for row in self._clsf]
        # get the total score for each class
        scores = np.array([np.sum(
            [int(res[i][j]) if j < i else int(not res[j][i]) if i < j else 0
             for j in range(self._K)]) for i in range(self._K)])
        return np.argmax(scores)
