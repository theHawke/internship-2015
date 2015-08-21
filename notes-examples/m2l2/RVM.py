# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
from scipy.linalg import cholesky, solve_triangular

class RVM:
    def fit(self, Phi, t, beta=None):
        # Φ is the design matrix, in sparse matrix format
        Phi = sp.csc_matrix(Phi)

        N, M = Phi.get_shape()

        if N != t.size:
            raise ValueError("Dimensions of input arguments incompatile")

        if beta is None:
            beta = 10/np.var(t)

        # Φ doesn't change and we need this several times
        PhiT = Phi.transpose()

        # for the initial basis vector, we choose the first column of Φ
        phi0 = Phi[:,0]
        # initphinorm = φ₀^T.φ₀, initphi_t = φ₀^T.t
        phi0norm = phi0.transpose().dot(phi0).toarray()[0,0]
        phi0_t = phi0.transpose().dot(t)[0]
        # α₀ = |φ₀|²/(|φ₀^T.|²/|φ₀|² - 1/β)
        alpha = np.array([phi0norm / (phi0_t**2 / phi0norm - 1/beta)])

        # tha array 'indices' keeps track of where the values correspondig to
        # different basis functions are located in the arrays alpha, mu and
        # Sigma and is used to get the correct collumns from Phi
        indices = np.array([0])

        # used to compute S, Q
        Sbase = PhiT.dot(Phi).diagonal()
        Qbase = PhiT.dot(t)

        L = 0

        it = 0

        while True:
            it = it+1

            # Bases currently in the model (used multiple times)
            Phi_ = Phi[:,indices]
            Phi_T = Phi_.transpose()

            # recompute Σ by explicit inversion of (A + β Φ^TΦ)
            c = cholesky(np.diag(alpha) + beta*Phi_T.dot(Phi_).toarray())
            cinv = solve_triangular(c, np.eye(c.shape[0]))
            Sigma = np.dot(cinv, cinv.T)

            mu = beta * np.dot(Sigma, Phi_T.dot(t))

            # recompute S, Q
            basep = PhiT.dot(Phi_).toarray()
            basep_Sigma = np.dot(basep, Sigma)
            S = beta * (Sbase - beta *
                        np.diag(np.dot(basep_Sigma, basep.T)))
            Phi_T_t = Phi_T.dot(t)
            Q = beta * (Qbase - beta * np.dot(basep_Sigma, Phi_T_t))

            assert(np.all(S > 0))
            assert(np.all(Sigma.T == Sigma))
            assert(np.all(np.linalg.eigvals(Sigma) > 0))
            assert(np.all(alpha > 0))

            # recompute beta
            beta = np.sum(alpha * np.diag(Sigma))/np.sum((t - Phi.dot(mu))**2)

            # choose a new basis function and recalculate alpha
            i = it % M
            if np.count_nonzero(indices == i) == 0:
                # phi_i is not in the model at the moment ...
                if Q[i]**2 > S[i]:
                    # ... but we add it to the model
                    deltaL = 0.5 * (Q[i]**2/S[i] - 1 - np.log(Q[i]**2/S[i]))

                    # update alphas with new value
                    alpha_i = S[i]**2/(Q[i]**2 - S[i])
                    alpha = np.append(alpha, alpha_i)

                    indices = np.append(indices, i)


                else:
                    # ... and we don't have to do anything
                    deltaL = 0

            else:
                # phi_i is already contained in the model ...
                j = np.nonzero(indices == i)[0][0]
                assert(indices[j] == i)
                s = alpha[j]*S[i]/(alpha[j] - S[i])
                q = alpha[j]*Q[i]/(alpha[j] - S[i])

                if q**2 > s:
                    # ... and we re-estimate its coefficients

                    deltaL = 0.5 * (Q[i]**2/S[i] - 1 - np.log(Q[i]**2/S[i]))

                    #old_a = alpha[j]
                    alpha_i = s**2/(q**2 - s)
                    alpha[j] = alpha_i


                else:
                    # ... but we remove it from the model
                    deltaL = 0.5 * (Q[i]**2/(S[i] - alpha[j]) -
                                    np.log(1 - S[i]/alpha[j]))

                    alpha = np.delete(alpha, j)
                    indices = np.delete(indices, j)

            assert(indices.size != 0)
            assert(deltaL >= 0)

            # check for termination
            if np.isfinite(deltaL) and deltaL > 0:
                L = L + deltaL
                if deltaL/L < 1e-7:
                    # final check on all basis functions
                    # check those currently in the model
                    inQ = Q[indices]
                    inS = S[indices]
                    inq = alpha*inQ/(alpha-inS)
                    ins = alpha*inS/(alpha-inS)
                    inCond = np.all(inq**2 > ins)

                    # check those currently not in the model
                    outq = np.delete(Q, indices)
                    outs = np.delete(S, indices)
                    outCond = np.all(outq**2 <= outs)

                    if inCond and outCond:
                        break

        # recompute Σ and μ one last time
        c = cholesky(np.diag(alpha) + beta*Phi_T.dot(Phi_).toarray())
        cinv = solve_triangular(c, np.eye(c.shape[0]))
        Sigma = np.dot(cinv, cinv.T)

        mu = beta * np.dot(Sigma, Phi_T.dot(t))

        self.mu = mu
        self.ind = indices
        self.M = M

        # diagnostic printouts
        print(indices)
        print(mu)
        print(alpha)
        print(beta)
        print(it)

    def predict(self, Phi_x_j):
        w = uncompress(self.ind, self.mu, self.M)
        return Phi_x_j.dot(w)


def uncompress(pos, val, n):
    arr = np.zeros(n)
    for i in range(pos.size):
        arr[pos[i]] = val[i]
    return arr
