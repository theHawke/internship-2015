# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
from scipy.misc import lena, imshow

class RVM:
    def fit(Phi, t):

        M, N = Phi.get_shape()

        assert(N == t.size)

        beta = 10/np.var(t)

        PhiT = Phi.transpose()

        initphi = Phi[:,0]
        initphinorm = initphi.transpose().dot(initphi).toarray()[0,0]
        initalpha = initphinorm / (
            initphi.transpose().dot(t)[0]**2 / initphinorm - beta)

        initSigma = 1 / (initalpha + beta*initphinorm)
        initmu = beta * initSigma * initphi.transpose().dot(t)[0]

        Cinv = sp.eye(N,N).tocsc()/beta
        Cinv[0,0] = 1/(beta + initalpha*initphinorm)

        phiTC = PhiT.dot(Cinv)

        S = phiTC.dot(Phi).diagonal()
        Q = phiTC.dot(t)

        Sigma = np.array([[initSigma]])
        mu = np.array([initmu])

        alpha = np.array([initalpha])

        # tha array 'indices' keeps track of where the values correspondig to
        # different basis functions are located in the arrays alpha, mu and Sigma
        # and is used to get the correct collumns from Phi
        indices = np.array([0])

        L = 0

        i = 0

        while True:
            i = (i+1) % M
            if np.nonzero(indices == i)[0].size == 0:
                # phi_i is not in the model at the moment ...
                if Q[i]**2 > S[i]:
                    # ... but we add it to the model
                    s = S[i]
                    q = Q[i]

                    deltaL = 0.5 * ((q**2 - s)/s + np.log(s/q**2))

                    # update alphas with new value
                    alpha_i = s**2/(q**2 - s)
                    alpha = np.append(alpha, alpha_i)

                    Sigma_ii = 1/(alpha_i + s)

                    mu_i = Sigma_ii * q

                    # quantities used multiple times later
                    phi_i = Phi[:,i]
                    Phi_ = Phi[:,indices]
                    Phi_T = Phi_.transpose()
                    Phi_T_phi_i = Phi_T.dot(phi_i).toarray()[:,0]

                    # common factor used in the other quantities: Sigma Phi^T phi_i
                    commF = np.dot(Sigma, Phi_T_phi_i)

                    # update all the S and Q values
                    # PhiT includes all collumns, while Phi_T includes only the
                    # relevant ones
                    e_m = beta * (PhiT.dot(phi_i).toarray()[:,0]
                                  - beta*PhiT.dot(Phi_).dot(commF))
                    S = S - Sigma_ii * e_m**2
                    Q = Q - mu_i * e_m

                    # update mu
                    mu = np.r_[mu - mu_i * beta * commF, [mu_i]]

                    # update Sigma
                    Sigma = np.r_[
                        np.c_[Sigma + beta**2 * Sigma_ii * np.outer(commF, commF),
                              - beta**2 * Sigma_ii * commF],
                        np.c_[[- beta**2 * Sigma_ii * commF], Sigma_ii]]

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

                Sigma_j = Sigma[:,j]
                Sigma_jj = Sigma_j[j]
                mu_j = mu[j]

                Phi_ = Phi[:,indices]

                commF = beta * PhiT.dot(Phi_).dot(Sigma_j)

                if q**2 > s:
                    # ... and we re-estimate its coefficients
                    oldalpha = alpha[j]
                    alpha_i = s**2/(q**2 - s)
                    alpha[j] = alpha_i

                    deltaL = 0.5 * (Q[i]**2/(S[i] + 1/(1/alpha_i - 1/oldalpha)) -
                                    np.log(1 + S[i] * (1/alpha_i - 1/oldalpha)))

                    k_j = 1/(Sigma_jj + 1/(alpha_i - oldalpha))

                    Sigma = Sigma - k_j * np.outer(Sigma_j, Sigma_j)

                    mu = mu - k_j * mu_j * Sigma_j

                    S = S + k_j * commF**2
                    Q = Q + k_j * mu_j * commF


                else:
                    # ... but we remove it from the model
                    deltaL = 0.5 * (Q[i]**2/(S[i] - alpha[j]) - np.log(1 - S[i]/alpha[j]))

                    Sigma = Sigma - np.outer(Sigma_j, Sigma_j)/Sigma_jj
                    mu = mu - mu_j*Sigma_j/Sigma_jj

                    S = S + commF**2/Sigma_jj
                    Q = Q + mu_j*commF/Sigma_jj

                    alpha = np.delete(alpha, j)
                    mu = np.delete(mu, j)
                    Sigma = np.delete(np.delete(Sigma, j, axis=0), j, axis=1)
                    indices = np.delete(indices, j)

            if np.isfinite(deltaL) and deltaL > 0:
                L = L + deltaL
                if deltaL/L < 1e-7:
                    # final check on all basis functions
                    inQ = Q[indices]
                    inS = S[indices]
                    inq = alpha*inQ/(alpha-inS)
                    ins = alpha*inS/(alpha-inS)
                    inCond = np.all(inq**2 > ins)

                    outq = np.delete(Q, indices)
                    outs = np.delete(S, indices)
                    outCond = np.all(outq**2 <= outs)

                    if inCond and outCond:
                        break

        self.mu = mu
        self.ind = indices
        self.alpha = alpha
        self.M = M
        return indices, mu, alpha

    def predict(Phi_x_j):
        w = uncompress(self.ind, self.mu, self.M)
        return Phi_x_j.dot(w)


def uncompress(pos, val, n):
    arr = np.zeros(n)
    for i in range(pos.size):
        arr[pos[i]] = val[i]
    return arr

###################
# Testing FastRVM #
###################
def _posmat(n):
    return [sp.coo_matrix(([1], [[i],[j]]), shape=(n,n))
              for i in range(n) for j in range(n)]

def _hb_sp(size, scale):
    if size == 0:
        return [sp.eye(1,1)]
    pos = _posmat(2**(size-1))
    if scale == 1:
        hbp = [sp.kron(p, [[1,1],[1,1]])/2 for p in pos]
    else:
        hbp = [sp.kron(p, [[1,1],[1,1]])/2
               for p in haarbasis_sparse(size-1, scale-1)]
    tr = [sp.kron(p, [[1,-1],[1, -1]])/2 for p in pos]
    bl = [sp.kron(p, [[1,1],[-1, -1]])/2 for p in pos]
    br = [sp.kron(p, [[1,-1],[-1, 1]])/2 for p in pos]
    return (hbp + tr + bl + br)

def haarbasis_sparse(size, scale = 0):
    a = _hb_sp(size, scale)
    # only the linked list sparse format implements reshape()
    return sp.hstack([m.tolil().reshape((1,2**(2*size))).tocsr().transpose()
                      for m in a], 'csc')
