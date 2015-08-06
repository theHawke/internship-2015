#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp

def fastRVM(t, Phi):

    M, N = Phi.get_shape()

    assert(N == t.size())

    beta = 1

    initphi = Phi[:,0]
    initphinorm = initphi.transpose().dot(initphi)
    initalpha = initphinorm / (
        initphi.transpose().dot(t)**2 / initphinorm - beta)

    initSigma = 1 / (initalpha + initphinorm)
    initmu = beta * initSigma * initphi.transpose().dot(t)

    Cinv = sp.eye(N,N).tocsc()/beta
    Cinv[0,0] = 1/(beta + initalpha*initphinorm)

    phiTC = initphi.transpose().dot(Cinv)

    S = phiTC.dot(initphi).diagonal()
    Q = phiTC.dot(t)

    Sigma = np.array([[initSigma]])
    mu = np.array([initmu])

    alpha = np.array([initalpha])

    # tha array 'indices' keeps track of where the values correspondig to
    # different basis functions are located in the arrays alpha, mu and Sigma
    # and is used to get the correct collumns from Phi
    indices = np.array([0])

    L = 0

    while True:
        i = np.random.randint(M)
        if np.nonzero(indices == i).size() == 0:
            # phi_i is not in the model at the moment ...
            if Q[i]**2 > S[i]:
                # ... but we add it to the model
                s = S[i]
                q = Q[i]

                deltaL = 0.5 * ((q**2 - s)/s + np.log(s/q**2))

                # update alphas with new value
                alpha_i = s**2/(q**2 - s)
                alpha = np.append(alpha, alpha_i)

                Sigma_ii = 1/(newalpha + s)

                mu_i = Sigma_ii * q

                # quantities used multiple times later
                phi_i = Phi[:,i]
                Phi_ = Phi[:,indices]
                Phi_T = Phi_.transpose()
                Phi_T_phi_i = Phi_T.dot(phi_i).todense()

                # common factor used in the other quantities: Sigma Phi^T phi_i
                commF = np.dot(Sigma, Phi_T_phi_i)

                # update all the S and Q values
                e_m = beta * (Phi_T_phi_i - beta * Phi_T.dot(Phi_).dot(commF))
                S = S - Sigma_ii * e_m**2
                Q = Q - mu_i * e_m

                # update mu
                mu = np.r_[mu - mu_i * beta * commF, mui]

                # update Sigma
                Sigma = np.r_[
                    np.c_[Sigma + beta**2 * Sigma_ii * np.outer(commF, commF),
                          - beta**2 * Sigma_ii * commF],
                    np.c_[[- beta**2 * Sigma_ii * commF], Sigma_ii]]

                indices = np.append(indices, i)

            else:
                # ... and we don't have to do anything
                continue

        else:
            # phi_i is already contained in the model ...
            j = np.nonzero(indices == i)[0]
            assert(indices[j] == i)
            s = alpha[ii]*S[i]/(alpha[ii] - S[i])
            q = alpha[ii]*Q[i]/(alpha[ii] - S[i])

            Sigma_j = Sigma[:,j]
            Sigma_jj = Sigma_j[j]
            mu_j = mu[j]

            Phi_ = Phi[:,indices]

            commF = beta * Phi_.transpose().dot(Phi_).dot(Sigma_j)

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
                # ... but we remove it from the modle
                deltaL = 0.5 * (Q[i]**2/(S[i] - alpha[j]) - np.log(1 - S[i]/alpha[j]))

                Sigma = Sigma - np.outer(Sigma_j, Sigma_j)/Sigma_jj
                mu = mu - mu_j*Sigma_j/Sigma_jj

                S = S + commF**2/Sigma_jj
                Q = Q + mu_j*commF/Sigma_jj

                alpha = np.delete(alpha, j)
                mu = np.delete(mu, j)
                Sigma = np.delete(np.delete(Sigma, j, axis=0), j, axis=1)

        L = L + deltaL
        if deltaL/L < 1e-7:
            # final check on all basis functions
            #TODO
            break

    return indices, mu, alpha


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
