2#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np

def rvm(y, data_mask, Psi):
    # Psi: N x N
    # y: M
    # sum(data_mask) = M
    # (data_mask)_i <- {0, 1}

    N = Psi.shape[0]

    Omega = np.compress(data_mask, np.eye(N), axis=0)

    Phi = np.dot(Omega, Psi)

    alpha = np.ones(N)
    beta = 1

    alpha_t = 100000000
    delta_t = 0.00000001

    relevant = np.arange(N)

    i = 0

    while True:
        Sigma = np.linalg.inv(np.diag(alpha) + beta*np.dot(Phi.T, Phi))
        mu = beta * np.dot(Sigma, np.dot(Phi.T, y))
        gamma = 1 - alpha*np.diag(Sigma)

        alpha_old = alpha

        alpha = gamma/mu**2

        beta = (alpha.size - np.sum(gamma))/np.sum((y - np.dot(Phi, mu))**2)

        mask = np.ones_like(alpha)
        do_mask = False

        for i in range(alpha.size):
            if alpha[i] > alpha_t:
                mask[i] = 0
                do_mask = True

        if do_mask:
            alpha = np.compress(mask, alpha)
            Phi = np.compress(mask, Phi, axis=1)
            relevant = np.compress(mask, relevant)
            if alpha.size == 0:
                return mu, 0, relevant, alpha

        elif np.sum(np.abs(alpha - alpha_old)) < delta_t or i > 100:
            return mu, 1/beta, relevant, alpha

        i += 1
