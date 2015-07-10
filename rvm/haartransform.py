#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc
from rvm import rvm



def haar1(block):
    a = block[0]+block[1]
    b = block[0]-block[1]
    c = block[2]+block[3]
    d = block[2]-block[3]
    block[0] = (a+c)
    block[1] = (b+d)
    block[2] = (a-c)
    block[3] = (b-d)

def blockise(arr):
    h, w = arr.shape
    return arr.reshape(h/2, 2, w/2, 2).transpose(0,2,1,3).reshape(h/2,w/2,4)

def unblockise(arr):
    hh, hw, _ = arr.shape
    return arr.reshape(hh, hw, 2, 2).transpose(0,2,1,3).reshape(2*hh, 2*hw)

def haart(arr):
    if arr.shape == (1,1):
        return arr
    ap = blockise(arr.copy())
    for row in ap:
        for block in row:
            haar1(block)
    app = ap.transpose(2,0,1)
    nt = haart(app[0])
    return np.concatenate((np.column_stack((nt, app[1])),
                           np.column_stack((app[2], app[3]))))

def invhaart(arr):
    if arr.shape == (1,1):
        return arr
    n = arr.shape[0]/2

    ap = np.array([invhaart(arr[:n,:n]), arr[:n,n:],
                            arr[n:,:n] , arr[n:,n:]]).transpose(1,2,0)

    for row in ap:
        for block in row:
            haar1(block)

    return unblockise(ap)/4

##########################################
def posmat(n):
    a = np.zeros((n*n,n,n))
    for i in range(n):
        for j in range(n):
            a[i*n + j, i, j] = 1
    return a

def haarbasis(n, scale = 0):
    if n == 0:
        return np.array([[[1]]])
    pos = posmat(2**(n-1))
    if scale == 1:
        hbp = np.kron(pos, np.ones((2,2)))
    else:
        hbp = np.kron(haarbasis(n-1, scale-1), np.ones((2,2)))
    tr = np.kron(pos, np.array([[1,-1],[1, -1]]))
    bl = np.kron(pos, np.array([[1,1],[-1, -1]]))
    br = np.kron(pos, np.array([[1,-1],[-1, 1]]))
    return np.concatenate((hbp, tr, bl, br))/2



basis16 = haarbasis(4,1).reshape(256,256).transpose()

blocks = misc.lena().reshape(32,16,32,16).transpose(0,2,1,3).reshape(1024,256)

def uncompress(pos, val, n):
    arr = np.zeros(n)
    for i in range(pos.size):
        arr[pos[i]] = val[i]
    return arr

#plt.imshow(invhaart(tf), cmap = cm.Greys_r)
#plt.show()
