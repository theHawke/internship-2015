#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import misc



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
    ap = blockise(arr)
    for row in ap:
        for block in row:
            haar1(block)
    app = ap.transpose(2,0,1)
    nt = haart(app[0])
    return np.concatenate((np.column_stack((nt, app[1])),
                           np.column_stack((app[2], app[3]))))
def invhaart(tf):
    if tf.shape == (1,1):
        return tf
    n = tf.shape[0]/2

    ap = np.array([reconstruct(tf[:n,:n]), tf[n:,:n],
                   tf[:n,n:],              tf[n:,n:]]).transpose(1,2,0)

    for row in ap:
        for block in row:
            haar1(block)

    return unblockise(ap)/4



tf = haart(misc.lena())



plt.imshow(invhaart(tf), cmap = cm.Greys_r)
plt.show()
