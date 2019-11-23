# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 15:25:58 2019

@author: deepak
"""

import numpy as np

def customAccuracy(A,F) :
    err = np.empty((A.shape[0],A.shape[1]), float)
    loss = []
    for i in range(A.shape[0]) :
        for j in range(A.shape[1]) :
            if A[i,j] == 0 :
                err[i,j] = F[i,j] = 0
            else :
                err[i,j] = np.abs(A[i,j] - F[i,j])/A[i,j]
    loss = np.mean(err, axis=1)
    loss = loss * 100
    return loss




D = np.full((2, 2), 10)
V = np.full((2, 2), 8)
acc = customAccuracy(D,V)

print(acc)