# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 12:53:21 2019

@author: deepak
"""

#Adding custom loss function

import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from math import sqrt


def customSmapeLoss(yTrue,yPred):
    epsilon = 0.1
    summ = tf.maximum(tf.abs(yTrue) + tf.abs(yPred) + epsilon, 0.5 + epsilon)
    smape = tf.abs(yPred - yTrue) / summ * 2.0
    return smape


# A - Actual
# F - Forecast/Prediction

def smape(A, F):
    loss = []
    for i in range(A.shape[0]) :
        loss.append(100/len(A[i]) * np.sum(2 * np.abs(F[i] - A[i]) / (np.abs(A[i]) + np.abs(F[i]))))
    return loss

# Here we take the absolute difference between actual and forecast and divide by actual to take the proportionate error
# The error is subtracted from 100 to take the accuracy percentage
def lossPercent(A,F) :
    err = np.empty((A.shape[0],A.shape[1]), float)
    loss = []
    for i in range(A.shape[0]) :
        for j in range(A.shape[1]) :
            if A[i,j] == 0 :
                err[i,j] = F[i,j] = 0
            else :
                err[i,j] = np.abs(A[i,j] - F[i,j])/A[i,j]
    loss = np.mean(err, axis=1)
    return loss * 100

# Mean Absolute Error
def MAE(A,F) :
    mae = []
    for i in range(A.shape[0]) :
        mae.append(mean_absolute_error(A[i], F[i]))
    return mae

# Mean Squared Error
def MSE(A,F) :
    mse = []
    for i in range(A.shape[0]) :
        mse.append(mean_squared_error(A[i], F[i]))
    return mse

# Root Mean Squared Error
def RMSE(A,F) :
    rmse = []
    for i in range(A.shape[0]) :
        rmse.append(sqrt(mean_squared_error(A[i], F[i])))
    return rmse


