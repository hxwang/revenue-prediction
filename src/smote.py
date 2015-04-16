#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Created on 24.11.2012

<at> author: karsten

This is an implementation of the SMOTE Algorithm. 
See: "SMOTE: synthetic minority over-sampling technique" by
Chawla, N.V et al.

slightly modified by Zhonghua Xi for regression problem
'''

import numpy as np
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors

def SMOTE(X, y, N, k):
    """
    Returns (N/100) * n_minority_samples synthetic minority samples.

    Parameters
    ----------
    X : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    y : array, shape = [n_minority_samples, 1]
        Holds the value of samples
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 

    Returns
    -------
    SX : array, shape = [(N/100) * n_minority_samples, n_features]
    Sy : array, shape = [(N/100) * n_minority_samples, 1]
    """    
    n_minority_samples, n_features = X.shape
    
    if N < 100:
        #create synthetic samples only for a subset of X.
        #TODO: select random minortiy samples
        N = 100
        pass

    if (N % 100) != 0:
        raise ValueError("N must be < 100 or multiple of 100")
    
    N = N/100
    n_synthetic_samples = N * n_minority_samples
    SX = np.zeros(shape=(n_synthetic_samples, n_features))
    Sy = np.zeros(shape=(n_synthetic_samples))
    
    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(X)
    
    #Calculate synthetic samples
    for i in xrange(n_minority_samples):
        nn = neigh.kneighbors(X[i], return_distance=False)
        for n in xrange(N):
            nn_index = choice(nn[0])
            #NOTE: nn includes T[i], we don't want to select it 
            while nn_index == i:
                nn_index = choice(nn[0])
                
            difX = X[nn_index] - X[i]
            dify = y[nn_index] - y[i]

            gap = np.random.random()

            SX[n + i * N, :] = X[i,:] + gap * difX[:]
            Sy[n + i * N] = y[i] + gap * dify
    
    return SX, Sy