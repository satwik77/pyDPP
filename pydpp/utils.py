#Author: Satwik Bhattamishra

import numpy as np
import scipy.linalg as la
from numpy.linalg import eig
import pdb


# Refer to paper: k-DPPs: Fixed-Size Determinantal Point Processes [ICML 11]

def elem_sympoly(lmbda, k):
    N = len(lmbda)
    E= np.zeros((k+1,N+1))
    E[0,:] =1
    for l in range(1,(k+1)):
        for n in range(1,(N+1)):
            E[l,n] = E[l,n-1] + lmbda[n-1]*E[l-1,n-1]
    return E

def sample_k_eigenvecs(lmbda, k):
    E = elem_sympoly(lmbda, k)
    i = len(lmbda)
    rem = k
    S = []
    while rem>0:
        if i==rem:
            marg = 1
        else:
            marg= lmbda[i-1] * E[rem-1,i-1]/E[rem,i]

        if np.random.rand()<marg:
            S.append(i-1)
            rem-=1
        i-=1
    S= np.array(S)
    return S
