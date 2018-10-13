import numpy as np
import scipy.linalg as la
from numpy.linalg import eig
import pdb


### Refer to paper: 1. k-DPPs: Fixed-Size Determinantal Point Processes [paper][ICML’11]

def elem_sympoly(lmbda, k):
    N = len(lmbda)
    E= np.zeros((k+1,N+1))
    E[0,:] =1
    for l in range(1,(k+1)):
        for n in range(1,(N+1)):
            E[l,n] = E[l,n-1] + lmbda[n-1]*E[l-1,n-1]
    return E

def sample_k(lmbda, k):
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
    print(S)
    return S

# implementation of algorithm 1
def dpp(A, k=-1):

    eigen_vals, eigen_vec = eig(A)
    eigen_vals =np.real(eigen_vals)
    eigen_vec =np.real(eigen_vec)
    eigen_vec = eigen_vec.T
    N =A.shape[0]
    Z= list(range(N))

    if k==-1:
        probs = eigen_vals/(eigen_vals+1)
        jidx = np.array(np.random.rand(N)<=probs)    # set j in paper

    else:
        jidx = sample_k(eigen_vals, k)

    V = eigen_vec[jidx]           # Set of vectors V in paper
    print(V.shape)
    num_v = len(V)
    print(num_v)

    Y = []
    while num_v>0:
        Pr = np.sum(V**2, 0)/np.sum(V**2)
        y_i=np.argmax(np.array(np.random.rand() <= np.cumsum(Pr), np.int32))

        # pdb.set_trace()
        Y.append(Z[y_i])
        print('--------')
        print(Z[y_i])
        Z.remove(Z[y_i])
        print(V.shape)
        print(V)
        ri = np.argmax(np.abs(V[:,y_i]) >0)
        V_r = V[ri]
        nidx = list(range(ri)) + list(range(ri+1, len(V)))
        # V = V[nidx]

        if num_v>0:
            try:
                V = la.orth(V- np.outer((V[:,y_i]/V_r[y_i]),V_r ))
            except:
                pdb.set_trace()

        num_v-=1
        print((num_v,y_i))

    Y.sort()
    out = np.array(Y)

    return out



