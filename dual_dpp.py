import numpy as np
import scipy.linalg as la
from numpy.linalg import eig
import pdb


def decompose_kernel(M):
	L={}
	L['M']= [M]
	D,V = eig(M)
	L['V']=np.real(V)
	L['D']=np.real(D)
    return L



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
def dpp(B, k=-1):
    C = decompose_kernel(np.dot(B.T,B))
    eigen_vals = C['D']
    eigen_vec= C['V']
    # eigen_vec= eigen_vec.T

    if k==-1:
        probs = eigen_vals / (1+eigen_vals)
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
        Y= Y.append(y_i)

        S = np.dot(B[y_i], V)
        r= np.argmax(np.abs(S) >0)
        V_r = V[:,r]
        S_r= S[r]
        nidx = list(range(r)) + list(range(r+1, len(S)))

        # V = V[:,nidx]
        # S = S[:,nidx]


        try:
            V = la.orth(V- np.outer(V_r,(S/S_r )))
        except:
            pdb.set_trace()

        num_v-=1
        print((num_v,y_i))

    Y.sort()
    out = np.array(Y)

    return out
