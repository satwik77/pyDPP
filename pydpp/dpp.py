# Authors: Satwik Bhattamishra

import numpy as np
import scipy.linalg as la
from numpy.linalg import eig
import pdb
from .utils import elem_sympoly, sample_k_eigenvecs
from .kernels import cosine_similary, rbf

# Refer to paper: k-DPPs: Fixed-Size Determinantal Point Processes [ICML 11]


class DPP():
    """

    Attributes
    ----------
    A : PSD/Symmetric Kernel


    Usage:
    ------

    >>> from pydpp.dpp import DPP
    >>> import numpy as np

    >>> X = np.random.random((10,10))
    >>> dpp = DPP(X)
    >>> dpp.compute_kernel(kernel_type='rbf', sigma=0.4)
    >>> samples = dpp.sample()
    >>> ksamples = dpp.sample_k(5)


    """

    def __init__(self, X=None,  A=None, **kwargs):
        self.X = X
        if A:
            self.A = A

    def compute_kernel(self, kernel_type='cos-sim', kernel_func=None, *args, **kwargs):
        if kernel_func == None:
            if kernel_type == 'cos-sim':
                self.A = cosine_similary(self.X )
            elif kernel_type == 'rbf':
                self.A =rbf(self.X, **kwargs)
        else:
            self.A = kernel_func(self.X, **kwargs)

    def sample(self):

        if not hasattr(self,'A'):
            self.compute_kernel(kernel_type='cos-sim')

        eigen_vals, eigen_vec = eig(self.A)
        eigen_vals =np.real(eigen_vals)
        eigen_vec =np.real(eigen_vec)
        eigen_vec = eigen_vec.T
        N = self.A.shape[0]
        Z= list(range(N))

        probs = eigen_vals/(eigen_vals+1)
        jidx = np.array(np.random.rand(N)<=probs)    # set j in paper


        V = eigen_vec[jidx]           # Set of vectors V in paper
        # print(V.shape)
        num_v = len(V)


        Y = []
        while num_v>0:
            Pr = np.sum(V**2, 0)/np.sum(V**2)
            y_i=np.argmax(np.array(np.random.rand() <= np.cumsum(Pr), np.int32))

            # pdb.set_trace()
            Y.append(y_i)
            V =V.T
            ri = np.argmax(np.abs(V[y_i]) >0)
            V_r = V[:,ri]


            if num_v>0:
                try:
                    V = la.orth(V- np.outer(V_r, (V[y_i,:]/V_r[y_i]) ))
                except:
                    pdb.set_trace()

            V= V.T

            num_v-=1
            # print((num_v,y_i))

        Y.sort()

        out = np.array(Y)

        return out

    def sample_k(self, k=5):

        if not hasattr(self,'A'):
            self.compute_kernel(kernel_type='cos-sim')

        eigen_vals, eigen_vec = eig(self.A)
        eigen_vals =np.real(eigen_vals)
        eigen_vec =np.real(eigen_vec)
        eigen_vec = eigen_vec.T
        N =self.A.shape[0]
        Z= list(range(N))

        if k==-1:
            probs = eigen_vals/(eigen_vals+1)
            jidx = np.array(np.random.rand(N)<=probs)    # set j in paper

        else:
            jidx = sample_k_eigenvecs(eigen_vals, k)

        V = eigen_vec[jidx]           # Set of vectors V in paper
        # print(V.shape)
        num_v = len(V)
        # print(num_v)

        Y = []
        while num_v>0:
            Pr = np.sum(V**2, 0)/np.sum(V**2)
            y_i=np.argmax(np.array(np.random.rand() <= np.cumsum(Pr), np.int32))

            # pdb.set_trace()
            Y.append(y_i)
            # print('--------')
            # print(y_i)
            # Z.remove(Z[y_i])
            # print(V.shape)
            # print(V)
            V =V.T
            try:
                ri = np.argmax(np.abs(V[y_i]) >0)
            except:
                print("Error: Check: Matrix PSD/Sym")
                exit()
            V_r = V[:,ri]
            # nidx = list(range(ri)) + list(range(ri+1, len(V)))
            # V = V[nidx]

            if num_v>0:
                try:
                    V = la.orth(V- np.outer(V_r, (V[y_i,:]/V_r[y_i]) ))
                except:
                    print("Error in Orthogonalization: Check: Matrix PSD/Sym")
                    pdb.set_trace()

            V= V.T

            num_v-=1
            # print((num_v,y_i))

        Y.sort()
        # print(Y)
        out = np.array(Y)

        return out





