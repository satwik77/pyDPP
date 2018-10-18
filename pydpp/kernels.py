from scipy.spatial.distance import pdist, squareform
import scipy
from numpy import dot
from numpy.linalg import norm
import numpy as np



def rbf(X, sigma=0.5):
	pairwise_dists = squareform(pdist(X, 'euclidean'))
	A = scipy.exp(-pairwise_dists ** 2 / sigma ** 2)
	return A

def cosine_similary(X):
	d=[]
	cos_sim = lambda a,b: dot(a, b)/(norm(a)*norm(b))
	for i in range(X.shape[0]):
		td=[]
		for j in range(X.shape[0]):
			td.append(cos_sim(X[i], X[j]))
		d.append(td)
	A= np.array(d)
	return A
