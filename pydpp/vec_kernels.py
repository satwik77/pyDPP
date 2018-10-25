import difflib
import pdb
import gensim
from time import time
from scipy.spatial.distance import pdist, squareform
import scipy
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pickle

embedding_path = '/scratchd/home/satwik/embeddings/'
print('Loading Word2Vec')
st_time = time()
with open('/scratchd/home/satwik/embeddings/quora_word2vec.pickle', 'rb') as handle:
    model = pickle.load(handle)

print('Word2vec Loaded')
etime = (time() - st_time)/60.0
print('Time Taken : {}'.format(etime))

# print('Loading Word2Vec')
# st_time = time()
# model = gensim.models.KeyedVectors.load_word2vec_format(embedding_path+'GoogleNews-vectors-negative300.bin.gz', binary=True)
# print('Word2vec Loaded')
# etime = (time() - st_time)/60.0
# print('Time Taken : {}'.format(etime))

cos_sim = lambda a,b: dot(a, b)/(norm(a)*norm(b))
rbf = lambda a,b, sigma : scipy.exp(-(np.sum( (a-b)**2 ) ** 2 )/ sigma ** 2)

def sent2wvec(s):
    v= []
    for w in s:
        try:
            vec =  model[w]
            v.append(vec)
        except:
            pass

    v = np.array(v)
    return v



def sentence_compare(s1, s2, kernel='cos', **kwargs):
    l1 = s1.split()
    l2 = s2.split()
    # pdb.set_trace()

    v1= sent2wvec(l1)
    v2= sent2wvec(l2)
    # v2 = np.array([model.wv.word_vec(w) for w in l2])
    score = 0
    len_s1 = v1.shape[0]
    # pdb.set_trace()
    for v in v1:
        if kernel == 'cos':
            wscore = np.max(np.array([cos_sim(v,i) for i in v2] ))
        elif kernel == 'rbf':
            wscore = np.max(np.array([rbf(v,i, kwargs['sigma']) for i in v2] ))
        else:
            print('Error in kernel type')
        score += wscore/len_s1

    return score


def sent_cosine_sim(X):
    d=[]

    for i in range(len(X) ):
        td=[]
        for j in range(len(X) ):
            td.append(sentence_compare(X[i], X[j], 'cos'))
        d.append(td)
        # pdb.set_trace()
    A= np.array(d)
    print(A.shape)
    V = (0.5*A)+ (0.5*A.T)

    return V



def sent_rbf(X, sigma=0.5):
    d=[]

    for i in range(len(X) ):
        td=[]
        for j in range(len(X) ):
            td.append(sentence_compare(X[i], X[j], kernel='rbf', sigma=sigma))
        d.append(td)
        # pdb.set_trace()
    A= np.array(d)
    print(A.shape)
    V = (0.5*A)+ (0.5*A.T)

    return V

if __name__ == '__main__':
    sents =[]
    sents.append('what is best way to make money online' )
    sents.append('what should i do to make money online' )
    sents.append('what should i do to earn money online' )
    sents.append('what is the easiest way to make money online' )
    sents.append('what is the easiest way to earn money online' )
    sents.append('what s the easiest way to make money online' )
    sents.append('what s the easiest way to earn money online' )
    sents.append('what should i do to make money online online' )
    sents.append('what is the best way to make money online' )
    sents.append('what is the easiest way to make money online online' )

    sent_cosine_sim(sents)


