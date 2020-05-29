import numpy as np
from statistics import stdev
#from sympy.utilities.iterables import multiset_permutations


def unit_vector(vec):
    """
    Returns unit vector
    """
    return vec / np.linalg.norm(vec)


def cos_sim(v1, v2):
    """
    Returns cosine of the angle between two vectors
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.tensordot(v1_u, v2_u, axes=(-1, -1)), -1.0, 1.0)


def weat_association(W, A, B):
    """
    Returns association of the word w in W with the attribute for WEAT score.
    s(w, A, B)
    :param W: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: (len(W), ) shaped numpy ndarray. each rows represent association of the word w in W
    """
    return np.mean(cos_sim(W, A), axis=-1) - np.mean(cos_sim(W, B), axis=-1)




def weat_score(X, Y, A, B):
    """
    Returns WEAT score
    X, Y, A, B must be (len(words), dim) shaped numpy ndarray
    CAUTION: this function assumes that there's no intersection word between X and Y
    :param X: target words' vector representations
    :param Y: target words' vector representations
    :param A: attribute words' vector representations
    :param B: attribute words' vector representations
    :return: WEAT score
    """

    x_association = weat_association(X, A, B)
    y_association = weat_association(Y, A, B)


    tmp1 = np.mean(x_association, axis=-1) - np.mean(y_association, axis=-1)
    tmp2 = np.std(np.concatenate((x_association, y_association), axis=0))
    #print(tmp1,tmp2)
    return tmp1 / tmp2


V = np.loadtxt('Input vector file here')

f1 = open('Input vocab file here','r')
wl = f1.readlines()

for i in range(len(wl)):
	wl[i] = wl[i].strip()

X = ['male','man','boy','brother','him','his','son']
Y = ['female','woman','girl','sister','her','hers','daughter']




for i in range(len(X)):
    X[i] = V[wl.index(X[i])]
for i in range(len(Y)):
    Y[i] = V[wl.index(Y[i])]
    

w = [0.0]*1000   
for j in range(1000):
	A =[]; B = []
	a = np.random.randint(len(V), size = 10)
	b = np.random.randint(len(V), size = 10)
	for i in range(10):
		A.append(wl[a[i]]); B.append(wl[b[i]])
	#print(A,B)       for checking words added if needed
	for i in range(len(A)):
		A[i] = V[wl.index(A[i])]
	for i in range(len(B)):
		B[i] = V[wl.index(B[i])]

	w[j] = weat_score(X,Y,A,B)

print('random-trials-weat-score: ', np.mean(w), stdev(w))
