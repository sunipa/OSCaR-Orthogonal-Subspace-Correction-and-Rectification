import numpy as np
from scipy.spatial.distance import cosine,euclidean
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
import math


# For applying oscar to word embeddings that are not contextualized like GloVe or word2vec or FastText.
# Files needed:
# 1. Vector file : in numpy .txt format with one vector per line and no words
# 2. Vocab file corresponding to vector file with one word per line
# 3. Gender direction file - here 'he-she.txt' used
# 4. Occupation subspace file - here 'occGloveDir.txt' used
# Output saved as a numpy .txt file again with one vector per line corresponding to the input vocab file. 
# This file will be debiased by OSCar.


def cosine1(x, y):
	return np.dot(x,y.T)

def maxSpan(V1,V2):
	maxVal = -2000000
	for i in range(len(V1)):
		for j in range(len(V2)):

			dot = np.abs(np.matmul(V1[i],V2[j].T)/(np.linalg.norm(V1[i])*np.linalg.norm(V2[j])))
			if dot >= maxVal:
				maxVal = dot
				vec = np.vstack((V1[i]/np.linalg.norm(V1[i]),V2[j]/np.linalg.norm(V2[j])))	
	return V1[i]/np.linalg.norm(V1[i]),V2[j]/np.linalg.norm(V2[j])

a = np.asmatrix(np.loadtxt('Subspace_Vectors/he-she.txt')); b = np.asmatrix(np.loadtxt('Subspace_Vectors/occGloveDir.txt')); 

v1, v2 = maxSpan(a,b) 
v1 = np.asarray(v1).reshape(-1)
v2 = np.asarray(v2).reshape(-1)

def proj(u,a):
	return ((np.dot(u,a.T))*u)/(np.dot(u,u))

def basis(vec):
	v1 = vec[0]; v2 = vec[1]; 
	v2Prime = v2 - v1*float(np.matmul(v1,v2.T)); 
	v2Prime = v2Prime/np.linalg.norm(v2Prime)
	return v2Prime


def gsConstrained(matrix,v1,v2):
	v1 = np.asarray(v1).reshape(-1)
	v2 = np.asarray(v2).reshape(-1)
	u = np.zeros((np.shape(matrix)[0],np.shape(matrix)[1]))
	u[0] = v1
	u[0] = u[0]/np.linalg.norm(u[0])
	u[1] = v2 - proj(u[0],v2)
	u[1] = u[1]/np.linalg.norm(u[1])
	for i in range(0,len(matrix)-2):
		p = 0.0
		for j in range(0,i+2):	
			p = p + proj(u[j],matrix[i])
		u[i+2] = matrix[i] - p
		u[i+2] = u[i+2]/np.linalg.norm(u[i+2])
	return u


# testing
#v1 = np.random.rand(300)
#v2 = np.random.rand(300)
v1 = v1/np.linalg.norm(v1)
v2 = v2/np.linalg.norm(v2)
#x = np.random.rand(300)

U = np.identity(300)
U = gsConstrained(U,v1,basis(np.vstack((v1,v2))))



def rotation(v1,v2,x):
	v1 = np.asarray(v1).reshape(-1); v2 = np.asarray(v2).reshape(-1); x = np.asarray(x).reshape(-1)
	v2P = basis(np.vstack((v1,v2))); xP = x[2:len(x)]
	#x = (np.dot(x,v1),np.dot(x,v2P)) 
	v2 = (np.matmul(v2,v1.T),np.sqrt( 1 - (np.matmul(v2,v1.T)**2))); v1 = (1,0)
	thetaX = 0.0
	theta = np.abs(np.arccos(np.dot(v1,v2)))
	thetaP = (np.pi/2.0) - theta
	phi = np.arccos(np.dot(v1,x/np.linalg.norm(x)))
	d = np.dot([0,1],x/np.linalg.norm(x))
	if phi<thetaP and d>0:
		thetaX = theta*(phi/thetaP)
	elif phi>thetaP and d>0:
		thetaX = theta*((np.pi - phi)/(np.pi - thetaP))
	elif phi>=np.pi - thetaP and d<0:
		thetaX = theta*((np.pi-phi)/thetaP)
	elif phi<np.pi - thetaP and d<0:
		thetaX = theta*(phi/(np.pi-thetaP))
	R = np.zeros((2,2))
	R[0][0] = np.cos(thetaX); R[0][1] = -np.sin(thetaX)
	R[1][0] = np.sin(thetaX); R[1][1] = np.cos(thetaX)
	return np.hstack((np.matmul(R,x),xP))

def correction(U,v1,v2,x):
	if np.count_nonzero(x) != 0:
		return np.matmul(U.T,rotation(v1,v2,np.matmul(U,x)))
	else:
		return x

#print(correction(U,v1,v2,x))

	






X = np.loadtxt('Input vector file')


for i in range(len(X)):
	X[i] = correction(U,v1,v2,X[i])

np.savetxt('Save vector file in numpy txt format',X)

















