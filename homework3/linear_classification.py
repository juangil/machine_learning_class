import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pylab import *
from matplotlib import cm
import scipy


file = open('iris.data', 'r')
table = [row.strip().split(',') for row in file]
data = np.matrix(table)
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
label_biclass_1 = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 1}


def msEstim(X,T):
    return (np.linalg.inv(np.transpose(X)*X)) * (np.transpose(X)*T)

def softMax(z):
    nume = np.exp(z)
    denom = np.sum(nume)
    ret = nume / denom
    return ret

def gmEstim(X,T, nclass):
    msize = X[0].shape[1]
    U = np.zeros((nclass,msize))
    for i in range(0,nclass):
        msum = np.zeros(msize)
        Ni = 0
        for j in range(0, len(X[:,0])):
            tmp = (X[j,:])*(T[j,i])
            Ni += (T[j,i])
            msum = msum + tmp        
        msum = msum / Ni
        #print msum
        U[i,:] = msum
    #print U
    Sigma = np.zeros((msize,msize))
    for i in range(0, nclass):
        S = np.zeros((msize,msize))
        for j in range(0, len(X[:,0])):
            tmp = np.matrix((X[j,:])*(T[j,i]) - (U[i,:])*(T[j,i]))
            tmp2 = np.transpose(tmp)
            tmps = tmp2*tmp
            S = S + tmps
        Sigma = Sigma + S
    #print Sigma
    Pi_ml = np.zeros((1,nclass))
    for i in range(0, nclass):
        Ni = 0
        for j in range(0, len(X[:,0])):
            Ni = Ni + T[j,i]
        Pi_ml[0,i] = Ni / len(X[:,0])
    
    
    Sigma = np.matrix(Sigma[1:, 1:])    
    #print np.linalg.inv(Sigma)*np.transpose(mio)
    Wgm = np.zeros((nclass, msize))
    for i in range(0, nclass):
        uk = np.transpose(np.matrix(U[i,1:]))
        wi = np.linalg.inv(Sigma)*uk
        w0 = ((-0.5*np.transpose(uk))*(np.linalg.inv(Sigma)*uk)) + np.log(Pi_ml[0,i])
        #print w0, wi
        tmp = np.zeros((1, msize))
        tmp[0,0] = w0
        tmp[0,1:] = np.transpose(wi)
        wk = tmp
        #print wk
        Wgm[i,:] = wk
    Wgm = np.matrix(Wgm)   
    return Wgm

def assignVar(N):
    a = np.zeros((N,1))
    b = np.zeros((N,1))
    c = np.zeros((N,1))
    d = np.zeros((N,1))
    e = np.zeros((N,3))
    for i in range(0, N):
        tmp = data[:,0]
        a[i,0] = float(tmp[i,0])
        tmp = data[:,1]
        b[i,0] = float(tmp[i,0])
        tmp = data[:,2]
        c[i,0] = float(tmp[i,0])
        tmp = data[:,3]
        d[i,0] = float(tmp[i,0])
        tmp = data[:,4]
        e[i, label[tmp[i,0]]] = 1.
    unos=np.ones((N,1))
    XX = np.concatenate((unos,a,b,c,d),1)
    return np.matrix(XX), np.matrix(e)

def extractMax(Yestim, nclass):
    for i in range(len(Yestim[:,0])):
        mayor = -1000000.0
        idmayor = -1
        for j in range(nclass):
            if(Yestim[i,j] > mayor):
                mayor = Yestim[i,j]
                idmayor = j

        for j in range(nclass):
            if (j == idmayor):
                Yestim[i,j] = 1.
            else:
                Yestim[i,j] = 0.
    return Yestim

def evalAccuracy(Yestim, Tvalid, nclass):
    true_positives = np.array([0.,0.,0.])
    true_negatives = np.array([0.,0.,0.])
    false_positives = np.array([0.,0.,0.])
    false_negatives = np.array([0.,0.,0.])
    for i in range(len(Yestim[:,0])):
        for j in range(nclass):
            if(Tvalid[i,j] == 1. and Tvalid[i,j] == Yestim[i,j]):
                true_positives[j] += 1.
            if(Tvalid[i,j] == 0. and Tvalid[i,j] == Yestim[i,j]):
                true_negatives[j] += 1.
            if(Tvalid[i,j] == 0. and Yestim[i,j] == 1.):
                false_positives[j] += 1.
            if(Tvalid[i,j] == 1. and Yestim[i,j] == 0):
                false_negatives[j] += 1.
    #print true_positives, true_negatives, false_positives, false_negatives
    ret = np.sum(true_positives + true_negatives)/np.sum(true_positives + false_positives + false_negatives + true_negatives)
    return ret

def evaluate(Xtrain, Xtest, Ttrain, Ttest):
    # Least squares estimation
    Wmse = msEstim(Xtrain,Ttrain)
    Yestim = np.transpose(Wmse)*np.transpose(Xtest)
    Yestim = np.transpose(Yestim)
    #print Yestim
    Yestim = extractMax(Yestim, 3)
    lsq_err = evalAccuracy(Yestim, Ttest, 3)

    # Generative model classification
    #print Xtrain
    Wgme = gmEstim(Xtrain, Ttrain, 3)
    Yestim_gme = Wgme*np.transpose(Xtest)
    for i in range(0, Yestim_gme.shape[1]):
        Yestim_gme[:,i] = softMax(Yestim_gme[:,i])
    Yestim_gme = extractMax(np.transpose(Yestim_gme), 3)
    gme_err = evalAccuracy(Yestim_gme, Ttest, 3)
    print lsq_err, gme_err


X, T = assignVar(len(data[:,1]))
evaluate(X,X,T,T)



'''
plt.figure(1)
plt.subplot(231)
plt.plot(x1[0:49], x2[0:49], 'bo')
plt.plot(x1[50:99], x2[50:99], 'rx')
plt.plot(x1[100:149], x2[100:149], 'g*')
plt.subplot(232)
plt.plot(x1[0:49], x3[0:49], 'bo')
plt.plot(x1[50:99], x3[50:99], 'rx')
plt.plot(x1[100:149], x3[100:149], 'g*')
plt.subplot(233)
plt.plot(x1[0:49], x4[0:49], 'bo')
plt.plot(x1[50:99], x4[50:99], 'rx')
plt.plot(x1[100:149], x4[100:149], 'g*')

plt.subplot(234)
plt.plot(x2[0:49], x3[0:49], 'bo')
plt.plot(x2[50:99], x3[50:99], 'rx')
plt.plot(x2[100:149], x3[100:149], 'g*')
plt.subplot(235)
plt.plot(x2[0:49], x4[0:49], 'bo')
plt.plot(x2[50:99], x4[50:99], 'rx')
plt.plot(x2[100:149], x4[100:149], 'g*')
plt.subplot(236)
plt.plot(x3[0:49], x4[0:49], 'bo')
plt.plot(x3[50:99], x4[50:99], 'rx')
plt.plot(x3[100:149], x4[100:149], 'g*')
plt.show()
'''
