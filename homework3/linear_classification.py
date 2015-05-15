import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pylab import *
from matplotlib import cm


file = open('iris.data', 'r')
table = [row.strip().split(',') for row in file]
data = np.matrix(table)
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}


def msEstim(X,T):
    return (np.linalg.inv(np.transpose(X)*X)) * (np.transpose(X)*T)

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

def extractMax(Yestim):
    for i in range(len(Yestim[:,0])):
        mayor = -1000000.0
        idmayor = -1
        for j in range(3):
            if(Yestim[i,j] > mayor):
                mayor = Yestim[i,j]
                idmayor = j

        for j in range(3):
            if (j == idmayor):
                Yestim[i,j] = 1.
            else:
                Yestim[i,j] = 0.
    return Yestim

def evaluate(Xtrain, Xtest, Ttrain, Ttest):
    # Least squares estimation
    Wmse = msEstim(Xtrain,Ttrain)
    Yestim = np.transpose(Wmse)*np.transpose(Xtest)
    Yestim = np.transpose(Yestim)
    #print Yestim
    Yestim = extractMax(Yestim)
    print Yestim


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
