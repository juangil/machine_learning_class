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


def assignVar(N):
    a = np.zeros((N,1))
    b = np.zeros((N,1))
    c = np.zeros((N,1))
    d = np.zeros((N,1))
    for i in range(0, N):
        tmp = data[:,0]
        a[i,0] = float(tmp[i,0])
        tmp = data[:,1]
        b[i,0] = float(tmp[i,0])
        tmp = data[:,2]
        c[i,0] = float(tmp[i,0])
        tmp = data[:,3]
        d[i,0] = float(tmp[i,0])
    return np.matrix(a),np.matrix(b),np.matrix(c),np.matrix(d)

x1, x2, x3, x4 = assignVar(len(data[:,1]))


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
