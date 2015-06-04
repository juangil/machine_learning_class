import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from matplotlib import cm
import scipy

def softMax(z):
    nume = np.exp(z)
    denom = np.sum(nume)
    ret = nume / denom
    return ret

class nn:
    def __init__(self, d, M, K, b = 'sig', p = 'clas'):
        self.dim = d + 1
        self.neurons = M
        self.outs = K
        self.w1 = np.transpose(np.matrix(np.random.standard_normal((d+1)*M)))
        self.w2 = np.transpose(np.matrix(np.random.standard_normal(M*K)))
        self.gradw1 = np.transpose(np.matrix(np.zeros((d+1)*M)))
        self.gradw2 = np.transpose(np.matrix(np.zeros(M*K)))
        self.W = np.concatenate((self.w1, self.w2), axis = 0)
        self.bas_fun = b
        self.ai = np.zeros((M, 1))
        self.zi = np.zeros((M, 1))
        self.wmat1 = np.matrix(np.zeros((M, d + 1)))
        self.wmat2 = np.matrix(np.zeros((K, M)))
        self.tp = p

    def weight1ToMat(self):
        l = 0
        r = self.neurons
        for i in range(self.dim):
            self.wmat1[:,i] = self.w1[l:r, 0]
            l = r
            r = r + self.neurons

    def weight2ToMat(self):
        l = 0
        r = self.outs
        for i in range(self.neurons):
            self.wmat2[:,i] = self.w2[l:r, 0]
            l = r
            r = r + self.outs

    def activ1Eval(self, xx):
        ret = self.wmat1 * xx
        return ret

    def activ2Eval(self, xx):
        ret = self.wmat2 * xx
        return ret

    def basFunEval(self, A):
        ret = 0.0
        if(self.bas_fun == 'sig'):
            ret = 1.0 / (1.0 + np.exp(-A))
        elif(self.bas_fun == 'tanh'):
            ret = (np.exp(A) - np.exp(-A))/(np.exp(A) + np.exp(-A))
        return ret

    def computeDelta2(self, Y, T):
        return Y - T

    def computeDelta1(self, Y, T):
        tmp = T - Y
        tmp = np.transpose(tmp)*self.wmat2
        h_der = 0
        if(self.bas_fun == 'sig'):
            tmp = 1.0 / (1.0 + np.exp(-self.ai))
            tmp2 = 1.0 - tmp
            h_der = np.multiply(tmp, tmp2)
        elif(self.bas_fun == 'tanh'):
            h_der = (np.exp(self.ai) - np.exp(-self.ai))/(np.exp(self.ai) + np.exp(-self.ai))
            h_der = 1.0 - np.square(h_der)
        return np.multiply(h_der,tmp)

    def forwardPropagation(self, xx):
        self.weight1ToMat()
        self.weight2ToMat()
        A = self.activ1Eval(xx)
        self.ai = A
        Z = self.basFunEval(A)
        self.zi = Z
        Y = self.activ2Eval(Z)
        if(self.tp == 'clas' and self.outs == 1):
            Y = 1.0 / (1.0 + np.exp(-Y))
        return Y

    def errorGradient(self, xx, Y, T):
        delta_2 = self.computeDelta2(Y,T)
        derivative2 = delta_2*np.transpose(self.zi)
        delta_1 = self.computeDelta1(Y,T)
        derivative1 = delta_1*np.transpose(xx)
        l = 0
        r = self.neurons
        for i in range(0, self.dim):
            self.gradw1[l:r,0] = derivative1[:,i]
            l = r
            r = r + self.neurons
        l = 0
        r = self.outs
        for i in range(0, self.neurons):
            self.gradw2[l:r,0] = derivative2[:,i]
            l = r
            r = r + self.outs

        return np.concatenate((self.gradw1, self.gradw2), axis = 0)

    def process(self, xx, T):
        cont = 0
        while(cont < 50):
            Y = self.forwardPropagation(xx)
            gradient_w = self.errorGradient(xx, Y, T)
            w_new = self.W - (0.5*gradient_w)
            self.W = w_new
            self.w1 = w_new[0:self.dim*self.neurons, 0]
            self.w2 = w_new[self.dim*self.neurons:w_new.shape[0], 0]
            print Y
            cont = cont + 1

    def debug(self):
        print self.w2
        print self.wmat2

'''
X = np.random.standard_normal(size = (2,6))*10.
T = np.random.binomial(1, 0.5, 6)
tmp = np.ones(X.shape[1])
mneural = nn(X.shape[0], 5, 1)
XX = np.zeros((X.shape[0] + 1, X.shape[1]))
XX[1:,:] = X
XX[0,:] = tmp
XX = np.matrix(XX)
print XX, T
mneural.process(XX, T
'''

mdata = scipy.io.loadmat('ejemplo_class_uno.mat')
X = np.array(mdata['X'])
t = np.array(mdata['t'])
for i in range(0, t.shape[0]):
    if(t[i,0] == -1):
        t[i,0] = 0
t = np.matrix(t)
t = np.transpose(t)
tmp = np.ones(X.shape[0])
XX = np.zeros((X.shape[0], X.shape[1] + 1))
XX[:,1:] = X
XX[:,0] = tmp
XX = np.matrix(XX)
XX = np.transpose(XX)
mneural = nn(XX.shape[0] - 1, 2, 1)
#print XX, t
mneural.process(XX, t)



'''
out = np.zeros(1000)
for i in range(0,len(out)):
    out[i] = Y[0,i]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(X[0,:], X[1,:], out, cmap=cm.jet, linewidth=0.2, vmin = 0.0, vmax = 1.0)
plt.show()


#mneural.debug()
#mneural.weight2ToMat()
#print ' '
#mneural.debug()
'''
