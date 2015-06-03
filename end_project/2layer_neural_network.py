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
        #print tmp[:,0], np.sum(tmp[:,0])
        #print tmp
        #print self.wmat2
        #print np.transpose(tmp[:,0])*self.wmat2
        tmp = np.transpose(tmp)*self.wmat2
        h_der = 0
        if(self.bas_fun == 'sig'):
            tmp = 1.0 / (1.0 + np.exp(-self.ai))
            tmp2 = 1.0 - tmp
            h_der = np.multiply(tmp, tmp2)
        elif(self.bas_fun == 'tanh'):
            h_der = (np.exp(self.ai) - np.exp(-self.ai))/(np.exp(self.ai) + np.exp(-self.ai))
            h_der = 1 - np.square(h_der)
        #print h_der.shape, tmp.shape
        return np.multiply(h_der,tmp)

    def forwardPropagation(self, xx):
        #print self.w1
        #print self.w2
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
        #print derivative1
        #print derivative2
        #print self.w1
        #print self.w2
        l = 0
        r = self.neurons
        for i in range(0, self.dim):
            self.w1[l:r,0] = derivative1[:,i]
            l = r
            r = r + self.neurons
        l = 0
        r = self.outs
        for i in range(0, self.neurons):
            self.w2[l:r,0] = derivative2[:,i]
            l = r
            r = r + self.outs
        #print self.w1
        #print self.w2

        '''
        derivative1 = delta_1*np.transpose(xx)
        print derivative1
        print self.wmat1
        print ' '
        print derivative2
        print self.wmat2
        '''


    def process(self, xx, T):
        Y = self.forwardPropagation(xx)
        self.errorGradient(xx, Y, T)
        return Y

    def debug(self):
        print self.w2
        print self.wmat2


X = np.random.standard_normal(size = (2,6))*10.
T = np.random.binomial(1, 0.5, 6)
tmp = np.ones(X.shape[1])
mneural = nn(X.shape[0], 5, 4)
XX = np.zeros((X.shape[0] + 1, X.shape[1]))
XX[1:,:] = X
XX[0,:] = tmp
XX = np.matrix(XX)
mneural.process(XX, T)

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
