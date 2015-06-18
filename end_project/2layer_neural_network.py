import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from matplotlib import cm
import scipy

def normalize_arr_gauss(x):
    msum = np.sum(x)
    N = x.shape[1]
    mmean = msum/N
    diff = np.square(x - mmean)
    desv = math.sqrt(np.sum(diff)/N)
    #print mmean, desv
    x = x - mmean
    x = np.divide(x, desv)
    return x

def softMax(z):
    nume = np.exp(z)
    denom = np.sum(nume)
    ret = nume / denom
    return ret

# Neural network class ---------------------------------------------------------
class nn:
    def __init__(self, d, M, K, b = 'sig', p = 'clas'):
        self.dim = d + 1 # input
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
            ret = np.divide((np.exp(A) - np.exp(-A)),(np.exp(A) + np.exp(-A)))
        return ret

    def cost_fun_eval(self, xx, T):
        if(self.tp == 'clas'):
            if(self.outs == 1):
                Y = self.forwardPropagation(xx)
                tmp1 = np.multiply(np.log(Y), T)
                tmp2 = np.multiply(np.log(1.0 - Y), (1.0 - T))
                ret = tmp1 + tmp2
                ret = np.sum(ret)
                return -ret

    def computeDelta2(self, Y, T):
        return Y - T

    def computeDelta1(self, Y, T):
        tmp = 0
        if(self.tp == 'clas' and self.outs == 1):
            tmp = Y - T
            tmp = np.transpose(tmp)*self.wmat2
        elif(self.tp == 'reg'):
            tmp = T - Y
            tmp = np.transpose(tmp)*self.wmat2

        h_der = 0
        if(self.bas_fun == 'sig'):
            tmp1 = 1.0 / (1.0 + np.exp(-self.ai))
            tmp2 = 1.0 - tmp1
            h_der = np.multiply(tmp1, tmp2)
        elif(self.bas_fun == 'tanh'):
            h_der = np.divide((np.exp(self.ai) - np.exp(-self.ai)),(np.exp(self.ai) + np.exp(-self.ai)))
            h_der = 1.0 - np.square(h_der)
        #print 'delta1'
        #print h_der.shape, tmp.shape
        return np.multiply(h_der,np.transpose(tmp))

    def forwardPropagation(self, xx):
        self.weight1ToMat()
        self.weight2ToMat()
        A = self.activ1Eval(xx)
        self.ai = A
        Z = self.basFunEval(A)
        self.zi = Z
        Y = self.activ2Eval(Z)
        #print Y
        if(self.tp == 'clas'):
            Y = 1.0 / (1.0 + np.exp(-Y))
        elif(self.tp == 'mult_clas'):
            for i in range(0, Y.shape[1]):
                Y[:,i] = softMax(Y[:,i])
        #print Y
        return Y

    def errorGradient(self, xx, Y, T):
        delta_2 = self.computeDelta2(Y,T)
        derivative2 = delta_2*np.transpose(self.zi)
        delta_1 = self.computeDelta1(Y,T)
        #print delta_1.shape, xx.shape
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
        #print np.concatenate((self.gradw1, self.gradw2), axis = 0)
        return np.concatenate((self.gradw1, self.gradw2), axis = 0)

    def process(self, xx, T):
        try:
            if(self.outs == T.shape[0]):
                1
            else:
                raise Exception
        except Exception as inst:
            print "Holy Shit!!!, the dimensions between real and computed outputs are not equal"
            return
        cont = 0
        Y = 0
        while(cont < 15):
            #print self.cost_fun_eval(xx, T)
            Y = self.forwardPropagation(xx)
            gradient_w = self.errorGradient(xx, Y, T)
            w_new = self.W - (0.1*gradient_w)
            diff = np.sqrt(np.sum(np.square(w_new)))
            self.W = w_new
            self.w1 = w_new[0:self.dim*self.neurons, 0]
            self.w2 = w_new[self.dim*self.neurons:w_new.shape[0], 0]
            cont = cont + 1

        print np.transpose(Y)

    def debug(self):
        print self.w2
        print self.wmat2

'''
mdata = scipy.io.loadmat('ejemplo_class_uno.mat')
X = np.array(mdata['X'])
t = np.array(mdata['t'])
for i in range(0, t.shape[0]):
    if(t[i,0] == -1):
        t[i,0] = 0.0
    else:
        t[i,0] = 1.0

t = np.matrix(t)
t = np.transpose(t)
tmp = np.ones(X.shape[0])
XX = np.zeros((X.shape[0], X.shape[1] + 1))
XX[:,1:] = X
XX[:,0] = tmp
XX = np.matrix(XX)
XX = np.transpose(XX)
print XX, t
#print np.amax(XX[1,:]), np.amin(XX[1,:])
#print np.amax(XX[2,:]), np.amin(XX[2,:])
#print ' '
#print np.transpose(XX), np.transpose(t)
mneural = nn(XX.shape[0] - 1, 2, 1, 'tanh')
#print ' '
mneural.process(XX, t)
#print XX, t
XX = np.transpose(XX)
t = np.transpose(t)

plt.figure(1)
plt.plot(XX[0:49,1], XX[0:49,2], 'bo')
plt.plot(XX[50:100,1], XX[50:100,2], 'rx')
plt.show()
'''


file = open('iris.data', 'r')
table = [row.strip().split(',') for row in file]
data = np.matrix(table)
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
label_biclass_1 = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 1}

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

XX, T = assignVar(len(data[:,1]))
XX = np.transpose(XX)
T = np.transpose(T)
T = T[0,:]
X = XX[0:3,:]
X[2,:] = XX[3,:]
# Normalization
X[1,:] = normalize_arr_gauss(X[1,:])
X[2,:] = normalize_arr_gauss(X[2,:])
#print X,T
#print np.amax(X[1,:]), np.amin(X[1,:])
#print np.amax(X[2,:]), np.amin(X[2,:])

mneural = nn(X.shape[0] - 1, 2, 1, 'tanh')
mneural.process(X, T)
'''
X = np.transpose(X)
T = np.transpose(T)
plt.figure(2)
plt.plot(X[0:49,1], X[0:49,2], 'bo')
plt.plot(X[50:150,1], X[50:150,2], 'rx')
plt.show()
'''
