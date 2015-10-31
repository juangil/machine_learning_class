import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from matplotlib import cm
import scipy

# Comment this line if you don't want numpy to take a fixed seed to random utilities
np.random.seed(2)

# Function to compute mean and standar deviation
def meanDesvCacl(z):
    mmean = np.sum(z)/len(z)
    desv = np.sum((z - mmean)*(z - mmean))/len(z)
    return mmean, desv

# Gaussian normalization
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

# min max normalization
def normalize_arr_min_max(x):
    b = x[0,np.argmax(x)]
    a = x[0,np.argmin(x)]
    x = x - a
    x = np.divide(x, b - a)
    return x

# Softmax
def softMax(z):
    nume = np.exp(z)
    denom = np.sum(nume)
    ret = nume*(1.0/denom)
    return ret

# Function for extracting the class that an instance belongs to
def extractMax(Yestim, nclass):
    if(nclass == 1):
        for i in range(0, len(Yestim[:,0])):
            if Yestim[i,0] > 0.1:
                Yestim[i,0] = 1.
            else:
                Yestim[i,0] = 0.
        return Yestim
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

def extractMaxDummy(Yestim, nclass):
    for i in range(len(Yestim[:,0])):
        for j in range(nclass):
            if(Yestim[i,j] < 0.5):
                Yestim[i,j] = 0.
            else:
                Yestim[i,j] = 1.

    return Yestim

# Accuracy evaluation for a prediction
def evalAccuracy(Yestim, Tvalid, nclass):
    #print Yestim.shape, Tvalid.shape
    true_positives = np.zeros(nclass)
    true_negatives = np.zeros(nclass)
    false_positives = np.zeros(nclass)
    false_negatives = np.zeros(nclass)
    for i in range(0,len(Yestim[:,0])):
        for j in range(0,nclass):
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

# Neural network class ---------------------------------------------------------
'''
 this neural network has all the elements neccesary to build a two layer neural network, briefly has the next methods:
 - weight1ToMat: takes the weight vector and covnerts it into a matrix
 - activ1eval: Ealuates the activations of the first hidden layer
 - basFunEval: Evaluates the activations funcions after the first hidden layer activation
 - cost_fun_eval: Evaluates the cost function for each specifica problem
 - computeDelta2: Evaluates the first part of derivatives of the cost function respect to second hidden layer weights
 - computeDelta1 Evaluates the first part of derivatives of the cost function respect to first hidden layer weights
 - forwardPropagation: Computes a prediction with a given set of weights and an input vector
 - errorGradient: Computes the gradient of the cost function
 - process: Performs the gradient descent algorithm
 '''
class nn:
    def __init__(self, d, M, K, lr, b = 'sig', p = 'clas'):
        self.dim = d + 1 # input
        self.neurons = M
        self.outs = K
        self.w1 = np.transpose(np.matrix(np.random.standard_normal((d+1)*M)))
        self.w2 = np.transpose(np.matrix(np.random.standard_normal(M*K)))
        self.iniw1 = self.w1
        self.iniw2 = self.w2
        self.gradw1 = np.transpose(np.matrix(np.zeros((d+1)*M)))
        self.gradw2 = np.transpose(np.matrix(np.zeros(M*K)))
        self.W = np.concatenate((self.w1, self.w2), axis = 0)
        #self.W_ini = np.concatenate((self.w1, self.w2), axis = 0)
        self.bas_fun = b
        self.ai = np.zeros((M, 1))
        self.zi = np.zeros((M, 1))
        self.a2i = np.zeros((K, 1))
        self.wmat1 = np.matrix(np.zeros((M, d + 1)))
        self.wmat2 = np.matrix(np.zeros((K, M)))
        self.tp = p
        self.lr = lr

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
        A = A + 0.001
        if(self.bas_fun == 'sig'):
            ret = 1.0 / (1.0 + np.exp(-A))
        elif(self.bas_fun == 'tanh'):
            ret = np.divide((np.exp(A) - np.exp(-A)),(np.exp(A) + np.exp(-A)))
        elif(self.bas_fun == 'exp'):
            ret = np.exp(-A)
        else:
            ret = A
        ret = ret + 0.001
        return ret

    def cost_fun_eval(self, xx, T, Y):
        if(self.tp == 'clas'):
            #Y = self.forwardPropagation(xx)
            tmp1 = np.multiply(np.log(Y), T)
            tmp2 = np.multiply(np.log((1.0 - Y) + 0.001), (1.0 - T))
            ret = tmp1 + tmp2
            if(self.outs > 1):
                ret = np.sum(ret, axis = 1)
            ret = np.sum(ret)
            return -ret
        elif(self.tp == 'mult_clas'):
            #for i in range(0, Y.shape[1]):
                #Y[:,i] = softMax(Y[:,i])
            ret = np.transpose(T)*Y
            ret = np.diagonal(ret)
            ret = np.sum(ret)
            #Y = np.transpose(Y)
            #print np.sum(Y, axis = 0)
            return -ret


    def computeDelta2(self, Y, T):
        #if(self.tp == 'mult_clas'):
        #    return (Y - T)*(1.0/Y.shape[1])
        return Y - T

    def computeDelta1(self, Y, T):
        tmp = 0.0
        if(self.tp == 'clas'):
            tmp = Y - T
            tmp = np.transpose(tmp)*self.wmat2
        elif(self.tp == 'mult_clas' and self.outs > 1):
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
        #print 'jeje', h_der.shape, tmp.shape
        return np.multiply(h_der,np.transpose(tmp))

    def forwardPropagation(self, xx):
        self.weight1ToMat()
        self.weight2ToMat()
        A = self.activ1Eval(xx) + 0.001
        self.ai = A
        Z = self.basFunEval(A) + 0.001
        self.zi = Z
        Y = self.activ2Eval(Z) + 0.001
        self.a2i = Y + 0.001
        #print Y
        if(self.tp == 'clas'):
            Y = 1.0 / (1.0 + np.exp(-Y))
        elif(self.tp == 'mult_clas'):
            for i in range(0, Y.shape[1]):
                Y[:,i] = softMax(Y[:,i])
            #print np.sum(Y, axis = 0)
        #print Y
        #Y = Y + 0.001
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
            print "Nooo!!!, the dimensions between real and computed outputs are not equal"
            return
        cont = 0
        Y = 0
        mmenor = 100000000.0
        super_Y = 0.0
        super_W = 0.0
        super_w1 = 0.0
        super_w2 = 0.0
        while(cont < 1000):
            Y = self.forwardPropagation(xx)
            tm_cost = self.cost_fun_eval(xx, T, Y)
            #print tm_cost
            if(tm_cost < mmenor):
                mmenor = tm_cost
                super_Y = Y
                super_W = self.W
                super_w1 = self.w1
                super_w2 = self.w2
            #print np.sum(Y[:,0])
            gradient_w = self.errorGradient(xx, Y, T)
            w_new = self.W - (self.lr*gradient_w)
            diff = np.sqrt(np.sum(np.square(w_new)))
            self.W = w_new
            self.w1 = w_new[0:self.dim*self.neurons, 0]
            self.w2 = w_new[self.dim*self.neurons:w_new.shape[0], 0]
            cont = cont + 1

        #print super_W
        self.w1 = super_w1
        self.w2 = super_w2
        return np.transpose(super_Y)

    def debug(self):
        print self.w2
        print self.wmat2


def crossValidation(X, T, nclass, neu_net, valid_num = 5):
    N = len(X[:, 0])
    Nvalid = (N/valid_num)
    Ntest = N - Nvalid
    index = np.random.permutation(N)
    #print Ntest
    l = 0
    r = Nvalid
    test_acc = np.zeros(valid_num)
    for i in range(0, valid_num):

        Xtrain = []
        Ttrain = []
        if l == 0:
            Xtrain = X[index[r:N], :]
            Ttrain = T[index[r:N], :]
        else:
            if(r < N):
                Xtrain = np.concatenate((X[index[0:l], :],X[index[r: N], :]))
                Ttrain = np.concatenate((T[index[0:l], :],T[index[r: N], :]))
            else:
                Xtrain = X[index[0:l], :]
                Ttrain = T[index[0:l], :]


        Xvalid = X[index[l:r], :]
        Tvalid = T[index[l:r], :]

        l = r
        r += Nvalid

        neu_net.w1 = neu_net.iniw1
        neu_net.w2 = neu_net.iniw2
        Xtrain = np.transpose(Xtrain)
        Ttrain = np.transpose(Ttrain)
        Xvalid = np.transpose(Xvalid)
        Tvalid = np.transpose(Tvalid)
        neu_net.process(Xtrain, Ttrain)
        Yestim = neu_net.forwardPropagation(Xvalid)
        #print neu_net.wmat1
        #print neu_net.wmat2
        Yestim = np.transpose(Yestim)
        Yestim = extractMax(Yestim, Yestim.shape[1])
        #print Yestim
        #print ' '
        #print neu_net.super_W
        tmp_acum = evalAccuracy(Yestim, np.transpose(Tvalid), Yestim.shape[1])
        #print tmp_acum
        #print ' '
        test_acc[i] = tmp_acum


    acc, desv = meanDesvCacl(test_acc)
    print 'Accuracy: ', acc, ' Standard Deviation', desv, ' Maximum accuracy: ', np.amax(test_acc)
    print '---------------------------------------------------------------'




# DATA SET: VEHICLES------------------------------------------------------
# Data reading
file = open('all_data_sorted.data', 'r')
table = [row.strip().split(' ') for row in file]
data = np.matrix(table)

def assignVar(N):
    a = np.zeros((N,1))
    b = np.zeros((N,1))
    c = np.zeros((N,1))
    #d = np.zeros((N,1))
    e = np.zeros((N,4))
    for i in range(0, N):
        tmp = data[:,0]
        a[i,0] = float(tmp[i,0])
        tmp = data[:,1]
        b[i,0] = float(tmp[i,0])
        tmp = data[:,2]
        c[i,0] = float(tmp[i,0])
        #tmp = data[:,3]
        #d[i,0] = float(tmp[i,0])
        tmp = data[:,3]
        e[i, tmp[i,0]] = 1.
    unos=np.ones((N,1))
    XX = np.concatenate((unos,a,b,c),1)
    return np.matrix(XX), np.matrix(e)

X, T = assignVar(len(data[:,1]))


# Parameter assignation, if you want to assign a parameter value, change this part
param_m = 7          # Amount of activations
param_lr = 0.01      # Learning rate
param_bs = 'sig'     # Type of basis function or activation function
prob2solve = 'clas'  # Type of problem you want to solve

# Object neural network instance, dont touch any of this coo
print 'Accuracy for multiple ', prob2solve, ': '
mneural = nn(X.shape[1] - 1, param_m, 4, param_lr, param_bs, prob2solve)
crossValidation(X,T,mneural.outs, mneural, 5)

'''
X = np.transpose(X)

x11 = X[0:104,1]
x12 = X[0:104,2]
x13 = X[0:104,3]

x21 = X[104:190,1]
x22 = X[104:190,2]
x23 = X[104:190,3]

x31 = X[190:606,1]
x32 = X[190:606,2]
x33 = X[190:606,3]

x41 = X[606:841,1]
x42 = X[606:841,2]
x43 = X[606:841,3]
print x11.shape

plt.figure(1)
y_dot, = plt.plot(x11, x12, 'yo')
r_dot, = plt.plot(x21, x22, 'ro')
g_dot, = plt.plot(x31, x32, 'go')
b_dot, = plt.plot(x41, x42, 'bo')
plt.ylabel('largo')
plt.xlabel('ancho')
plt.legend([r_dot, y_dot, g_dot, b_dot], ["Taxi", "Vehiculos grandes", "Particulares", "Motocicletas"])
plt.show()
'''
