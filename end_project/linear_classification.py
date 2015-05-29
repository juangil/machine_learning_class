import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from pylab import *
from matplotlib import cm
import scipy


file = open('all_data_sorted.data', 'r')
table = [row.strip().split(' ') for row in file]
data = np.matrix(table)
label = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
label_biclass_1 = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 1}
#print data

# Some utility functions------------------------------------------------------

    # Function soft max
def softMax(z):
    nume = np.exp(z)
    denom = np.sum(nume)
    ret = nume / denom
    return ret

    # Logistic function
def logisFunc(a):
    ret = 1.0/(1.0 + np.exp(a))
    return ret

    # Function to read the input data
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

    # Mean and standard deviation evaluation
def meanDesvCacl(z):
    mmean = np.sum(z)/len(z)
    desv = np.sum((z - mmean)*(z - mmean))/len(z)
    return mmean, desv

    # Extracting the class wich probability is the highest
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

    # Evaluationg accuracy of a preditcion
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


# Several learners------------------------------------------------------------

    # Least squares
def msEstim(X,T):
    return (np.linalg.inv(np.transpose(X)*X)) * (np.transpose(X)*T)

    # Generative models
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


    # Logistic regression with Newton Raphson optimization
def logRegEstim(X,T, Xtest):
    PHI = X
    N = X.shape[0]
    M = X.shape[1]
    #Wold = np.transpose(np.matrix(np.random.random_sample(M)))
    Wold = np.transpose(np.matrix(np.zeros(M))) + 0.001
    Wnew = np.transpose(np.matrix(np.zeros(M)))
    EPS = 1e-6
    diff = np.linalg.norm(Wold - Wnew)
    cont = 0
    Yestim = T
    #print T
    while cont < 10:
        yy = np.transpose(Wold) * np.transpose(PHI)
        yy = np.transpose(logisFunc(yy))
        yy = 1.0 - yy
        Yestim = yy
        #print yy
        #print ' '
        R = np.zeros((N,N))
        for i in range(0,N):
            #print yy[i,0]
            R[i,i] = (yy[i,0] * (1 - yy[i,0])) + 0.001

        z = (PHI*Wold) - (np.linalg.inv(R)*(yy - T))
        Wnew = np.linalg.inv((np.transpose(PHI) * (R * PHI))) * (np.transpose(PHI)*R*z)
        diff = np.linalg.norm(Wold - Wnew)
        if diff < EPS:
            break
        else:
            Wold = Wnew
        cont += 1

    #print Xtest.shape
    ret = np.transpose(Wnew) * np.transpose(Xtest)
    ret = np.transpose(logisFunc(ret))
    ret = 1.0 - ret
    #print ret.shape
    return ret

# Cross validation -----------------------------------------------------------
def evaluate(Xtrain, Xtest, Ttrain, Ttest, nclass):
    # Least squares estimation
    Wmse = msEstim(Xtrain,Ttrain)
    Yestim = np.transpose(Wmse)*np.transpose(Xtest)
    Yestim = np.transpose(Yestim)
    Yestim = extractMax(Yestim, nclass)
    lsq_err = evalAccuracy(Yestim, Ttest, nclass)

    # Generative model classification
    #print Xtrain
    Wgme = gmEstim(Xtrain, Ttrain, nclass)
    Yestim_gme = Wgme*np.transpose(Xtest)
    for i in range(0, Yestim_gme.shape[1]):
        Yestim_gme[:,i] = softMax(Yestim_gme[:,i])
    Yestim_gme = extractMax(np.transpose(Yestim_gme), nclass)
    gme_err = evalAccuracy(Yestim_gme, Ttest, nclass)

    # Discriminative model classification(logistic regression)
    Yestim_log_reg = np.zeros((len(Ttest[:,0]), nclass))
    Yestim_log_reg = np.matrix(Yestim_log_reg)
    for i in range(0, nclass):
        Yestim_log_reg[:,i] = logRegEstim(Xtrain, Ttrain[:,i], Xtest)
    Yestim_log_reg = extractMax(Yestim_log_reg,nclass)
    #print Yestim_log_reg, Ttest
    log_reg_err = evalAccuracy(Yestim_log_reg, Ttest, nclass)

    return lsq_err, gme_err, log_reg_err

def crossValidation(X, T, nclass, valid_num = 5):
    N = len(X[:, 0])
    Nvalid = (N/valid_num)
    Ntest = N - Nvalid
    index = np.random.permutation(N)
    l = 0
    r = Nvalid
    err_lse_arr = np.zeros(valid_num)
    err_gme_arr = np.zeros(valid_num)
    err_log_arr = np.zeros(valid_num)
    for i in range(0, valid_num):

        Xtrain = []
        Ttrain = []
        if l == 0:
            Xtrain = X[index[r:N], :]
            Ttrain = T[index[r:N], :]
        else:
            if(r < N):
                Xtrain = np.concatenate((X[index[0:l], :],X[index[r: N], :]))
                Ttrain = np.concatenate((X[index[0:l], :],X[index[r: N], :]))
            else:
                Xtrain = X[index[0:l], :]
                Ttrain = T[index[0:l], :]


        Xvalid = X[index[l:r], :]
        Tvalid = T[index[l:r], :]

        l = r
        r += Nvalid
        '''
        print Xvalid.shape
        print Tvalid.shape
        print Xtrain.shape
        print Ttrain.shape
        print ' '
        '''
        err_lse_arr[i], err_gme_arr[i], err_log_arr[i] = evaluate(Xtrain, Xvalid, Ttrain, Tvalid, nclass)

    print 'Accuracy least squares classification: '
    #print err_lse_arr
    acc, desv = meanDesvCacl(err_lse_arr)
    print 'Accuracy: ', acc, ' Standard Deviation', desv, ' Maximum accuracy: ', np.amax(err_lse_arr)
    print '---------------------------------------------------------------'

    print 'Accuracy generative models classification: '
    #print err_gme_arr
    acc, desv = meanDesvCacl(err_gme_arr)
    print 'Accuracy: ', acc, ' Standard Deviation', desv, ' Maximum accuracy: ', np.amax(err_gme_arr)
    print '---------------------------------------------------------------'

    print 'Accuracy logistic regression classification: '
    #print err_log_arr
    acc, desv = meanDesvCacl(err_log_arr)
    print 'Accuracy: ', acc, ' Standard Deviation', desv, ' Maximum accuracy: ', np.amax(err_log_arr)
    print '---------------------------------------------------------------'



X, T = assignVar(len(data[:,1]))
#print X
#evaluate(X,X,T,T, 3)
#crossValidation(X,T,3)

x11 = np.transpose(X[0:104,1])
x12 = np.transpose(X[0:104,2])
x13 = np.transpose(X[0:104,3])

x21 = X[104:190,1]
x22 = X[104:190,2]
x23 = X[104:190,3]

x31 = X[190:606,1]
x32 = X[190:606,2]
x33 = X[190:606,3]

x41 = X[606:841,1]
x42 = X[606:841,2]
x43 = X[606:841,3]

print x11

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x11, x12, x13, c='r', marker='o')
#ax.scatter(x21, x22, x23, c='b', marker='*')
#ax.scatter(x31, x32, x33, c='r', marker='x')
#ax.scatter(x41, x42, x43, c='g', marker='+')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

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
