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


# Some utility functions------------------------------------------------------

    # Function soft max
def softMax(z):
    nume = np.exp(z)
    denom = np.sum(nume)
    ret = nume / denom
    return ret

    # Logistic function
def logisFunc(a):
    ret = 1.0/(1.0 + np.exp(-a))
    return ret

    # Function to read the input data
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

    # Evaluating accuracy of a preditcion
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

    ret = np.sum(true_positives + true_negatives)/np.sum(true_positives + false_positives + false_negatives + true_negatives)
    ret_class1 = (true_positives[0] + true_negatives[0])/(true_positives[0] + false_positives[0] + false_negatives[0] + true_negatives[0])
    return ret, ret_class1


# Several learners------------------------------------------------------------

    # Least squares
def msEstim(X,T):
    return (np.linalg.inv(np.transpose(X)*X)) * (np.transpose(X)*T)

    # Generative models
def gmEstim(X,T, nclass):
    msize = X[0].shape[1]
    # Calculate the mean vectors
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

    # Calculate the covariance matrix.
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
    # Calculate the priors over each class
    Pi_ml = np.zeros((1,nclass))
    for i in range(0, nclass):
        Ni = 0
        for j in range(0, len(X[:,0])):
            Ni = Ni + T[j,i]
        Pi_ml[0,i] = Ni / len(X[:,0])


    Sigma = np.matrix(Sigma[1:, 1:])
    #print np.linalg.inv(Sigma)*np.transpose(mio)
    Wgm = np.zeros((nclass, msize))
    # For each class we have to find a predictive distribution
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
    # Newton raphson with several iteration
    while cont < 10:
        yy = np.transpose(Wold) * np.transpose(PHI)
        yy = np.transpose(logisFunc(yy))
        yy = yy
        Yestim = yy
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
    #print ret.shape
    return ret

# Cross validation -----------------------------------------------------------
def evaluate(Xtrain, Xtest, Ttrain, Ttest, nclass):
    # Least squares estimation

    Wmse = msEstim(Xtrain,Ttrain)
    Yestim = np.transpose(Wmse)*np.transpose(Xtest)
    Yestim = np.transpose(Yestim)
    Yestim = extractMax(Yestim, nclass)
    lsq_err, lsq_err1 = evalAccuracy(Yestim, Ttest, nclass)

    # Generative model classification

    Wgme = gmEstim(Xtrain, Ttrain, nclass)
    Yestim_gme = Wgme*np.transpose(Xtest)
    for i in range(0, Yestim_gme.shape[1]):
        Yestim_gme[:,i] = softMax(Yestim_gme[:,i])
    Yestim_gme = extractMax(np.transpose(Yestim_gme), nclass)
    gme_err, gme_err1 = evalAccuracy(Yestim_gme, Ttest, nclass)

    # Discriminative model classification(logistic regression)

    Yestim_log_reg = np.zeros((len(Ttest[:,0]), nclass))
    Yestim_log_reg = np.matrix(Yestim_log_reg)
    for i in range(0, nclass):
        Yestim_log_reg[:,i] = logRegEstim(Xtrain, Ttrain[:,i], Xtest)
    Yestim_log_reg = extractMax(Yestim_log_reg,nclass)
    log_reg_err, log_reg_err1 = evalAccuracy(Yestim_log_reg, Ttest, nclass)

    # setting the class 1 accuracy
    class_1_err = np.array([lsq_err1, gme_err1, log_reg_err1])

    return lsq_err, gme_err, log_reg_err, class_1_err


def crossValidation(X, T, nclass, valid_num = 5):
    N = len(X[:, 0])
    Nvalid = (N/valid_num)
    Ntest = N - Nvalid
    # Setting a seed for random elections, comment this line if you want a new permutation in each execution of this subroutine
    np.random.seed(3)
    # Generating a random permutation of the indexes of the array
    index = np.random.permutation(N)
    l = 0
    r = Nvalid
    # Arrays to save the values of the accuracy obtained in each  validation
    err_lse_arr = np.zeros(valid_num)
    err_gme_arr = np.zeros(valid_num)
    err_log_arr = np.zeros(valid_num)
    err_class_1 = np.zeros(nclass)
    for i in range(0, valid_num):
        # Dividing the training groups of the validation group for each validation
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
        '''
        print Xvalid.shape
        print Tvalid.shape
        print Xtrain.shape
        print Ttrain.shape
        print ' '
        '''
        # calculating the accuracy for the ith validation for each method
        err_lse_arr[i], err_gme_arr[i], err_log_arr[i], tmp = evaluate(Xtrain, Xvalid, Ttrain, Tvalid, nclass)
        err_class_1 = err_class_1 + tmp

    print 'Accuracy least squares classification: '
    #print err_lse_arr
    acc, desv = meanDesvCacl(err_lse_arr)
    print 'Accuracy: ', ("%.3f" % acc), ' Standard Deviation', ("%.3f" % desv), ' Maximum accuracy: ', ("%.3f" % np.amax(err_lse_arr))
    print '---------------------------------------------------------------'

    print 'Accuracy generative models classification: '
    #print err_gme_arr
    acc, desv = meanDesvCacl(err_gme_arr)
    print 'Accuracy: ', ("%.3f" % acc), ' Standard Deviation', ("%.3f" % desv), ' Maximum accuracy: ', ("%.3f" % np.amax(err_gme_arr))
    print '---------------------------------------------------------------'

    print 'Accuracy logistic regression classification: '
    #print err_log_arr
    acc, desv = meanDesvCacl(err_log_arr)
    print 'Accuracy: ', ("%.3f" % acc), ' Standard Deviation', ("%.3f" % desv), ' Maximum accuracy: ', ("%.3f" % np.amax(err_log_arr))
    print '---------------------------------------------------------------'

    print 'Accuracy for class 1 prediction: '
    print ' Least squares: ', ("%.3f" % (err_class_1[0]/valid_num))
    print ' Generative models: ', ("%.3f" % (err_class_1[1]/valid_num))
    print ' Logistic regression: ', ("%.3f" % (err_class_1[2]/valid_num))



X, T = assignVar(len(data[:,1]))
#evaluate(X,X,T,T, 3)
crossValidation(X,T,3)

x1 = X[:,1]
x2 = X[:,2]
x3 = X[:,3]
x4 = X[:,4]


plt.figure(1)
plt.subplot(231)
plt.xlabel('attribute 1')
plt.ylabel('attribute 2')
plt.plot(x1[0:49], x2[0:49], 'bo')
plt.plot(x1[50:99], x2[50:99], 'rx')
plt.plot(x1[100:149], x2[100:149], 'g*')
plt.subplot(232)
plt.xlabel('attribute 1')
plt.ylabel('attribute 3')
plt.plot(x1[0:49], x3[0:49], 'bo')
plt.plot(x1[50:99], x3[50:99], 'rx')
plt.plot(x1[100:149], x3[100:149], 'g*')
plt.subplot(233)
plt.xlabel('attribute 1')
plt.ylabel('attribute 4')
plt.plot(x1[0:49], x4[0:49], 'bo')
plt.plot(x1[50:99], x4[50:99], 'rx')
plt.plot(x1[100:149], x4[100:149], 'g*')

plt.subplot(234)
plt.xlabel('attribute 2')
plt.ylabel('attribute 3')
plt.plot(x2[0:49], x3[0:49], 'bo')
plt.plot(x2[50:99], x3[50:99], 'rx')
plt.plot(x2[100:149], x3[100:149], 'g*')
plt.subplot(235)
plt.xlabel('attribute 2')
plt.ylabel('attribute 4')
plt.plot(x2[0:49], x4[0:49], 'bo')
plt.plot(x2[50:99], x4[50:99], 'rx')
plt.plot(x2[100:149], x4[100:149], 'g*')
plt.subplot(236)
plt.xlabel('attribute 3')
plt.ylabel('attribute 4')
plt.plot(x3[0:49], x4[0:49], 'bo')
plt.plot(x3[50:99], x4[50:99], 'rx')
plt.plot(x3[100:149], x4[100:149], 'g*')
plt.show()
