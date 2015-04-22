import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

mean = np.matrix('0;0')
mean_array = np.array((0.0,0.0))
covariance = np.matrix('0.5 0.25; 0.25 1.0')
pi = np.pi
x = np.random.multivariate_normal(mean_array,covariance,1000)
x3 = np.zeros(len(x))

x_unif = np.random.random_sample((1000, 2))
L = np.linalg.cholesky(covariance)
x_chol = np.zeros((len(x_unif),2))
x_chol_out = np.zeros(len(x_chol))
for i in range (0,len(x_unif)):
    tmp = mean_array + np.transpose(L*np.transpose(np.matrix(x[i])))
    x_chol[i] = tmp

#print x
def gaussian_eval(x):
    cov_det = np.linalg.det(covariance)
    tmp = (1.0 / (2.0 * pi))  * (1.0 / math.sqrt(cov_det))
    #print x, np.array(mean_array)
    tmp2 = np.matrix(x - np.array(mean_array))
    tmp3 = (-1.0 / 2.0) * (tmp2*np.linalg.inv(covariance)*np.transpose(tmp2))
    ret = tmp*math.exp(tmp3)
    return ret

def generate_data():
    for i in range(0,len(x_chol)):
        #tmp = np.array([x1[i], x2[i]])
        x_chol_out[i] = gaussian_eval(x_chol[i])

def mean_estim():
    msum = np.array((0.0,0.0));
    msumx = 0.0
    for i in range(0,len(x)):
        #print msum, np.array((x1[i], x2[i]))
        msum += x_chol[i]#np.array((x1[i], x2[i]))
        #msumx += x1[i]
    #print msum,msumx/len(x1)
    return msum/len(x_chol)

'''
def cov_estim(mean_estim):
    msum = np.matrix('0.0 0.0; 0.0 0.0')
    for i in range(0,len(x)):
        tmp_x = np.transpose(np.matrix((x1[i],x2[i])))
        tmp_mean = np.transpose(np.matrix(mean_estim))
        tmp1 = (tmp_x - tmp_mean)
        tmp_mat = tmp1 * np.transpose(tmp1)
        msum = msum + np.transpose(tmp1)
    return msum/len(x)
'''

generate_data()
#print x[0:,0], x[0:,1], x3
mean_hat = mean_estim()
#cov_hat = cov_estim(mean_hat)

print mean_hat, #cov_hat
#print mean_array, covariance

'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x[0:,0], x[0:,1], x3, cmap=cm.jet, linewidth=0.2, vmin = -4.0, vmax = 4.0)
plt.show()
'''
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(x_chol[0:,0], x_chol[0:,1], x_chol_out, cmap=cm.jet, linewidth=0.2, vmin = -4.0, vmax = 4.0)
plt.show()
