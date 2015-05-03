import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math

mdata = scipy.io.loadmat('ejemplo_regresion.mat')
x = np.array(mdata['x'])
y = np.array(mdata['y'])
t = np.array(mdata['t'])
N = len(x)

plt.figure(1)
plt.plot(x, y, 'r', x, t, 'bo')
plt.show()

def basis_functions_eval(xx, MM, op = 'poly'):
	PHI = np.zeros((len(xx),MM))
	PHI[:,0] = np.ones(len(xx))
	mu = np.linspace(0.0, 1.0, MM - 1)
	aux = np.diff(mu)
	s2 = 0.2*aux[0]
	for i in range(1, MM):
		if(op == 'poly'):
			phi[:, i] = (xx**i)
		else if(op == 'exp'):
			phi[:,i] = exp(-((xx - mu[i])**2.0)/(2.0*s2))
		else if(op == 'tanh'):
			phi[:, i] = 1.0/(1.0 + exp(-(xx - mu[i])/(math.sqrt(s2))))
	return PHI


def maximum_likelihood_estim(PHI,t):
	PHI = np.matrix(PHI)
	tt = np.matrix(t)
	PHIt = np.transpose(PHI)
	tmp_inv = PHIt*PHI
	tmp_inv = np.linalg.inv(tmp_inv)
	Wml = tmp_inv*(PHIt*np.transpose(tt))
	return Wml

def debug_validation(Ntest):	
	index = np.random.permutation(N)
	xtest = x[index[0:Ntest + 1]]
	ttest = t[index[0:Ntest + 1]]
	xvalid = x[index[Ntest + 2:N]]
	tvalid = t[index[Ntest + 2:N]]