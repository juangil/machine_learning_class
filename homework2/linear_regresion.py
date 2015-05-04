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

def basis_functions_eval(xx, MM, op = 'poly'):
	PHI = np.zeros((len(xx),MM))
	#print PHI
	PHI[:,0] = np.ones(len(xx))
	mu = np.linspace(0.0, 1.0, MM - 1)
	aux = np.diff(mu)
	s2 = 0.2*aux[0]
	xx.shape = (len(xx))
	#print xx.shape, PHI[:,1].shape
	for i in range(1, MM):
		if(op == 'poly'):
			PHI[:, i] = (xx**i)
		elif(op == 'exp'):
			PHI[:,i] = exp(-((xx - mu[i])**2.0)/(2.0*s2))
		elif(op == 'tanh'):
			PHI[:, i] = 1.0/(1.0 + exp(-(xx - mu[i])/(math.sqrt(s2))))
	return PHI

# Maximum Likelihood ----------------------------------------------------
def maximum_likelihood_estim(PHI,tt):
	PHI = np.matrix(PHI)
	tt = np.matrix(tt)
	PHIt = np.transpose(PHI)
	tmp_inv = PHIt*PHI
	tmp_inv = np.linalg.inv(tmp_inv)
	Wml = tmp_inv*PHIt*tt
	return Wml

# Regularization -------------------------------------------------------
def regularization_estim(PHI, tt, lambda_param = 0.5):
	PHI = np.matrix(PHI)
	tt = np.matrix(tt)
	PHIt = np.transpose(PHI)
	tmp_inv = (lambda_param*np.eye(PHI.shape[1])) + PHIt*PHI
	tmp_inv = np.linalg.inv(tmp_inv)
	Wreg = tmp_inv*PHIt*tt
	return Wreg;

# Bayes estimation -----------------------------------------------------
def bayes_lin_estim(PHI, tt, alpha, beta):
	PHI = np.matrix(PHI)
	PHIt = np.transpose(PHI)
	tt = np.matrix(tt)
	Ia = np.eye(PHI.shape[1])
	Sn_inv = (alpha*Ia) + (beta*PHIt*PHI)
	Sn = np.linalg.inv(Sn_inv)
	mn = beta*Sn*PHIt*tt
	return mn, Sn

def maxim_likelihood_2(PHI, tt, niters = 20):
	alpha = 0.003
	beta = 1.0
	PHI = np.matrix(PHI)
	PHIt = np.transpose(PHI)
	tt = np.matrix(tt)
	rho,v = np.linalg.eig(PHIt*PHI)
	lambda_par = beta*rho

	alpha_arr = np.ones((niters + 1, 1))
	beta_arr = np.ones((niters + 1, 1))
	beta_arr[0] = beta
	alpha_arr[0] = alpha

	for i in range(1, niters + 1):
		A = alpha*np.eye(PHI.shape[1]) +  beta*(PHIt*PHI)
		Ainv = np.linalg.inv(A)*np.eye(PHI.shape[1])
		mn = beta*Ainv*PHIt*tt
		mnt = np.transpose(mn)
		gamma = np.sum(lambda_par/(lambda_par + alpha))
		alpha = float((gamma/(mnt*mn)))
		aux = (tt - PHI*mn);
		betaInv = (float(np.transpose(aux)*aux))/(PHI.shape[0] - gamma)
		beta = 1.0/betaInv
		lambda_par = beta*rho
		beta_arr[i] = beta
		alpha_arr[i] = alpha

	print alpha, beta
	return alpha,beta


def debug_validation(Ntest, basis_num = 10):
	index = np.random.permutation(N)
	xtest = x[index[0:Ntest]]
	ttest = t[index[0:Ntest]]
	xvalid = x[index[Ntest + 1:N]]
	tvalid = t[index[Ntest + 1:N]]
	PHItest = basis_functions_eval(xtest, basis_num)
	PHI = basis_functions_eval(x, basis_num)

	# Maximum likelihood
	Wml = maximum_likelihood_estim(PHItest, ttest)
	tpredicted_ml = PHI*Wml
	plt.plot(x, tpredicted_ml, 'y')

	# Regularization
	'''
	Wreg = regularization_estim(PHItest, ttest, lambda_param = 0.000000001)
	print Wreg
	tpredicted_reg = PHI*Wreg
	print tpredicted_reg
	#plt.plot(x, tpredicted_reg, 'g')
	'''

	# Bayesian linear regresion
	'''
	PHItest = basis_functions_eval(xtest, basis_num)
	alpha, beta = maxim_likelihood_2(PHItest, ttest)
	mn, Sn = bayes_lin_estim(PHItest, ttest, alpha, beta)
	tpredicted_bay = PHI*mn
	plt.plot(x, tpredicted_bay, 'g')
	'''

debug_validation(50, 20)
plt.show()
