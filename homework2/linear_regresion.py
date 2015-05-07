import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math

mdata = scipy.io.loadmat('ejemplo_regresion.mat')
x = np.array(mdata['x'])
y = np.array(mdata['y'])
t = np.array(mdata['t'])
N = len(x)
mdiff = (t - y)*(t - y)
mvar = (np.sum(mdiff))/N
#print mvar, math.sqrt(mvar)


# Evaluation of design matrix -----------------------------------------------------
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
			PHI[:,i] = np.exp(-((xx - mu[i - 1])**2.0)/(2.0*s2))
		elif(op == 'tanh'):
			PHI[:, i] = 1.0/(1.0 + np.exp(-(xx - mu[i - 1])/(math.sqrt(s2))))
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
	Sn = np.linalg.inv(Sn_inv)*np.eye(PHI.shape[1])
	mn = beta*(Sn*PHIt*tt)
	#print mn
	#print Sn
	return mn, Sn

def maxim_likelihood_2(PHI, tt, niters = 20):
	alpha = 0.0000000000000003
	beta = 0.09079
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

	#alpha = 0.000000000003
	#beta = 0.09079
	return alpha,beta

# Square error computing -----------------------------------------------------

def squareErrorEval(real_arr, pred_arr):
	print len(real_arr), len(pred_arr)
	diff = np.array(real_arr) - np.array(pred_arr)
	#print real_arr
	diff = diff ** 2
	msum = np.sum(diff)
	error = (msum/len(real_arr)) * 100
	return error

# Debugging and cross validation -----------------------------------------------------
def debug_validation(Ntest, basis_num = 10):
	index = np.random.permutation(N)
	xtest = x[index[0:Ntest]]
	ttest = t[index[0:Ntest]]
	xvalid = x[index[Ntest + 1:N]]
	tvalid = y[index[Ntest + 1:N]]
	PHItest = basis_functions_eval(xtest, basis_num)
	PHI = basis_functions_eval(x, basis_num)
	PHI_valid = basis_functions_eval(xvalid, basis_num)

	# Maximum likelihood
	#'''
	Wml = maximum_likelihood_estim(PHItest, ttest)
	tpredicted_ml = PHI*Wml
	tpredicted_ml_valid = PHI_valid*Wml
	plt.figure(1)
	plt.plot(x, y, 'b', x, t, 'bo')
	plt.plot(x, tpredicted_ml, 'r')
	#'''

	# Regularization

	PHItest = basis_functions_eval(xtest, basis_num)
	Wreg = regularization_estim(PHItest, ttest, lambda_param = 0.000000000003)
	tpredicted_reg = PHI*Wreg
	tpredicted_reg_valid = PHI_valid*Wreg
	plt.figure(2)
	plt.plot(x, y, 'b', x, t, 'bo')
	plt.plot(x, tpredicted_reg, 'r')


	# Bayesian linear regresion
	PHItest = basis_functions_eval(xtest, basis_num)
	alpha, beta = maxim_likelihood_2(PHItest, ttest)
	mn, Sn = bayes_lin_estim(PHItest, ttest, alpha, beta)
	tpredicted_bay = PHI*mn
	tpredicted_bay_valid = PHI_valid*mn
	plt.figure(3)
	plt.plot(x, y, 'b', x, t, 'bo')
	plt.plot(x, tpredicted_bay, 'r')
	print 'Error max likelihood: ', squareErrorEval(tvalid, tpredicted_ml_valid)
	print 'Error Regularization: ', squareErrorEval(tvalid, tpredicted_reg_valid)
	print 'Error Bayesian regression: ', squareErrorEval(tvalid, tpredicted_bay_valid)

def crossValidation()

debug_validation(50, 20)
plt.show()
