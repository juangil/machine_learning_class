import numpy as np
import pylab as pb
import GPy
import urllib


def SetInputTMOGP(X, input_t, n_signals=5):
    sup_input = input_t[:,0]
    sup_input_id = np.zeros(input_t.shape[0])
    sup_output = X[0,:]
    tmp_output = sup_output
    tmp_input = sup_input

    for i in range(1,n_signals):
        tmp_input = np.append(tmp_input, sup_input)
        tmp_input_id = np.linspace(i,i,X.shape[1])
        sup_input_id = np.append(sup_input_id, tmp_input_id)
        tmp_output = np.append(tmp_output, X[i,:])

    sup_input = tmp_input
    #sup_input = sup_input*(1/100.0)
    sup_output = tmp_output
    sup_output = sup_output[:,None]
    all_X = np.vstack((sup_input, sup_input_id))
    all_X = np.transpose(all_X)
    return all_X, sup_output

def ComputeMSQERR(a,b):
    mean_sq_err = a - b
    mean_sq_err = np.power(mean_sq_err, 2)
    val_mean_sq_err = np.sum(mean_sq_err) / mean_sq_err.shape[0]
    return val_mean_sq_err

def GPTest(X, Y, n_signals=5):
    #print Y, X
    # Defining the covariance and the coregionalization matrix, first one can try with an RBF
    print n_signals
    kern = GPy.kern.RBF(1., lengthscale=50.)**GPy.kern.Coregionalize(input_dim = 1,output_dim=X.shape[0], rank=n_signals-1)
    model = GPy.models.GPRegression(X, Y, kern)
    return model


url = ("http://mocap.cs.cmu.edu/subjects/10/10_06.amc") # Tell to python where is the file
urllib.urlretrieve(url, '10_06.amc') # Tell python to retrieve the file

amc_file = open('10_06.amc', 'r')
cont = 0
read_frames = False
print 'Reading header...'
frame = 1
bone = 0
bone_map_dof = {}
bone_name = {}
all_samples = {}
samples_frame = 0
for mline in amc_file:
	if(mline == ":DEGREES\r\n" and read_frames == False):
	    read_frames = True
	    print 'now reading frames'
	    continue
	elif(read_frames):
	    params = mline.split(' ')
	    if params[0] == str(frame) + '\r\n':
	    	#print 'reading frame: ' + str(frame)
	    	if(frame >= 2):
	    		#print samples_frame
	    		all_samples[frame - 1] = samples_frame

	    	samples_frame = 0
	        frame += 1
	        bone = 1
	    else:
	    	if(bone == 1):
	    		samples_frame = np.array(params[1:len(params)], dtype=float)
	    		#print samples_frame
	    		bone_map_dof[bone] = len(params) - 1
	    		bone_name[bone] = params[0]
	    	else:
	        	bone_sample = np.array(params[1:len(params)], dtype=float)
	        	samples_frame = np.append(samples_frame, bone_sample)
	        	bone_map_dof[bone] = len(params) - 1
	        	bone_name[bone] = params[0]
	        bone += 1
all_samples[frame - 1] = samples_frame
#print all_samples[1], bone_map, bone_name
X = np.zeros((all_samples[1].shape[0], len(all_samples)))
for i in range(0,len(all_samples)):
	X[:,i] = all_samples[i + 1]
print 'File succesfully loaded'

time_stamps = np.arange(len(all_samples))
'''pb.plot(time_stamps, X[40,:])


print X.shape[0]
for i in range(0,X.shape[0]):
    # extract the event
    x_event = time_stamps
    y_event = X[i,:]
    pb.plot(x_event, y_event)
'''

#Making predictions over all data set
'''
n_outputs = 5
print X.shape, input_t.shape
X_train, Y_train = SetInputTMOGP(X, input_t, n_outputs)
print X_train, Y_train
model = GPTest(X_train, Y_train, n_outputs)

for i in range(n_outputs):
    model.plot(fignum=1,fixed_inputs=[(1, i)], plot_raw=True)

for i in range(0,n_outputs):
    # extract the event
    x_event = time_stamps
    y_event = X[i,:]
    pb.plot(x_event, y_event)

mu, var = model._raw_predict(X_train)
print ComputeMSQERR(mu,Y_train)
'''

# Making predictions using some test dataset and some training dataset
n_frames = time_stamps.shape[0]
n_outputs = 25
index = np.random.permutation(n_frames)
Ntest = 100

X_train = index[Ntest:X.shape[1]]
Y_train = X[:, X_train]
X_train = X_train[:, None]

X_test = index[0:Ntest]
Y_test = X[:, X_test]
X_test = X_test[:, None]

print X_train.shape, Y_train.shape
X_train, Y_train = SetInputTMOGP(Y_train, X_train, n_outputs)
X_test, Y_test = SetInputTMOGP(Y_test, X_test, n_outputs)

model = GPTest(X_train, Y_train, n_outputs)
'''
for i in range(n_outputs):
    model.plot(fignum=1,fixed_inputs=[(1, i)],plot_raw=True)
for i in range(0,n_outputs):
    # extract the event
    x_event = time_stamps
    y_event = X[i,:]
    pb.plot(x_event, y_event)'''

mu, var = model._raw_predict(X_test)
print ComputeMSQERR(mu,Y_test)
