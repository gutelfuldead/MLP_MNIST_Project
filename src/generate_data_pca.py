import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import pickle
import math

def pca(X):
	'''
	ref : https://github.com/bytefish/facerecognition_guide/blob/master/src/py/tinyfacerec/subspace.py
	'''
	[n,d] = X.shape
	print n,d
	mu = X.mean(axis=0)
	X = X - mu
	if n>d:
		C = np.dot(X.T,X)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
	else: # this one always runs...
		C = np.dot(X,X.T)
		[eigenvalues,eigenvectors] = np.linalg.eigh(C)
		eigenvectors = np.dot(X.T,eigenvectors)
		for i in xrange(n):
			eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])
	idx = np.argsort(-eigenvalues)
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:,idx]
	return [eigenvalues, eigenvectors, mu]

def norm_vectors(eigenvectors):
	'''
	ref : https://github.com/Sebac/Eigenfaces_FaceRecognition/blob/master/face_recognition.py
	'''
	eigenvectors = eigenvectors.transpose()
	res = []
	for i in range(eigenvectors.shape[0]):
		suma = math.sqrt(sum([x**2 for x in eigenvectors[i]]))
		res.append([x/suma for x in eigenvectors[i]])
		# print res[i]
	return np.array(res).transpose()


def calc_eigface_var(e,var):
	'''
	calculate how many of the eigenvectors are needed in order to account for VAR
	of all the possible variation
	input:
		e : np array
			array of eigenvalues
		var : float
			0 < var < 1 the amount of variation to account for
	returns:

	'''
	eigsum = sum(e)
	csum = 0.0
	tv = 0.0
	for i in range(0,len(e)):
		csum += e[i]
		tv = csum/eigsum
		if tv > var:
			# print("{} prinicpal components required to account for {}% of the total variance").format(i,var*100)
			return i

# Fetch Data and split into training and test sets
mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
X_train, X_validate, X_test = X[:60000], X[60000:70000], X[70000:]
y_train, y_validate, y_test = y[:60000], y[60000:70000], y[70000:]

# Generate the eigen-numbers
[e_trn,EV_trn,mu] = pca(X_train)
EV_trn = norm_vectors(EV_trn)
best_pca = calc_eigface_var(e_trn,.99)
print ("To keep 99% of the variance need {} principal components").format(best_pca)
EV_trn = EV_trn[:,0:best_pca]
print EV_trn.shape, y_train.shape

# number of iterations to run
sz = 10

# data format...
data_trn = np.empty((sz,4)) # best convergence training
data_vld = np.empty((sz,4)) # best convergence testing
data_err = np.empty((sz,4)) # best convergence error rate
error_one_hl_nomom = []
error_one_hl_mom = []
error_two_hl_nomom = []
error_two_hl_mom = []

# loop one hidden layer from 1 --> 100 PEs
for i in range(1,sz):
    print("num PEs={}\n").format(i*10)

    mlp1 = MLPClassifier(hidden_layer_sizes=(i*10), activation='relu', momentum=0,max_iter=100, alpha=1e-4,
                        solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.1)

    # print("\nmlp2 no momentum; num PEs={}").format(i+1)
    mlp2 = MLPClassifier(hidden_layer_sizes=(i*10,i*10), activation='relu', momentum=0,max_iter=100, alpha=1e-4,
                        solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.1)

    # print("\nmlp1 momentum; num PEs={}").format(i+1)
    mlp3 = MLPClassifier(hidden_layer_sizes=(i*10), activation='relu', max_iter=100, alpha=1e-4,
    solver='sgd', verbose=False, tol=1e-4, random_state=1,momentum=0.9,nesterovs_momentum=False, learning_rate_init=.1)

    # print("\nmlp2 momentum; num PEs={}").format(i+1)
    mlp4 = MLPClassifier(hidden_layer_sizes=(i*10,i*10), activation='relu', max_iter=100, alpha=1e-4,
    solver='sgd', verbose=False, tol=1e-4, random_state=1,momentum=0.9,nesterovs_momentum=False,learning_rate_init=.1)

    # Train with eigen data
    mlp1.fit(EV_trn, y_train)
    mlp2.fit(EV_trn, y_train)
    mlp3.fit(EV_trn, y_train)
    mlp4.fit(EV_trn, y_train)

    # capture the convergence error over epochs
    data_err[i,0] = min(mlp1.loss_curve_)
    data_err[i,1] = min(mlp2.loss_curve_)
    data_err[i,2] = min(mlp3.loss_curve_)
    data_err[i,3] = min(mlp4.loss_curve_)

    # capture the training error with the original training data
    data_trn[i,0] = mlp1.score(EV_trn, y_train)
    data_trn[i,1] = mlp2.score(EV_trn, y_train)
    data_trn[i,2] = mlp3.score(EV_trn, y_train)
    data_trn[i,3] = mlp4.score(EV_trn, y_train)

    # capture the validation error
    data_vld[i,0] = mlp1.score(X_validate, y_validate)
    data_vld[i,1] = mlp2.score(X_validate, y_validate)
    data_vld[i,2] = mlp3.score(X_validate, y_validate)
    data_vld[i,3] = mlp4.score(X_validate, y_validate)

    error_one_hl_nomom.append(mlp1.loss_curve_)
    error_one_hl_mom.append(mlp2.loss_curve_)
    error_two_hl_nomom.append(mlp3.loss_curve_)
    error_two_hl_mom.append(mlp4.loss_curve_)

def pickleme(name,data):
    output = open('../pkls/'+name+'.pkl','wb')
    pickle.dump(data,output)
    output.close()

pickleme('data_err_egn',data_err)
pickleme('data_trn_egn',data_trn)
pickleme('data_vld_egn',data_vld)
pickleme('error_one_hl_nomom_egn',error_one_hl_nomom)
pickleme('error_one_hl_mom_egn',error_one_hl_mom)
pickleme('error_two_hl_nomom_egn',error_two_hl_nomom)
pickleme('error_two_hl_mom_egn',error_two_hl_mom)
