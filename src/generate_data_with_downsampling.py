# test over huge iteration set
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from scipy.signal import decimate
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from mlp_nmist_project_functions import pickleme, depickle

PE_rng = 10 # will range number of PEs used form 10 -> 10*PE_rng in intervals of 10
decimation_rng = 6 # will range total decimation runs from 1 -> decimation_rng
data_trn = np.empty((PE_rng,4)) # best convergence training
data_vld = np.empty((PE_rng,4)) # best convergence testing

# initial retrieval of data
mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
y_train = y[:60000]
X_train = X[:60000]

# In loaded MNIST data first 60,000 used for training equally distributed 0-9
# 60,000 - 70,000 used for testing/validating equally distirbuted 0-9
# need to randomly reindex 60,000-70,000 and split in half for validation and testing data
x_shuffle,y_shuffle = shuffle(X[60000:], y[60000:], random_state=0)
X_validate, X_test = x_shuffle[:5000], x_shuffle[5000:]
y_validate, y_test = y_shuffle[:5000], y_shuffle[5000:]
pickleme('X_validate',X_validate)
pickleme('X_test',X_test)
pickleme('y_validate',y_validate)
pickleme('y_test',y_test)

plt.figure()
plt.subplot(2,1,1)
plt.hist(y_validate)
plt.title('Histogram of validation values')
plt.ylabel('Frequency')
plt.subplot(212)
plt.hist(y_test)
plt.title('Histogram of test values')
plt.ylabel('Frequency')
plt.savefig('../imgs/histogram-validation-test-values.png')
plt.close()

# Iterate over all decimations interested in
for z in range(1,decimation_rng):

    # Fetch Data and split into training,validation, and test sets
    X_train = X[:60000]
    X_validate = depickle('X_validate')

    # Decimate by a factor of n
    dec = z
    if dec > 0:
        X_train = decimate(X_train, dec, axis=1)
        X_validate = decimate(X_validate, dec, axis=1)
    print("Decimated by a factor of {}").format(dec)
    # number of iterations to run

    # data format...
    error_one_hl_nomom = []
    error_one_hl_mom = []
    error_two_hl_nomom = []
    error_two_hl_mom = []

    # loop one hidden layer from 1 --> 100 PEs
    start = time.time()
    for i in range(1,PE_rng):
        mlp0 = MLPClassifier(hidden_layer_sizes=(i*10), activation='relu', momentum=0,max_iter=1000, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.1)

        mlp1 = MLPClassifier(hidden_layer_sizes=(i*10), activation='relu', max_iter=1000, alpha=1e-4,
            solver='sgd', verbose=False, tol=1e-4, random_state=1,momentum=0.9,nesterovs_momentum=False, learning_rate_init=.1)

        mlp2 = MLPClassifier(hidden_layer_sizes=(i*10,i*10), activation='relu', momentum=0,max_iter=1000, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.1)

        mlp3 = MLPClassifier(hidden_layer_sizes=(i*10,i*10), activation='relu', max_iter=1000, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1,momentum=0.9,nesterovs_momentum=False,learning_rate_init=.1)

        # Train Data
        print("downsamp - {}: Generating data for one hidden layer, with no momentum with {} PEs per layer").format(dec, i*10)
        mlp0.fit(X_train, y_train)
        print("downsamp - {}: Generating data for one hidden layer, with momentum with {} PEs per layer").format(dec, i*10)
        mlp1.fit(X_train, y_train)
        print("downsamp - {}: Generating data for two hidden layers, with no momentum with {} PEs per layer").format(dec, i*10)
        mlp2.fit(X_train, y_train)
        print("downsamp - {}: Generating data for two hidden layers, with momentum with {} PEs per layer\n").format(dec, i*10)
        mlp3.fit(X_train, y_train)

        # capture the training error
        data_trn[i,0] = mlp0.score(X_train, y_train)
        data_trn[i,1] = mlp1.score(X_train, y_train)
        data_trn[i,2] = mlp2.score(X_train, y_train)
        data_trn[i,3] = mlp3.score(X_train, y_train)

        # capture the test error
        data_vld[i,0] = mlp0.score(X_validate, y_validate)
        data_vld[i,1] = mlp1.score(X_validate, y_validate)
        data_vld[i,2] = mlp2.score(X_validate, y_validate)
        data_vld[i,3] = mlp3.score(X_validate, y_validate)

        error_one_hl_nomom.append(mlp0.loss_curve_)
        error_one_hl_mom.append(mlp1.loss_curve_)
        error_two_hl_nomom.append(mlp2.loss_curve_)
        error_two_hl_mom.append(mlp3.loss_curve_)

    end = time.time()
    ttl_time = end-start

    print("Took {} s for decimation of {}").format(ttl_time,dec)

    pickleme('data_trn_dsamp_'+str(dec),data_trn)
    pickleme('data_vld_dsamp_'+str(dec),data_vld)
    pickleme('error_one_hl_nomom_dsamp_'+str(dec),error_one_hl_nomom)
    pickleme('error_one_hl_mom_dsamp_'+str(dec),error_one_hl_mom)
    pickleme('error_two_hl_nomom_dsamp_'+str(dec),error_two_hl_nomom)
    pickleme('error_two_hl_mom_dsamp_'+str(dec),error_two_hl_mom)
    pickleme('ttl_time_dsamp_'+str(dec),ttl_time)
