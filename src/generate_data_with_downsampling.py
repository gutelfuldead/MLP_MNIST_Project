#!/usr/bin/python2.7
'''
USAGE:
    ./generate_data_with_downsampling.py minPE maxPE decimation_rng
    minPE and maxPE MUST be a multiple of 10
    Will generate data for #PEs from minPE --> maxPE in multiples of 10
    For each of these sets will repeat data for decimation levels 1 -> decimation_rng

example :
    ./generate_data_with_downsampling.py 10 100 3
    will generate ANNs ranging from 10 -> 100 PEs for decimations 1,2, and 3

Will pickle data for usage in find_optimal_parameters.py
'''

import numpy as np
import pickle
import time
import sys
import matplotlib.pyplot as plt
from scipy.signal import decimate
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from mlp_nmist_project_functions import pickleme, depickle

minPE = int(sys.argv[1])/10
maxPE = int(sys.argv[2])/10+1
PE_pkl_rng = [minPE,maxPE]
pickleme('PE_pkl_rng',PE_pkl_rng)
decimation_rng = int(sys.argv[3])+1
pickleme('decimation_rng',decimation_rng)
data_trn = np.empty((maxPE-minPE,4)) # best convergence training
data_vld = np.empty((maxPE-minPE,4)) # best convergence testing

# initial retrieval of data
mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
y_train = y[:60000]
X_train = X[:60000]

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p], p

# In loaded MNIST data first 60,000 used for training equally distributed 0-9
# 60,000 - 70,000 used for testing/validating equally distirbuted 0-9
# need to randomly reindex 60,000-70,000 and split in half for validation and testing data
x_shuffle,y_shuffle, p = unison_shuffled_copies(X[60000:], y[60000:])
X_validate, X_test = x_shuffle[:5000], x_shuffle[5000:]
y_validate, y_test = y_shuffle[:5000], y_shuffle[5000:]
pickleme('random_index',p)

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
    X_validate = x_shuffle[:5000]

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
    for i in range(minPE,maxPE):
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
        data_trn[i-minPE,0] = mlp0.score(X_train, y_train)
        data_trn[i-minPE,1] = mlp1.score(X_train, y_train)
        data_trn[i-minPE,2] = mlp2.score(X_train, y_train)
        data_trn[i-minPE,3] = mlp3.score(X_train, y_train)

        # capture the test error
        data_vld[i-minPE,0] = mlp0.score(X_validate, y_validate)
        data_vld[i-minPE,1] = mlp1.score(X_validate, y_validate)
        data_vld[i-minPE,2] = mlp2.score(X_validate, y_validate)
        data_vld[i-minPE,3] = mlp3.score(X_validate, y_validate)

        error_one_hl_nomom.append(mlp0.loss_curve_)
        error_one_hl_mom.append(mlp1.loss_curve_)
        error_two_hl_nomom.append(mlp2.loss_curve_)
        error_two_hl_mom.append(mlp3.loss_curve_)

    end = time.time()
    ttl_time = end-start

    print("Took {} s for decimation of {}").format(ttl_time,dec)

    pickleme('data_trn_dsamp_'+str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs',data_trn)
    pickleme('data_vld_dsamp_'+str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs',data_vld)
    pickleme('error_one_hl_nomom_dsamp_'+str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs',error_one_hl_nomom)
    pickleme('error_one_hl_mom_dsamp_'+str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs',error_one_hl_mom)
    pickleme('error_two_hl_nomom_dsamp_'+str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs',error_two_hl_nomom)
    pickleme('error_two_hl_mom_dsamp_'+str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs',error_two_hl_mom)
    pickleme('ttl_time_dsamp_'+str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs',ttl_time)
