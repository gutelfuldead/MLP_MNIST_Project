# test over huge iteration set
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
import pickle
from scipy.signal import decimate
import time
from mlp_nmist_project_functions import pickleme

mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
sz = 10
decimation_rng = 6
data_trn = np.empty((sz,4)) # best convergence training
data_vld = np.empty((sz,4)) # best convergence testing
y_train, y_validate, y_test = y[:50000], y[50000:60000], y[60000:]

# Iterate over all decimations interested in
for z in range(1,decimation_rng):

    # Fetch Data and split into training,validation, and test sets
    X_train, X_validate, X_test = X[:50000], X[50000:60000], X[60000:]

    # Decimate by a factor of n
    dec = z
    if dec > 0:
        X_train = decimate(X_train, dec, axis=1)
        X_validate = decimate(X_validate,dec,axis=1)
    print("Decimated by a factor of {}").format(dec)
    # number of iterations to run

    # data format...
    error_one_hl_nomom = []
    error_one_hl_mom = []
    error_two_hl_nomom = []
    error_two_hl_mom = []

    # loop one hidden layer from 1 --> 100 PEs
    start = time.time()
    for i in range(1,sz):
        mlp1 = MLPClassifier(hidden_layer_sizes=(i*10), activation='relu', momentum=0,max_iter=1000, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.1)

        # print("\nmlp2 no momentum; num PEs={}").format(i+1)
        mlp2 = MLPClassifier(hidden_layer_sizes=(i*10,i*10), activation='relu', momentum=0,max_iter=1000, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.1)

        # print("\nmlp1 momentum; num PEs={}").format(i+1)
        mlp3 = MLPClassifier(hidden_layer_sizes=(i*10), activation='relu', max_iter=1000, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1,momentum=0.9,nesterovs_momentum=False, learning_rate_init=.1)

        # print("\nmlp2 momentum; num PEs={}").format(i+1)
        mlp4 = MLPClassifier(hidden_layer_sizes=(i*10,i*10), activation='relu', max_iter=1000, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1,momentum=0.9,nesterovs_momentum=False,learning_rate_init=.1)

        # Train Data
        print("downsamp - {}: Generating data for one hidden layer, with no momentum with {} PEs per layer").format(dec, i*10)
        mlp1.fit(X_train, y_train)
        print("downsamp - {}: Generating data for one hidden layer, with momentum with {} PEs per layer").format(dec, i*10)
        mlp2.fit(X_train, y_train)
        print("downsamp - {}: Generating data for two hidden layers, with no momentum with {} PEs per layer").format(dec, i*10)
        mlp3.fit(X_train, y_train)
        print("downsamp - {}: Generating data for two hidden layers, with momentum with {} PEs per layer\n").format(dec, i*10)
        mlp4.fit(X_train, y_train)

        # capture the training error
        data_trn[i,0] = mlp1.score(X_train, y_train)
        data_trn[i,1] = mlp2.score(X_train, y_train)
        data_trn[i,2] = mlp3.score(X_train, y_train)
        data_trn[i,3] = mlp4.score(X_train, y_train)

        # capture the test error
        data_vld[i,0] = mlp1.score(X_validate, y_validate)
        data_vld[i,1] = mlp2.score(X_validate, y_validate)
        data_vld[i,2] = mlp3.score(X_validate, y_validate)
        data_vld[i,3] = mlp4.score(X_validate, y_validate)

        error_one_hl_nomom.append(mlp1.loss_curve_)
        error_one_hl_mom.append(mlp2.loss_curve_)
        error_two_hl_nomom.append(mlp3.loss_curve_)
        error_two_hl_mom.append(mlp4.loss_curve_)

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
