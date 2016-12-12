from scipy.signal import decimate
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from mlp_nmist_project_functions import depickle, pickleme, plotrate, plotsuccess, fetch_MNIST_data

PLOT = False
PLOT_SAVE = True

###################################################
# Analyze pickled data to determine best parameters
###################################################

decimation_rng = 6
best = 0.0
times = np.empty(decimation_rng-1)
lbls=['One HL No Momentum','One HL with Momentum', 'Two HLs no Momentum','Two HLs with Momentum']
for dec in range(1,decimation_rng):

    error_one_hl_nomom = depickle('error_one_hl_nomom_dsamp_' + str(dec))
    error_one_hl_mom   = depickle('error_one_hl_mom_dsamp_' + str(dec))
    error_two_hl_nomom = depickle('error_two_hl_nomom_dsamp_' + str(dec))
    error_two_hl_mom   = depickle('error_two_hl_mom_dsamp_' + str(dec))
    data_trn           = depickle('data_trn_dsamp_' + str(dec))
    data_vld           = depickle('data_vld_dsamp_' + str(dec))
    times[dec-1]       = depickle('ttl_time_dsamp_' + str(dec))

    print("With decimation factor of {} total time was {}").format(dec,times[dec-1])

    plotrate("{}x decimation - One HL No Momentum MLP 10-100 PEs".format(dec),error_one_hl_nomom,PLOT,PLOT_SAVE)
    plotrate("{}x decimation - One HL with Momentum MLP 10-100 PEs".format(dec),error_one_hl_mom,PLOT,PLOT_SAVE)
    plotrate("{}x decimation - Two HL No momentum MLP 10-100 PEs per layer".format(dec),error_two_hl_nomom,PLOT,PLOT_SAVE)
    plotrate("{}x decimation - Two HL with momentum MLP 10-100 PEs per layer".format(dec),error_two_hl_mom,PLOT,PLOT_SAVE)

    _,_,_ = plotsuccess("{}x decimation - Training Data Error".format(dec),lbls=lbls,data=data_trn,PLOT=PLOT,PLOT_SAVE=PLOT_SAVE)
    pe_idx,lbl_idx,err = plotsuccess("{}x decimation - Validation Data Error".format(dec),lbls=lbls,data=data_vld,show_error=True,PLOT=PLOT,PLOT_SAVE=PLOT_SAVE)

    if err > best:
        best = err
        best_pe = pe_idx
        best_lbl = lbl_idx
        best_decimation = dec

times = times/max(times) # normalize times
plt.figure()
plt.plot(range(1,len(times)+1),times)
plt.title("Normalized times for training data for each decimation level")
plt.xlabel("X decimation")
plt.ylabel("Normalized time (s)")
if PLOT_SAVE == True:
    plt.savefig('../imgs/TIME-all-decimation.png', bbox_inches='tight')
if PLOT == True:
    plt.show()
plt.close()

print("\nBest error on validation set: {}, with {} PEs, in {}x decimation: {}\n").format(best,(best_pe+1)*10,best_decimation,lbls[best_lbl])

#############################################################
# Find best training rate for optimal data from previous runs
#############################################################

X_train, y_train, X_validate, y_validate, _, _ = fetch_MNIST_data()
vl_lrn_rate = np.zeros(10)
best_vl = 0.0
best_lrn = 0.0
for i in range(1,11):
    print("\nTesting with {} PEs, {}x decimation, {}: learning rate = {}\n").format((best_pe+1)*10, best_decimation, lbls[best_lbl], .01*i)
    if best_lbl == 0: #One HL No Momentum
        mlp = MLPClassifier(hidden_layer_sizes=(best_pe+1)*10, activation='relu', momentum=0,max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.1*i)
    elif best_lbl == 1: #One HL with Momentum
        mlp = MLPClassifier(hidden_layer_sizes=(best_pe+1)*10, activation='relu', momentum=0.9,max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.1*i)
    elif best_lbl == 2: # Two HLs no Momentum
        mlp = MLPClassifier(hidden_layer_sizes=((best_pe+1)*10, (best_pe+1)*10), activation='relu', momentum=0.0, max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.1*i)
    elif best_lbl == 3: #One HL with Momentum
        mlp = MLPClassifier(hidden_layer_sizes=((best_pe+1)*10, (best_pe)*10), activation='relu', momentum=0.9, max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.1*i)

    mlp.fit(X_train, y_train)
    tr_scr = mlp.score(X_train, y_train)
    vl_lrn_rate[i-1] = mlp.score(X_validate, y_validate)

    if vl_lrn_rate[i-1] > best_vl:
        best_vl = vl_lrn_rate[i-1]
        best_lrn = .1*i
    
print("Best learning rate is = {} yielding validation score of {}").format(best_lrn, best_vl)
plt.figure()
plt.plot(np.linspace(.1,1,10),vl_lrn_rate)
plt.title("Recognition on validation set wrt learning rate")
plt.xlabel("Learning Rate")
plt.ylabel("Recognition Rate")
if PLOT_SAVE == True:
    plt.savefig('../imgs/best-learning-rate.png', bbox_inches='tight')
if PLOT == True:
    plt.show()
plt.close()

# Pickle Optimal Parameters for Neural Net
if best_lbl == 0:
    opt_param = {'hidden_layer_sizes' : (best_pe+1)*10, 'activation' : 'relu', 'momentum' : 0, 'max_iter' : 100,
                'alpha' : 1e-4, 'solver' : 'sgd', 'verbose' : False, 'tol' : 1e-4, 'random_state' : 1, 'learning_rate_init' : best_lrn}
if best_lbl == 1:
    opt_param = {'hidden_layer_sizes' : (best_pe+1)*10, 'activation' : 'relu', 'momentum' : 0.9, 'max_iter' : 100,
                'alpha' : 1e-4, 'solver' : 'sgd', 'verbose' : False, 'tol' : 1e-4, 'random_state' : 1, 'learning_rate_init' : best_lrn}
if best_lbl == 2:
    opt_param = {'hidden_layer_sizes' : ((best_pe+1)*10,(best_pe+1)*10), 'activation' : 'relu', 'momentum' : 0, 'max_iter' : 100,
                'alpha' : 1e-4, 'solver' : 'sgd', 'verbose' : False, 'tol' : 1e-4, 'random_state' : 1, 'learning_rate_init' : best_lrn}
if best_lbl == 3:
    opt_param = {'hidden_layer_sizes' : ((best_pe+1)*10,(best_pe+1)*10), 'activation' : 'relu', 'momentum' : 0.9, 'max_iter' : 100,
                'alpha' : 1e-4, 'solver' : 'sgd', 'verbose' : False, 'tol' : 1e-4, 'random_state' : 1, 'learning_rate_init' : best_lrn}

pickleme('optimal_parameters',opt_param)
pickleme('optimal_decimation',best_decimation)
