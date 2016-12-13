from scipy.signal import decimate
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from mlp_nmist_project_functions import depickle, pickleme, plotrate, plotsuccess, fetch_MNIST_data

PLOT = False
PLOT_SAVE = True
per = depickle('PE_pkl_rng')
minPE = per[0]
maxPE = per[1]
decimation_rng = depickle('decimation_rng')
###################################################
# Analyze pickled data to determine best parameters
###################################################

best = 0.0
best_dec = np.zeros(decimation_rng-1)
best_pe_dec = np.zeros(decimation_rng-1)
best_lbl_dec = np.zeros(decimation_rng-1)
times = np.empty(decimation_rng-1)
lbls=['One HL No Momentum','One HL with Momentum', 'Two HLs no Momentum','Two HLs with Momentum']
for dec in range(1,decimation_rng):
    error_one_hl_nomom = depickle('error_one_hl_nomom_dsamp_' + str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs')
    error_one_hl_mom   = depickle('error_one_hl_mom_dsamp_' + str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs')
    error_two_hl_nomom = depickle('error_two_hl_nomom_dsamp_' + str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs')
    error_two_hl_mom   = depickle('error_two_hl_mom_dsamp_' + str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs')
    data_trn           = depickle('data_trn_dsamp_' + str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs')
    data_vld           = depickle('data_vld_dsamp_' + str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs')
    times[dec-1]       = depickle('ttl_time_dsamp_' + str(dec)+'_'+str(minPE*10)+'-'+str((maxPE-1)*10)+'PEs')

    print("With decimation factor of {} total time was {}").format(dec,times[dec-1])

    plotrate("{}x decimation - One HL No Momentum MLP {}-{} PEs".format(dec,10*minPE, 10*(maxPE-1)),error_one_hl_nomom,minPE,maxPE,PLOT=PLOT,PLOT_SAVE=PLOT_SAVE)
    plotrate("{}x decimation - One HL with Momentum MLP {}-{} PEs".format(dec,10*minPE, 10*(maxPE-1)),error_one_hl_mom,minPE,maxPE,PLOT=PLOT,PLOT_SAVE=PLOT_SAVE)
    plotrate("{}x decimation - Two HL No momentum MLP {}-{} PEs per layer".format(dec,10*minPE, 10*(maxPE-1)),error_two_hl_nomom,minPE,maxPE,PLOT=PLOT,PLOT_SAVE=PLOT_SAVE)
    plotrate("{}x decimation - Two HL with momentum MLP {}-{} PEs per layer".format(dec,10*minPE, 10*(maxPE-1)),error_two_hl_mom,minPE,maxPE,PLOT=PLOT,PLOT_SAVE=PLOT_SAVE)

    _,_,_ = plotsuccess("{}x decimation - Training Data Error".format(dec),data_trn,lbls,minPE,maxPE,PLOT=False,PLOT_SAVE=PLOT_SAVE)
    pe_idx,lbl_idx,err = plotsuccess("{}x decimation - Validation Data Error".format(dec),data_vld,lbls,minPE,maxPE,show_error=True,PLOT=False,PLOT_SAVE=PLOT_SAVE)

    # Find the best values for the current decimaiton level
    if err > best_dec[dec-1]:
        best_dec[dec-1] = err
        best_pe_dec[dec-1] = pe_idx*10 + minPE*10
        best_lbl_dec[dec-1] = lbl_idx

    # Find the best value against ALL decimation levels
    if err > best or err >= best and dec > best_decimation and pe_idx < best_pe:
        best = err
        best_pe = pe_idx
        best_lbl = lbl_idx
        best_decimation = dec

# Plot the most accurate model for each decimation level
labels = []
small_lbls=['1HL','1HLw/mom', '2HLs','2HLsw/mom']
for i in range(0,decimation_rng-1):
    labels.append("{}:{}".format(small_lbls[int(best_lbl_dec[i])], int(best_pe_dec[i])))
z=np.linspace(1,decimation_rng-1,decimation_rng-1)
fig, ax = plt.subplots()
ax.plot(z,best_dec)
for i, txt in enumerate(labels):
    ax.annotate(txt, (z[i],best_dec[i]))
title = "Best model per decimation level"
plt.title(title)
plt.xlabel("Decimation level")
plt.ylabel("Accuracy of Validation Set")
if PLOT_SAVE == True:
    title = title.replace(" ", "-")
    plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
if PLOT == True:
    plt.show()
plt.close('all')

# Plot the time cost for generating the models per decimatino level
times = times/max(times) # normalize times
plt.figure()
plt.plot(range(1,len(times)+1),times)
title = "Normalized times for training data for each decimation level with {}-{} PEs".format(10*minPE, 10*(maxPE-1))
plt.title(title)
plt.xlabel("X decimation")
plt.ylabel("Normalized time (s)")
if PLOT_SAVE == True:
    title = title.replace(" ", "-")
    plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
if PLOT == True:
    plt.show()
plt.close()

print("\nBest error on validation set: {}, with {} PEs, in {}x decimation: {}\n").format(best,(best_pe+minPE)*10,best_decimation,lbls[best_lbl])

#############################################################
# Find best training rate for optimal data from previous runs
#############################################################

X_train, y_train, X_validate, y_validate, _, _ = fetch_MNIST_data()
vl_lrn_rate = np.zeros(10)
best_vl = 0.0
best_lrn = 0.0
for i in range(1,11):
    print("\nTesting with {} PEs, {}x decimation, {}: learning rate = {}\n").format((best_pe+minPE)*10, best_decimation, lbls[best_lbl], .01*i)
    if best_lbl == 0: #One HL No Momentum
        mlp = MLPClassifier(hidden_layer_sizes=(best_pe+1)*10, activation='relu', momentum=0,max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.01*i)
    elif best_lbl == 1: #One HL with Momentum
        mlp = MLPClassifier(hidden_layer_sizes=(best_pe+1)*10, activation='relu', momentum=0.9,max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.01*i)
    elif best_lbl == 2: # Two HLs no Momentum
        mlp = MLPClassifier(hidden_layer_sizes=((best_pe+1)*10, (best_pe+1)*10), activation='relu', momentum=0.0, max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.01*i)
    elif best_lbl == 3: #One HL with Momentum
        mlp = MLPClassifier(hidden_layer_sizes=((best_pe+1)*10, (best_pe)*10), activation='relu', momentum=0.9, max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=False, tol=1e-4, random_state=1, learning_rate_init=.01*i)

    mlp.fit(X_train, y_train)
    tr_scr = mlp.score(X_train, y_train)
    vl_lrn_rate[i-1] = mlp.score(X_validate, y_validate)

    if vl_lrn_rate[i-1] > best_vl:
        best_vl = vl_lrn_rate[i-1]
        best_lrn = .01*i

optimal_predictor_stats = "{}:learning rate = {}, {} PEs, {}x Decimation --> yielding validation score of {}".format(lbls[best_lbl],best_lrn,(best_pe+minPE)*10, best_decimation,best_vl)
print optimal_predictor_stats
pickleme("optimal_predictor_for_{}-{}PEs_{}-declevels".format(10*minPE,(maxPE-1)*10,dec),optimal_predictor_stats)
plt.figure()
plt.plot(np.linspace(.01,.1,10),vl_lrn_rate)
title = "Recognition on validation set wrt learning rate"
plt.title(title)
plt.xlabel("Learning Rate")
plt.ylabel("Recognition Rate")
if PLOT_SAVE == True:
    title = title.replace(" ", "-")
    plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
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
