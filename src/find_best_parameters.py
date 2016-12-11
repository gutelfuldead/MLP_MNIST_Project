from scipy.signal import decimate
import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pprint, pickle

PLOT      = False
PLOT_SAVE = True

###########
# Functions
###########

def depickle(filename):
    pth = '../pkls/'
    fn = pth+filename+'.pkl'
    pkl_file = open(fn, 'rb')
    return pickle.load(pkl_file)

def plotrate(title,data):
    mymin = 1000.0
    idx = 100
    plt.figure()
    for i in range(0,9):
        label = "{} PEs".format(10*(i+1))
        plt.plot(range(0,len(data[i])),data[i],label=label)
        if min(data[i]) < mymin:
            mymin = min(data[i])
            idx = i
    titlen =  title + "\nlowest error = %f @ %d PEs" % (mymin,10*idx)
    plt.title(titlen)
    plt.xlabel("Epochs")
    plt.ylabel("Convergence Error")
    plt.legend()
    if PLOT_SAVE == True:
        title = title.replace(" ", "")
        plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
    if PLOT == True:
        plt.show()
    plt.close()

def plotsuccess(title,data,lbls,show_error=False):
    # find num of PEs associated with max element
    i,j = np.unravel_index(data.argmax(), data.shape) # i refers to num PEs; j to the label
    plt.figure()
    plt.plot(np.linspace(20,100,9), data[1::,0],label=lbls[0])
    plt.plot(np.linspace(20,100,9), data[1::,1],label=lbls[1])
    plt.plot(np.linspace(20,100,9), data[1::,2],label=lbls[2])
    plt.plot(np.linspace(20,100,9), data[1::,3],label=lbls[3])
    plt.legend(loc='lower right')
    if show_error == True:
        title2 = title + "\nbest error = %f for " % (data[i,j]) + lbls[j] + " with %d PEs (per layer)" % ((i+1)*10)
    else:
        title2 = title
    plt.title(title2)
    plt.xlabel("Number of PEs per hidden layer")
    plt.ylabel("Final Convergence Error")
    if PLOT_SAVE == True:
        title = title.replace(" ", "")
        plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
    if PLOT == True:
        plt.show()
    plt.close()
    return i,j,data[i,j]

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
        horizontalalignment="center",
        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

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

    plotrate("{}x decimation - One HL No Momentum MLP 10-100 PEs".format(dec),error_one_hl_nomom)
    plotrate("{}x decimation - One HL with Momentum MLP 10-100 PEs".format(dec),error_one_hl_mom)
    plotrate("{}x decimation - Two HL No momentum MLP 10-100 PEs per layer".format(dec),error_two_hl_nomom)
    plotrate("{}x decimation - Two HL with momentum MLP 10-100 PEs per layer".format(dec),error_two_hl_mom)

    _,_,_ = plotsuccess("{}x decimation - Training Data Error".format(dec),lbls=lbls,data=data_trn)
    pe_idx,lbl_idx,err = plotsuccess("{}x decimation - Validation Data Error".format(dec),lbls=lbls,data=data_vld,show_error=True)

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

mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
X_train, X_validate, X_test = X[:60000], X[60000:70000], X[70000:]
y_train, y_validate, y_test = y[:60000], y[60000:70000], X[70000:]
X_train = decimate(X_train, best_decimation, axis=1)
X_validate = decimate(X_validate,best_decimation,axis=1)

vl_lrn_rate = np.zeros(10)
best_vl = 0.0
best_lrn = 0.0
for i in range(1,11):
    print("\nTesting with {} PEs, {}x decimation, {}: learning rate = {}\n").format((best_pe+1)*10, best_decimation, lbls[best_lbl], .01*i)
    if best_lbl == 0: #One HL No Momentum
        mlp = MLPClassifier(hidden_layer_sizes=(best_pe+1)*10, activation='relu', momentum=0,max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.01*i)
    elif best_lbl == 1: #One HL with Momentum
        mlp = MLPClassifier(hidden_layer_sizes=(best_pe+1)*10, activation='relu', momentum=0.9,max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.01*i)
    elif best_lbl == 2: # Two HLs no Momentum
        mlp = MLPClassifier(hidden_layer_sizes=((best_pe+1)*10, (best_pe+1)*10), activation='relu', momentum=0.0, max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.01*i)
    elif best_lbl == 3: #One HL with Momentum
        mlp = MLPClassifier(hidden_layer_sizes=((best_pe+1)*10, (best_pe)*10), activation='relu', momentum=0.9, max_iter=100, alpha=1e-4,
                            solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.01*i)

    mlp.fit(X_train, y_train)
    tr_scr = mlp.score(X_train, y_train)
    vl_lrn_rate[i-1] = mlp.score(X_validate, y_validate)

    if vl_lrn_rate[i-1] > best_vl:
        best_vl = vl_lrn_rate[i-1]
        best_lrn = .01*i

print("Best learning rate is = {} yielding validation score of {}").format(best_lrn, best_vl)
plt.figure()
plt.plot(np.linspace(.01,.1,10),vl_lrn_rate)
plt.title("Recognition on validation set wrt learning rate")
plt.xlabel("Learning Rate")
plt.ylabel("Recognition Rate")
if PLOT_SAVE == True:
    plt.savefig('../imgs/best-learning-rate.png', bbox_inches='tight')
if PLOT == True:
    plt.show()
plt.close()
