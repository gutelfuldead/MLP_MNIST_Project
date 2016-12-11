#TODO: ITERATE THROUGH ARRAYS TO FIND THE AMOUNT OF PES THAT CORRELATES WITH THE BEST Convergence
# IN THE VALIDATION SET!!! IE PLOTSUCCESS FUNCTION
#TODO: RECREATE PLOT TITLES WAY TO VERBOSE!!!!!!

import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import pprint, pickle

PLOT      = True
PLOT_SAVE = True
dec = 3

##############################################
# MAIN
##############################################

def depickle(filename):
    pth = '../pkls/'
    fn = pth+filename+'.pkl'
    pkl_file = open(fn, 'rb')
    return pickle.load(pkl_file)


error_one_hl_nomom = depickle('error_one_hl_nomom_dsamp_' + str(dec))
error_one_hl_mom   = depickle('error_one_hl_mom_dsamp_' + str(dec))
error_two_hl_nomom = depickle('error_two_hl_nomom_dsamp_' + str(dec))
error_two_hl_mom   = depickle('error_two_hl_mom_dsamp_' + str(dec))
data_trn           = depickle('data_trn_dsamp_' + str(dec))
data_vld           = depickle('data_vld_dsamp_' + str(dec))
timetook           = depickle('ttl_time_dsamp_' + str(dec))

print("With decimation factor of {} total time was {}").format(dec,timetook)


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
    title = title + "; best error = %f @ %d PEs" % (mymin,10*idx)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Convergence Error")
    plt.legend()
    if PLOT_SAVE == True:
        plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
    if PLOT == True:
        plt.show()

plotrate("One hidden layer, no momentum convergence rate",error_one_hl_nomom)
plotrate("One hidden layer, with momentum convergence rate",error_one_hl_mom)
plotrate("Two hidden layers, no momentum convergence rate",error_two_hl_nomom)
plotrate("Two hidden layers, with momentum convergence rate",error_two_hl_mom)

def plotsuccess(title,data):
    mymax = 0.0
    idx_lbl = 0 # refers to which type of training was best
    idx_pes = 0
    lbls=['One hidden layer, no momentum','One hidden layer, w/ momentum', 'Two hidden layers, no momentum','Two hidden layers, w/ momentum']
    for i in range(0,4):
        if max(data[1::,i]) > mymax:
            mymax = max(data[1::,i])
            idx_lbl = i
            # for j in range(0,len(data[1::,i])):
                # print data.shape
                # if data[1::,i] == mymax:
                    # idx_pes = j

    plt.figure()
    plt.plot(np.linspace(20,100,9), data[1::,0],label=lbls[0])
    plt.plot(np.linspace(20,100,9), data[1::,1],label=lbls[1])
    plt.plot(np.linspace(20,100,9), data[1::,2],label=lbls[2])
    plt.plot(np.linspace(20,100,9), data[1::,3],label=lbls[3])
    plt.legend(loc='lower right')
    title2 = title + "; best error = %f for " % (mymax) + lbls[idx_lbl] #+ "%d PEs" % (idx_pes)
    plt.title(title2)
    plt.xlabel("Number of PEs per hidden layer")
    plt.ylabel("Final Convergence Error")
    if PLOT_SAVE == True:
        plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
    if PLOT == True:
        plt.show()

plotsuccess("Training Data Error",data_trn)
plotsuccess("Validation Data Error",data_vld)
