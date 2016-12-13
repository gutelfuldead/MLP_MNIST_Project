import math
import itertools
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata

def fetch_MNIST_data():
    mnist = fetch_mldata("MNIST original")
    X, y = mnist.data / 255., mnist.target
    X_train = X[:60000]
    y_train = y[:60000]
    p = depickle('random_index')
    x_shuffle = X[60000:]
    y_shuffle = y[60000:]
    x_shuffle = x_shuffle[p]
    y_shuffle = y_shuffle[p]
    X_validate, X_test = x_shuffle[:5000], x_shuffle[5000:]
    y_validate, y_test = y_shuffle[:5000], y_shuffle[5000:]
    return X_train, y_train, X_validate, y_validate, X_test, y_test

def depickle(filename):
    pkl_file = open('../pkls/'+filename+'.pkl', 'rb')
    fn = pickle.load(pkl_file)
    pkl_file.close()
    return fn

def pickleme(name,data):
    output = open('../pkls/'+name+'.pkl','wb')
    pickle.dump(data,output)
    output.close()
    return

def plotrate(title,data,minPE,maxPE,PLOT=False,PLOT_SAVE=False):
    mymin = 1000.0
    idx = 100
    plt.figure()
    for i in range(0,maxPE-minPE):
        label = "{} PEs".format(10*(i+minPE))
        plt.plot(range(0,len(data[i])),data[i],label=label)
        if min(data[i]) < mymin:
            mymin = min(data[i])
            idx = i
    titlen =  title + "\nlowest error = %f @ %d PEs" % (mymin,10*(idx+minPE))
    plt.title(titlen)
    plt.xlabel("Epochs")
    plt.ylabel("Convergence Error")
    plt.legend()
    if PLOT_SAVE == True:
        title = title.replace(" ", "-")
        plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
    if PLOT == True:
        plt.show()
    plt.close()
    return

def plotsuccess(title,data,lbls,minPE,maxPE,show_error=False,PLOT=False,PLOT_SAVE=False):
    # find num of PEs associated with max element
    i,j = np.unravel_index(data.argmax(), data.shape) # i refers to num PEs; j to the label
    plt.figure()
    # rng = np.linspace(10*minPE,10*maxPE,(maxPE-minPE))
    rng = range(minPE*10,maxPE*10,10)
    plt.plot(rng, data[:,0],label=lbls[0])
    plt.plot(rng, data[:,1],label=lbls[1])
    plt.plot(rng, data[:,2],label=lbls[2])
    plt.plot(rng, data[:,3],label=lbls[3])
    plt.legend(loc='lower right')
    if show_error == True:
        title2 = title + "\nbest error = %f for " % (data[i,j]) + lbls[j] + " with %d PEs (per layer)" % ((i+minPE)*10)
    else:
        title2 = title
    plt.title(title2)
    plt.xlabel("Number of PEs per hidden layer")
    plt.ylabel("Final Convergence Error")
    if PLOT_SAVE == True:
        title = title.replace(" ", "-")
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
    return

def plotweights(coefs,PLOT=False,PLOT_SAVE=False):
    # Plot some example Weights (looks like shit for decimated data set)
    a = int(math.floor(math.sqrt(len(coefs)))) # new dimension for image
    # use global min / max to ensure all weights are shown on the same scale
    title = "The first 16 weights"
    fig, axes = plt.subplots(4, 4)
    vmin, vmax = coefs.min(), coefs.max()
    for coef, ax in zip(coefs.T, axes.ravel()):
        coef = coef[0:a**2]
        ax.matshow(coef.reshape(a,a), cmap=plt.cm.gray, vmin=.5 * vmin,
                   vmax=.5 * vmax)
        ax.set_xticks(())
        ax.set_yticks(())
    if PLOT_SAVE == True:
        title = title.replace(" ", "-")
        plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
    if PLOT == True:
        plt.show()
    plt.close()
    return
