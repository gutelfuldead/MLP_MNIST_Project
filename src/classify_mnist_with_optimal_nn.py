import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from scipy.signal import decimate
from mlp_nmist_project_functions import depickle, plotweights, pickleme, plot_confusion_matrix, fetch_MNIST_data

PLOT      = True
PLOT_SAVE = True

# load the optimal parameters generated from find_optimal_parameters.py
opt_parameters = depickle('optimal_parameters')
best_decimation = depickle('optimal_decimation')
pe_rng = depickle('PE_pkl_rng')
minPE, maxPE = pe_rng[0], pe_rng[1]
dec = depickle('decimation_rng')
best_ann = depickle("optimal_predictor_for_{}-{}PEs_{}-declevels".format(10*minPE,(maxPE-1)*10,dec-1))
print "Using optimal predictor:"
print best_ann

# class_names used for confusion matrix
class_names = np.array(['0','1','2','3','4','5','6','7','8','9'])

# Fetch Data and split into training, validation, and test sets
X_train, y_train, X_validate, y_validate, X_test, y_test = fetch_MNIST_data()

# decimate data down to optimal value
X_train = decimate(X_train, best_decimation, axis=1)
X_validate = decimate(X_validate, best_decimation, axis=1)
X_test = decimate(X_test, best_decimation, axis=1)

# Create Classifier MLP
mlp = MLPClassifier(**opt_parameters)

# Train Data
mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Validation set score: %f" % mlp.score(X_validate, y_validate))
print("Test set score: %f" % mlp.score(X_test, y_test))

# Plot optimal training curve
plt.figure()
plt.plot(mlp.loss_curve_)
title = "The adaptation for the optimal neural network"
plt.title(title)
plt.xlabel("Epochs")
plt.ylabel("Training Error")
if PLOT_SAVE == True:
    title = title.replace(" ", "-")
    plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
if PLOT == True:
    plt.show()
plt.close()

# Compute confusion matrix
y_pred = mlp.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')
if PLOT_SAVE == True:
    plt.savefig('../imgs/confusion-matrix.png', bbox_inches='tight')
if PLOT == True:
    plt.show()
plt.close()

# find distribution of wrong values
wrong_index = []
wrong_value = []
for i in range(0,len(X_test)):
    if y_pred[i] != y_test[i]:
        wrong_index.append(i)
        wrong_value.append(y_test[i])

# plot histogram of wrong values
plt.figure()
plt.hist(wrong_value)
title = "Histogram of missclassified digits"
plt.title(title)
plt.xlabel("Misclassified digits")
plt.ylabel("Frequency")
if PLOT_SAVE == True:
    title = title.replace(" ", "-")
    plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
if PLOT == True:
    plt.show()
plt.close()

print "Total number of misclassified digits = %d" % len(wrong_value)

wrong_digit = np.zeros(9)
for k in range(1,10):
    for i in range(0,len(wrong_value)):
        if wrong_value[i] == k:
            wrong_digit[k-1] = int(wrong_index[i])

# Plot a few examples of incorrectly classified digits
plotweights(mlp.coefs_[0],PLOT,PLOT_SAVE)
X_train, y_train, X_validate, y_validate, X_test, y_test = fetch_MNIST_data()
plt.figure()
for i in range(0,9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[wrong_digit[i]].reshape(28,28))
    plt.axis('off')
    lbl = "{}->{}".format(y_test[wrong_digit[i]], y_pred[wrong_digit[i]])
    plt.title(lbl)
if PLOT_SAVE == True:
    plt.savefig('../imgs/misclassified_digits.png', bbox_inches='tight')
if PLOT == True:
    plt.show()
plt.close()
