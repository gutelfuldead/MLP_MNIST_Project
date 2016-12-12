# Training set score: 0.999980
# Validation set score: 0.382200
# Test set score: 0.883300
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from scipy.signal import decimate
from mlp_nmist_project_functions import depickle, pickleme, plot_confusion_matrix, fetch_MNIST_data


PLOT      = True
PLOT_SAVE = True

##############################################
# MAIN
##############################################

# load the optimal parameters generated from find_optimal_parameters.py
opt_parameters = depickle('optimal_parameters')
best_decimation = depickle('optimal_decimation')

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
    title = title.replace(" ", "")
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
    title = title.replace(" ", "")
    plt.savefig('../imgs/confusion-matrix.png', bbox_inches='tight')
if PLOT == True:
    plt.show()
plt.close()

# Plot a few examples of incorrectly classified digits
wrong = []
for i in range(0,len(X_test)):
    if y_pred[i] != y_test[i]:
        wrong.append(i)

# plt.figure()
# offset = len(wrong)-15
# for i in range(0,4):
#     plt.subplot(2,2,i+1)
#     plt.imshow(X_test[wrong[i+offset]].reshape(a,a))
#     lbl = "Should be {}; classified as {}".format(y_test[wrong[i+offset]], y_pred[wrong[i+offset]])
#     plt.title(lbl)
# plt.title('Examples of misclassified digits in optimal predictor')
# if PLOT_SAVE == True:
#     title = title.replace(" ", "")
#     plt.savefig('../imgs/'+title+'.png', bbox_inches='tight')
# if PLOT == True:
#     plt.show()
# plt.close()

# Plot some example Weights
# fig, axes = plt.subplots(4, 4)
# # use global min / max to ensure all weights are shown on the same scale
# vmin, vmax = mlp.coefs_[0].min(), mlp.coefs_[0].max()
# for coef, ax in zip(mlp.coefs_[0].T, axes.ravel()):
#     ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
#                vmax=.5 * vmax)
#     ax.set_xticks(())
#     ax.set_yticks(())
#
# plt.show()
