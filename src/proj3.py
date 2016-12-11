#
# http://stats.stackexchange.com/questions/19048/what-is-the-difference-between-test-set-and-validation-set

import numpy as np
import itertools
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

##################
# Confusion matrix
##################
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

##############################################
# MAIN
##############################################

# Fetch Data and split into training and test sets
mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
class_names = np.array(['0','1','2','3','4','5','6','7','8','9'])
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Create Classifier MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,100), activation='tanh', max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-4, random_state=1, learning_rate_init=.1)

# Train Data
mlp.fit(X_train, y_train)
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

plt.figure()
plt.plot(mlp.loss_curve_)
plt.show()

# Compute confusion matrix
y_pred = mlp.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix')
plt.show()

# Plot a few examples of incorrectly classified digits
wrong = []
for i in range(0,len(X_test)):
    if y_pred[i] != y_test[i]:
        wrong.append(i)

plt.figure()
offset = len(wrong)-15
for i in range(0,4):
    plt.subplot(2,2,i+1)
    plt.imshow(X_test[wrong[i+offset]].reshape(28,28))
    lbl = "Should be {}; classified as {}".format(y_test[wrong[i+offset]], y_pred[wrong[i+offset]])
    plt.title(lbl)
plt.show()

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
