import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets

# different learning rate schedules and momentum parameters
params = [{'hidden_layer_sizes' : (100), 'activation': 'relu', 'max_iter' : 100, 'alpha' :1e-4,'nesterovs_momentum': False,
            'solver' : 'sgd', 'verbose' : 10, 'tol' : 1e-4, 'random_state' : 1, 'learning_rate_init' : .1, 'momentum' : 0},
          {'hidden_layer_sizes' : (100), 'activation': 'relu', 'max_iter' : 100, 'alpha' :1e-4,'nesterovs_momentum': False,
            'solver' : 'sgd', 'verbose' : 10, 'tol' : 1e-4, 'random_state' : 1, 'learning_rate_init' : .1, 'momentum' : 0.9},
          {'hidden_layer_sizes' : (100,100), 'activation': 'relu', 'max_iter' : 100, 'alpha' :1e-4,'nesterovs_momentum': False,
            'solver' : 'sgd', 'verbose' : 10, 'tol' : 1e-4, 'random_state' : 1, 'learning_rate_init' : .1, 'momentum' : 0},
          {'hidden_layer_sizes' : (100,100), 'activation': 'relu', 'max_iter' : 100, 'alpha' :1e-4,'nesterovs_momentum': False,
            'solver' : 'sgd', 'verbose' : 10, 'tol' : 1e-4, 'random_state' : 1, 'learning_rate_init' : .1, 'momentum' : 0.9}]

labels = ["SLP w/ 100 PEs, Momentum = 0", "SLP w/ 100 PEs, Momentum = 0.9",
          "2LP w/ 100 PEs (each), Momentum = 0", "2LP w/ 100 PEs (each), Momentum = 0.9"]

plot_args = [{'c': 'red', 'linestyle': '-'},
             {'c': 'green', 'linestyle': '--'},
             {'c': 'blue', 'linestyle': '-'},
             {'c': 'red', 'linestyle': '--'},]

def plot_on_dataset(X, y, ax, name):
    # for each dataset, plot learning for each learning strategy
    print("\nlearning on dataset %s" % name)
    ax.set_title(name)
    X = MinMaxScaler().fit_transform(X)
    mlps = []

    for label, param in zip(labels, params):
        print("training: %s" % label)
        mlp = MLPClassifier(**param)
        mlp.fit(X, y)
        mlps.append(mlp)
        print("Training set score: %f" % mlp.score(X, y))
        print("Training set loss: %f" % mlp.loss_)
    for mlp, label, args in zip(mlps, labels, plot_args):
            ax.plot(mlp.loss_curve_, label=label, **args)

# Fetch Data and split into training and test sets
mnist = fetch_mldata("MNIST original")
X, y = mnist.data / 255., mnist.target
class_names = np.array(['0','1','2','3','4','5','6','7','8','9'])
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# load / generate some toy datasets
data_sets = [(X_train, y_train)]

for ax, data, name in zip(axes.ravel(), data_sets, ['training']):
    plot_on_dataset(*data, ax=ax, name=name)

fig.legend(ax.get_lines(), labels=labels, ncol=3, loc="upper center")
plt.savefig('plot_mlp_training_cruves.png')
plt.show()
