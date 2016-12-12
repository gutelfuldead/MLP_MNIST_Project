# MLP_NMIST_Project
Multi-Layer Perceptron Analysis on NMIST handwriting set

To view results run ./src/classify_mnist_with_optimal_nn.py

All data necessary to run this has been prepopulated in the git

To regenerate all data run in order:
1. ./src/generate_data_with_downsampling.py
2. ./src/find_optimal_parameters.py
3. ./src/classify_mnist_with_optimal_nn.py

## ./src -- Contains all source python code

./src/generate_data_with_downsampling.py

> Imports MNIST data set and splits last 10k values into two groups, validation and testing and pickles it for later usage
> iterates over 200 different configurations of multilayer perceptrons with the training data
> 5 different decimations on the training data size, 10 different levels of neuron density, and 4 different unique MLP configurations with single or double hidden layers and with or without momentum
> pickles all of the generated MLP data for later Analysis
> tests all of the configurations using both the training data and validation data and pickles the results

./src/find_optimal_parameters.py

> Creates plots based on the pickle files from generate_data_with_downsampling.py and finds the optimal configuration
> Toggling `PLOT` and `PLOT_SAVE` between `True` and `False` to show all of the generated plots and/or save them to ../imgs respectively.
> Tweaks optimal configuration by finding the best learning rate for that configuration using the training data again
> pickles the configurations for the best MLP

./src/classify_mnist_with_optimal_nn.py

> Uses the optimal configurations from ./src/find_optimal_parameters.py to classify the training, validation, and testing data
> Generates plots of the learning curve of the optimal MLP and produces a confusion matrix for the test data's classification

./src/mlp_nmist_project_functions.py

> various functions made for this project

## ./pkls -- Contains all of the pickled data

## ./imgs -- Contains all of the plot images generated from ./src files

## ./docs -- Contains project description and results
