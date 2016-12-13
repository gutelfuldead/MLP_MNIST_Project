Multi-Layer Perceptron Analysis on MNIST handwriting set
--------------------------------------------------------
Done for completion of UFL EEL5840: Elements of Machine Intelligence.

All data necessary to view final results has been pre-populated in the git.

To view results run `./src/classify_mnist_with_optimal_nn.py`

To regenerate all data run in order:

1. `python ./src/generate_data_with_downsampling.py minPE maxPE DecimationRange**`

2. `python ./src/find_optimal_parameters.py`

3. `python ./src/classify_mnist_with_optimal_nn.py`

Structure:
```
MLP_MNIST_Project
|   README.md
|___docs
|   |   Project3.pdf      # project description and requirements
|   |   proj3_results.pdf # analysis of the systems results
|___imgs
|   |   Contains images produced form source code
|___pkls
|   |   Contains .pkl files from pre-generated data and location for newly generated data
|___src
|   |   classify_mnist_with_optimal_nn.py  # used to classify with optimal ANN
|   |   find_optimal_parameters.py         # used to find optimal ANN
|   |   generate_data_with_downsampling.py # used to create all the models
|   |   mlp_mnist_project_functions.py     # Functions used throughout programs
```

## ./src

**./src/generate_data_with_downsampling.py**

*Usage*

`python ./src/generate_data_with_downsampling.py minPE maxPE DecimationRng`

* minPE : must be multiple of 10; is the minimum number of Neurons used in each configuration
* maxPE : must be multiple of 10; is the maximum number of Neurons used in each configuration
* DecimationRng : Will iterate over all configurations(minPE -> maxPE) over a decimation level of training data

4 Different Configurations used:

1. Single hidden layer ANN with no momentum

1. Single hidden layer ANN with momentum

1. Double hidden layer ANN with no momentum (equal number of neurons in each layer)

1. Double hidden layer ANN with momentum (equal number of neurons in each layer)

**Will create a total of 4 x DecimationRng x (maxPE-minPE + 1)/10 unique runs**

Imports MNIST data set and splits last 10k values into two groups, validation and testing and pickles it for later usage.

Data distribution:

| Data set| number data points |
|---------|--------------------|
|Training   | 60k|
|Validation | 5k |
|Testing    | 5k |


**./src/find_optimal_parameters.py**

Creates plots based on the pickle files from generate_data_with_downsampling.py and finds the optimal configuration

Toggling `PLOT` and `PLOT_SAVE` between `True` and `False` to show all of the generated plots and/or save them to ../imgs respectively.

Tweaks optimal configuration by finding the best learning rate for that configuration using the training data again

pickles the configurations for the best MLP

**./src/classify_mnist_with_optimal_nn.py**

Uses the optimal configurations from ./src/find_optimal_parameters.py to classify the training, validation, and testing data

Generates plots of the learning curve of the optimal MLP and produces a confusion matrix for the test data's classification

**./src/mlp_mnist_project_functions.py**

various functions made for this project
