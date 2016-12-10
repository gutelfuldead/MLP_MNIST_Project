# MLP_NMIST_Project
Multi-Layer Perceptron Analysis on NMIST handwriting set

## ./src -- Contains all source python code

./src/generate_data_with_downsampling.py -- Creates the pickled data for both single and double layer perceptrons with and without momentum using gradient descent over a range of 10 --> 100 (in multiples of 10) number of Processing Elements per Hidden Layer. 

./src/analyze_data.py -- Creates plots based on the pickle files from generate_data_with_downsampling.py; Toggling `PLOT` and `PLOT_SAVE` between `True` and `False` to show all of the generated plots and/or save them to ../imgs respectively.

## ./pkls -- Contains all of the pickled data

## ./imgs -- Contains all of the plot images generated from src files
