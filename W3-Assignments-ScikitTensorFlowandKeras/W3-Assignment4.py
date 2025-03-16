# Given
# from numpy import sqrt
# from pandas import read_csv
# from sklearn.model_selection import train_test_split
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np
#
# # load the dataset
# path =
# 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
# df = read_csv(path, header=None)
# # split into input and output columns
# X, y = df.values[:, :-1], df.values[:, -1]
# # split into train and test datasets
# X_train, X_test, y_train, y_test =
# .
# .
# .
#
# row2 =
# np.array([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.9
#            0, 4.98]])
# yhat = model.predict([row2])
# print('Predicted: %.3f' % yhat)


# Use Sequential model to develop a regression model using Boston Housing dataset,
# display
# c) Squared sum error
# d) Prediction results

from numpy import sqrt
from pandas import read_csv
from scipy.odr import Output
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
# numpy: Provides numerical operations (e.g., sqrt for square root).
# pandas: Used to load and process the dataset.
# sklearn.model_selection.train_test_split: Splits data into training and testing sets.
# tensorflow.keras.Sequential: Defines a sequential deep learning model.
# tensorflow.keras.layers.Dense: Creates fully connected (dense) layers in the neural network.
# scipy.odr stands for Orthogonal Distance Regression (ODR), which is a part of the SciPy library.
# It is used for fitting models to data where errors exist in both the independent (X) and dependent (Y) variables.
# ODR is useful when ordinary least squares (OLS) regression is not sufficient due to measurement errors in both X and Y.

# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(path, header=None)
# Reads the Boston Housing Prices dataset from an online CSV file.
# The dataset contains 13 input features and 1 target variable (house price).

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# X: Input features (all columns except the last one).
# y: Target values (last column, which represents house prices).

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# train_test_split() randomly splits the dataset:
# 67% (default) for training (X_train, y_train).
# 33% for testing (X_test, y_test).
# print() displays the shapes of training and test sets.

# determine the number of input features
n_features = X_train.shape[1]
# Determines the number of input features (should be 13 for this dataset).


# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))
# Sequential(): Defines a feedforward neural network.
# Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,))
# 10 neurons in the first hidden layer.
# ReLU activation function for non-linearity.
# He-normal initialization to handle deep networks effectively.
# Input layer expects 13 features.
# Dense(8, activation='relu', kernel_initializer='he_normal')
# 8 neurons in the second hidden layer.
# Dense(1)
# Output layer with 1 neuron (since we are predicting a single value: house price).


# compile the model
model.compile(optimizer='adam', loss='mse')
# optimizer='adam': Adam optimizer is used for efficient learning.
# loss='mse': The Mean Squared Error (MSE) loss function is used for regression problems.

# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# epochs=150: Trains for 150 iterations (epochs).
# batch_size=32: Processes 32 samples at a time.
# verbose=0: Suppresses detailed output

# evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))
# model.evaluate(X_test, y_test): Computes the loss (MSE) on the test set.
# sqrt(error): Computes Root Mean Squared Error (RMSE) for better interpretability

# make a prediction
row1 = [0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98]
row2 = np.array([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.90, 4.98]])
yhat = model.predict([row2])
print('Predicted: %.3f' % yhat)
# Defines a new house feature vector (row2).
# model.predict([row2]): Predicts the house price for the given input.

# Expected Output
# (339, 13) (167, 13) (339,) (167,)
# MSE: 20.543, RMSE: 4.533
# Predicted: 23.789

# The model is trained on 339 samples and tested on 167 samples.
# The MSE and RMSE indicate how well the model performed.
# The predicted house price is displayed for row2.
# Summary
# This program: ✅ Loads and prepares the Boston Housing Dataset
# ✅ Splits data into training & testing sets
# ✅ Builds a Neural Network with 2 hidden layers
# ✅ Trains the model with 150 epochs
# ✅ Evaluates model performance using MSE & RMSE
# ✅ Makes house price predictions