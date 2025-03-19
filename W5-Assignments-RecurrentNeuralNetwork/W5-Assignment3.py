'''
Consider the following sequence
We want to generate the following set of sequences as x and predict y values, in this case the
window size is 3
Use the Sequential model to develop a simple RNN neural network with the following tasks
a) Construct one hidden layer with “SimpleRNN” composed of 10 neurons
b) The output layer is with “Linear” activation function
c) Add optimizer with “Adm”, loss function with “mse”, and metrics with “mae”
d) Run your model with epochs = 10, 50, 500
e) Evaluate the model with the training set of sequences
f) Given an input sequence [2, 3, 4], make predictions

from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np
from numpy import asarray
from numpy import sqrt
train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
windowSize, X_train, y_train = 3, [], []
for index in range(len(train)-windowSize):
X_train.append(train[index:index+windowSize])
y_train.append(train[index+windowSize])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((len(X_train), 3, 1))
model =
.
.
.
'''

from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np
from numpy import asarray
from numpy import sqrt
# Sequential: A model type in Keras where layers are stacked sequentially.
# SimpleRNN: A basic Recurrent Neural Network layer that processes sequential data.
# Dense: A fully connected layer.
# numpy: For handling numerical arrays and reshaping data.

# Creating the Training Dataset
train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
windowSize, X_train, y_train = 3, [], []  # set window size and initialize lists
for index in range(len(train) - windowSize):
    X_train.append(train[index:index + windowSize])
    y_train.append(train[index + windowSize])
# train is a simple time series dataset: [1, 2, 3, 4, ..., 10].
# Sliding Window Approach:
# The windowSize is 3, meaning we use 3 previous values to predict the next one.
# X_train stores sequences of 3 numbers.
# y_train stores the next number corresponding to each sequence.
#
# X_train (Input)	y_train (Target)
# [1, 2, 3]	         4
# [2, 3, 4]          5
# [3, 4, 5]	         6
# ...	...

# Converting to NumPy Arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((len(X_train), 3, 1))  # reshape X_train to proper 3-d array
# X_train and y_train are converted to NumPy arrays for efficient computation.
# Since RNNs expect 3D input, we reshape X_train to (samples, time steps, features):
# samples = number of sequences (e.g., 7 sequences)
# time steps = 3 (window size)
# features = 1 (each value is a single feature)


#  Defining the RNN Model
model = Sequential()
model.add(SimpleRNN(10, input_shape=(3, 1)))
model.add(Dense(1, activation='linear'))
# A Sequential model is created.
# SimpleRNN(10): Adds an RNN layer with 10 neurons.
# input_shape=(3,1): Defines input as sequences of 3 time steps with 1 feature each.
# Dense(1, activation='linear'): A fully connected layer with 1 neuron (for output).

# Compiling and Training the Model
model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
model.fit(X_train, y_train, epochs=500) # history = model.fit(X_train,y_train,epochs=50)
# A Sequential model is created.
# SimpleRNN(10): Adds an RNN layer with 10 neurons.
# input_shape=(3,1): Defines input as sequences of 3 time steps with 1 feature each.
# Dense(1, activation='linear'): A fully connected layer with 1 neuron (for output).

mse, mae = model.evaluate(X_train, y_train, verbose=0)
print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, sqrt(mse), mae))
# The model is evaluated on the training data.
# MSE (Mean Squared Error), RMSE (Root Mean Squared Error), and MAE (Mean Absolute Error) are printed.

row = np.array([[2, 3, 4]])
row1 = row.reshape(1, windowSize, 1)
yhat = model.predict([row1])
row2 = asarray([2, 3, 4]).reshape((1, windowSize, 1))
yhat = model.predict(row2)
print(yhat)
# The model is given an input sequence [2, 3, 4] to predict the next number.
# The input is reshaped to (1, 3, 1) before prediction.
# yhat contains the predicted next value.

# Summary
# This program builds an RNN-based time series predictor.
# The dataset consists of a simple sequence [1,2,3,...,10].
# A sliding window approach is used to create training data.
# A SimpleRNN layer with 10 neurons processes the sequences.
# A Dense output layer predicts the next number in the series.
# The model is trained for 500 epochs and then used for prediction.
# Expected Output
# For X = [2, 3, 4], the expected output should be close to 5 (since the series is linear),
# but the actual value depends on training quality
