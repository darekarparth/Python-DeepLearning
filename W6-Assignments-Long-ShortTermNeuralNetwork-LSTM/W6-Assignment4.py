'''
Consider the following sequence
We want to generate the following set of sequences, in this case the window size is 3
Use the Sequential model and “SimpleRNN” to develop a stack RNN with the following tasks
a) Construct 4 hidden layers with “SimpleRNN”, the respective layers consist of 20, 32,
16, 8 neurons
b) Add “return_sequences=True”
c) Add  “tanh” activation function in one layer
d) The output layer with “Linear” activation function
e) Add optimizer with “Adm”, loss function with “mse”, and metrics with “mae”
f) Run your model with epochs = 10, 50, 500
g) Evaluate the model with the training set of sequences
h) Given an input sequence [2, 3, 4], make predictions

'''

from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np
from numpy import asarray
from numpy import sqrt
# Sequential: Used to define a sequential neural network model.
# SimpleRNN: A recurrent neural network (RNN) layer that processes sequential data.
# Dense: Fully connected (FC) layer for output prediction.
# numpy: Used for handling numerical arrays.
# asarray: Converts input lists to NumPy arrays.
# sqrt: Computes the square root (used for RMSE calculation).


train = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
windowSize, X_train, y_train = 3, [], []
for index in range(len(train)-windowSize):
    X_train.append(train[index:index+windowSize])
    y_train.append(train[index+windowSize])
# train is a simple time series: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# windowSize = 3: The model will look at the last 3 numbers to predict the next one.
# X_train: Stores sequences of 3 consecutive numbers (input).
# y_train: Stores the next number after each 3-number sequence (output).

# X_train (Input)	y_train (Target Output)
# [1, 2, 3]	4
# [2, 3, 4]	5
# [3, 4, 5]	6
# [4, 5, 6]	7
# [5, 6, 7]	8
# [6, 7, 8]	9
# [7, 8, 9]	10

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((len(X_train), 3, 1))
# Converts lists to NumPy arrays.
# Reshapes X_train into (samples, time steps, features):
# Samples = Number of sequences (7 in this case).
# Time steps = Each sequence has 3 time steps.
# Features = Each time step has 1 feature (a single number).



model = Sequential()
model.add(SimpleRNN(20, input_shape=(3,1), return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(16))
model.add(Dense(8,activation='tanh'))
model.add(Dense(1,activation='linear'))
# Sequential(): Initializes a sequential model.
# SimpleRNN(20, input_shape=(3,1), return_sequences=True):
# First RNN layer with 20 neurons.
# input_shape=(3,1): Input shape has 3 time steps, 1 feature each.
# return_sequences=True: Outputs entire sequence (needed for stacking RNN layers).
# SimpleRNN(32, return_sequences=True): Second RNN layer with 32 neurons.
# SimpleRNN(16): Third RNN layer with 16 neurons (outputs final hidden state).
# Dense(8, activation='tanh'): Fully connected layer with 8 neurons and tanh activation.
# Dense(1, activation='linear'): Final layer for output (predicts a single number).


model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
model.fit(X_train,y_train, epochs=50)
#model.fit(X_train,y_train, epochs=500)
#history = model.fit(X_train,y_train,epochs=50)
# loss='mse': Uses Mean Squared Error (MSE) as the loss function.
# optimizer='Adam': Uses the Adam optimizer for efficient training.
# metrics=['mae']: Tracks Mean Absolute Error (MAE).
# epochs=50: Trains for 50 iterations.



mse, mae = model.evaluate(X_train, y_train, verbose=0)
print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, sqrt(mse), mae))
# Evaluates the model on training data.
# Computes MSE, RMSE (Root Mean Squared Error), and MAE.
# Prints the performance metrics.

row = np.array([[2, 3, 4]])
row1 = row.reshape(1, windowSize, 1)
yhat = model.predict([row1])
# Creates a test input [2, 3, 4].
# Reshapes it into (1, 3, 1) (same format as training data).
# predict([row1]): Predicts the next number.

row2 = asarray([2, 3, 4]).reshape((1, windowSize, 1))
yhat = model.predict(row2)
print(yhat)
# Another way to prepare the test input.
# Predicts again and prints the result.

# Summary
# ✔ Goal: Predict the next number in a sequence using SimpleRNN.
# ✔ Data Preparation: Uses sliding window (size 3) to create input-output pairs.
# ✔ Model: Stacks 3 SimpleRNN layers + 2 Dense layers.
# ✔ Training: Uses MSE loss and Adam optimizer.
# ✔ Evaluation: Computes MSE, RMSE, and MAE.
# ✔ Prediction: Given [2,3,4], predicts ~5.