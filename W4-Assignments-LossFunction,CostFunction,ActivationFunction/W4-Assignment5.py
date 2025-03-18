'''Given the following fragment code

from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# load the dataset
path =
'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# split into train and test datasets
X_train, X_test, y_train, y_test =
.
.
.

row2 =
np.array([[0.00632, 18.00, 2.310, 0, 0.5380, 6.5750, 65.20, 4.0900, 1, 296.0, 15.30, 396.9
           0, 4.98]])
yhat = model.predict([row2])
print('Predicted: %.3f' % yhat)

Use the Functional model to develop a regression model using Boston Housing dataset,
display
c) Squared sum error
d) Prediction results '''

from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow import keras
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification

# numpy: Used for mathematical operations (like square root).
# pandas: Used to read the dataset.
# sklearn.model_selection.train_test_split: Splits data into training and testing sets.
# tensorflow.keras: Used to build and train the neural network.

# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(path, header=None)
# The dataset is loaded from an online repository using pandas.read_csv().
# This dataset (Boston Housing dataset) contains 13 input features and 1 target value (house price).

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# X: Contains input features (all columns except the last one).
# y: Contains the target values (house prices).

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# 67% of data is used for training, and 33% for testing.
# train_test_split() randomly divides the dataset.

# determine the number of input features
n_features = X_train.shape[1]
# n_features stores the number of input features (13 for this dataset).

# define Input, Hidden and output layers
x_in = Input(shape=(n_features,)) #This defines the input layer of the neural network with 13 neurons.
x = Dense(10, activation='relu', kernel_initializer='he_normal')(x_in)
x_out = Dense(1)(x)
# Hidden Layer:
# 10 neurons.
# relu activation function (Rectified Linear Unit) for non-linearity.
# he_normal initialization improves training stability.
# Output Layer:
# A single neuron (since this is a regression task).
# No activation function (outputs a continuous value).

# define the model
model = keras.Model(inputs=x_in, outputs=x_out)
# This creates a Keras Model with the defined input and output layers.

# compile the model
model.compile(optimizer='adam', loss='mse')
# Optimizer: adam (Adaptive Moment Estimation) is used for efficient training.
# Loss Function: mse (Mean Squared Error) is used since this is a regression task.

# fit/Training the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# The model is trained for 150 epochs (iterations over the dataset).
# Batch size of 32: Processes 32 samples at a time.
# verbose=0: Suppresses output logs.

# evaluate the model
error = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f' % (error, sqrt(error)))
# Evaluates the model on the test set.
# Computes Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).
# Lower RMSE indicates better performance.
# make a prediction

#Making a prediction
row1 = [0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]
row2 = np.array([[0.00632,18.00,2.310,0,0.5380,6.5750,65.20,4.0900,1,296.0,15.30,396.90,4.98]])
yhat = model.predict([row2])
print('Predicted: %.3f' % yhat)
# A single sample (house data) is given as input to predict the price.
# The model predicts and prints the estimated house price.

# Summary
# ✅ Dataset: Boston Housing Dataset (Predicts house prices).
# ✅ Model: A simple feedforward neural network.
# ✅ Architecture:
#
# 1 input layer (13 features).
# 1 hidden layer (10 neurons, ReLU).
# 1 output layer (1 neuron, continuous output). ✅ Training: 150 epochs, batch size 32.
# ✅ Loss Function: Mean Squared Error (MSE).
# ✅ Evaluation: RMSE to measure prediction accuracy.
# ✅ Prediction: Predicts house price from given features.
