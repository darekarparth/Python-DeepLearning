'''
Given the following fragment code
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)
# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# determine the number of input features
n_features = X_train.shape[1]
.
.
.

Follow 5 steps introduced in Lecture 3 and use the Sequential model to develop a multiclass
classification model with the following tasks:

f)Construct 3 hidden layers, the fist hidden layer consists of 20 neurons with “relu”
activation function, the second composed of 10 neurons with “tanh” activation
function, the third composed of 8 neurons with “sigmoid” activation function.
g) The final layer is with “softmax” activation function (think about why? - Gives Probablities at output)
h) Save the trained model
i) Load the model.Given an array[5.1, 3.5, 1.4, 0.2], use the loaded model to predict its
result
'''

from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
# Numpy (numpy): Used for numerical computations.
# Pandas (read_csv): Used for loading the dataset.
# Scikit-learn (train_test_split, LabelEncoder): Used for data preprocessing.
# TensorFlow/Keras (Sequential, Dense): Used for building and training the neural network.


# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)
# The Iris dataset is loaded from an online source using read_csv().
# This dataset contains 150 samples of iris flowers, with 4 input features and 1 target label.

# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# X: The first 4 columns (features: sepal length, sepal width, petal length, petal width).
# y: The last column (flower species name).


# ensure all data are floating point values
X = X.astype('float32')
# Converts feature values to float32 format for compatibility with TensorFlow.

# encode strings to integer
y = LabelEncoder().fit_transform(y)
# The LabelEncoder converts categorical labels (species names) into integer values:
# Setosa → 0
# Versicolor → 1
# Virginica → 2

# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Splits the dataset into training (67%) and testing (33%) subsets.


# determine the number of input features
n_features = X_train.shape[1] #Extracts the number of input features (4, for the 4 iris measurements).
# define model
model = Sequential()
model.add(Dense(20, activation='relu', kernel_initializer='he_normal',input_shape=(n_features,)))
model.add(Dense(10, activation='tanh', kernel_initializer='he_normal'))
model.add(Dense(8, activation='sigmoid', kernel_initializer='he_normal'))
model.add(Dense(3, activation='softmax'))
# The Sequential model is a feedforward multi-layer neural network.
# Layers:
# Input Layer: Dense(20, activation='relu')
# 20 neurons with ReLU activation.
# he_normal initializes weights using He Initialization.
# Hidden Layer: Dense(10, activation='tanh')
# 10 neurons with Tanh activation.
# Hidden Layer: Dense(8, activation='sigmoid')
# 8 neurons with Sigmoid activation.
# Output Layer: Dense(3, activation='softmax')
# 3 neurons for 3 iris species.
# Softmax activation to get probability scores for each class.

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Optimizer: adam (Adaptive Moment Estimation) for efficient learning.
# Loss Function: sparse_categorical_crossentropy (used for multi-class classification).
# Metrics: accuracy (to evaluate performance)

# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# Trains the model using:
# epochs=150: Runs the training for 150 iterations.
# batch_size=32: Processes 32 samples at a time.
# verbose=0: No training logs displayed

# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
# Tests the model on unseen test data.
# Prints the accuracy (e.g., 0.980 means 98% accuracy).

model.save('model2.h5')
# Saves the trained model as model2.h5

from tensorflow.keras.models import load_model
# load the model from file
model = load_model('model2.h5')
# Reloads the saved model for later use.


# make a prediction
row1 = [5.1, 3.5, 1.4, 0.2]
row2 = np.array([[5.1, 3.5, 1.4, 0.2]])
# #Defines a sample flower measurement.

yhat = model.predict([row2])
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
# Uses model.predict() to get the probability scores for each class.
# argmax(yhat) selects the class with the highest probability.

# Expected output
# Predicted: [[0.98 0.01 0.01]] (class=0)
# This means the model predicts class 0 (Setosa) with 98% confidence

# ✅ Loads the Iris dataset.
# ✅ Preprocesses the data (converts labels and splits into train/test sets).
# ✅ Defines a deep neural network with 3 hidden layers.
# ✅ Trains the model on the dataset.
# ✅ Evaluates its accuracy on the test dataset.
# ✅ Saves and reloads the trained model.
# ✅ Predicts the class of a new flower measurement.
#
# This is a multi-class classification problem solved using a feedforward neural network (DNN) with softmax output.