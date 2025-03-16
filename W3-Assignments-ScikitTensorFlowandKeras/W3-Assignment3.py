# Given
# from numpy import argmax
# from pandas import read_csv
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# import numpy as np
#
# # load the dataset
# path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
# df = read_csv(path, header=None)
# # split into input and output columns
# X, y = df.values[:, :-1], df.values[:, -1]
# # ensure all data are floating point values
# X = X.astype('float32')
# # encode strings to integer
# y = LabelEncoder().fit_transform(y)
# # split into train and test datasets
# X_train, X_test, y_train, y_test =
# .
# .
# .
#
# row2 = np.array([[5.1, 3.5, 1.4, 0.2]])
#
# yhat = model.predict([row2])
# print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

# Use Sequential model to develop a multiclass classification model using Iris dataset,
# display
# a) Training accuracy
# b) Prediction results

from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
# numpy: Used for numerical operations.
# pandas: Used for loading the dataset.
# scikit-learn:
# train_test_split: Splits the dataset into training and testing sets.
# LabelEncoder: Converts categorical labels into numerical form.
# tensorflow.keras:
# Sequential: Defines the model structure.
# Dense: Adds fully connected layers to the neural network


# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)
# Loads the Iris dataset from an online source.
# The dataset consists of 150 samples (rows), each with:
# 4 numerical features (sepal length, sepal width, petal length, petal width).
# 1 categorical label (species name: Setosa, Versicolor, or Virginica).


# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# X: Extracts the first 4 columns (features).
# y: Extracts the last column (species labels).


# ensure all data are floating point values
X = X.astype('float32')
# Converts the feature values to float32 for compatibility with TensorFlow.


# encode strings to integer
y = LabelEncoder().fit_transform(y)
# Converts categorical class labels (Setosa, Versicolor, Virginica) into numerical labels:
# Setosa → 0
# Versicolor → 1
# Virginica → 2



# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Splits the dataset into:
# 67% training data
# 33% testing data

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Prints the shapes of training and testing datasets.

# determine the number of input features
n_features = X_train.shape[1]
# Determines the number of input features (which is 4 for the Iris dataset).

# define model
model = Sequential() #Creates a sequential neural network model.
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
# First hidden layer:
# 10 neurons
# ReLU activation (Rectified Linear Unit) → Helps in non-linearity.
# He-normal initializer → Helps in proper weight initialization.
# Input shape is (4,) → Since we have 4 input features.

model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
# Second hidden layer:
# 8 neurons
# ReLU activation
# He-normal initialization

model.add(Dense(3, activation='softmax'))
# Output layer:
# 3 neurons (since we have 3 classes or 3 Labels Setosa, Versicolor, Virginica).
# Softmax activation (converts outputs into probability distribution).

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# Optimizer: adam (Adaptive Moment Estimation) - Efficient for deep learning models.
# Loss Function: sparse_categorical_crossentropy
# Used when class labels are integers (0, 1, 2) instead of one-hot encoded vectors.
# Metric: accuracy - Measures how well the model classifies the samples.

# fit the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# Trains the model using:
# 150 epochs (iterations over the dataset).
# Batch size of 32 (processes 32 samples at a time).
# verbose=0 → Suppresses training output.

# evaluate the model/Test the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
# Evaluates the model on the test data.
# Prints the accuracy of the model on unseen data.

# make a prediction
row1 = [5.1, 3.5, 1.4, 0.2]
row2 = np.array([[5.1, 3.5, 1.4, 0.2]])
# row1 represents a new sample (similar to Setosa class).
# row2 converts the sample into a NumPy array for model compatibility

yhat = model.predict([row2])
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
# Uses model.predict() to predict class probabilities.
# argmax(yhat) gets the class index with the highest probability.


# The predicted output [0.98 0.01 0.01] shows:
# 98% probability of being Setosa (class 0).
# 1% probability of being Versicolor (class 1).
# 1% probability of being Virginica (class 2).
#
# Loads the Iris dataset.
# Prepares the data (encoding labels, splitting into train-test).
# Defines a neural network with:
# 2 hidden layers (10 and 8 neurons).
# ReLU activations.
# Softmax output layer (3 classes).
# Trains the model using Adam optimizer.
# Evaluates model accuracy on test data.
# Predicts class for a new sample.