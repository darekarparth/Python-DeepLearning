'''Given the following fragment code

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

row2 = np.array([[5.1, 3.5, 1.4, 0.2]])

yhat = model.predict([row2])
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))

Use Functional model to develop a multiclass classification model using Iris dataset,
display
a) Training accuracy
b) Prediction results
'''

import keras
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np
# keras and tensorflow.keras: Used for building and training the neural network.
# numpy: Handles numerical operations.
# pandas: Loads the dataset.
# sklearn.model_selection.train_test_split: Splits the dataset into training and testing.
# sklearn.preprocessing.LabelEncoder: Converts categorical labels into numeric values.


path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)
# The Iris dataset is loaded from an online CSV file.
# This dataset contains 150 samples of Iris flowers with:
# 4 numerical features (sepal length, sepal width, petal length, petal width).
# 1 categorical target label (Iris-setosa, Iris-versicolor, Iris-virginica)


# split into input and output columns
X, y = df.values[:, :-1], df.values[:, -1]
# X: Stores the 4 numerical features (sepal length, width, petal length, width).
# y: Stores the flower species names (categorical).
# LabelEncoder(): Converts categorical names (e.g., "Iris-setosa") into numerical labels (0, 1, 2)


# ensure all data are floating point values
X = X.astype('float32')
# encode strings to integer
y = LabelEncoder().fit_transform(y)
# LabelEncoder(): Converts categorical names (e.g., "Iris-setosa") into numerical labels (0, 1, 2)


# split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# The dataset is split into 67% training data and 33% testing data.
# print() shows the shape of training and testing sets.

# determine the number of input features
n_features = X_train.shape[1]
# Determines the number of input features (4) from the dataset.


# define model
# define model by Functional API
x_in = Input(shape=(n_features,))
x2 = Dense(10, activation='relu', kernel_initializer='he_normal')(x_in)
x3 = Dense(10, activation='relu', kernel_initializer='he_normal')(x2)
x_out = Dense(3, activation='softmax')(x3)
# Input(shape=(4,)): Defines the input layer with 4 neurons (for 4 features).
# Dense(10, activation='relu', kernel_initializer='he_normal'): Two hidden layers with 10 neurons each and ReLU activation.
# Dense(3, activation='softmax'): Output layer with 3 neurons (for 3 flower classes), using softmax activation for multi-class classification

# Creating the model
model = keras.Model(inputs=x_in, outputs=x_out)
# Defines the neural network model using the Functional API.

# compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# optimizer='adam': Uses Adam optimizer for efficient training.
# loss='sparse_categorical_crossentropy': Used for multi-class classification where labels are integers (0,1,2).
# metrics=['accuracy']: Evaluates model performance using accuracy.

# fit/Train the model
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)
# epochs=150: Trains the model for 150 iterations over the dataset.
# batch_size=32: Updates weights after 32 samples in each iteration.
# verbose=0: Hides training output.

# evaluate the model
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print('Test Accuracy: %.3f' % acc)
# Evaluates the model’s performance on the test set.
# Prints the accuracy of the model.

# make a prediction
row1 = [5.1, 3.5, 1.4, 0.2]
row2 = np.array([[5.1, 3.5, 1.4, 0.2]])
yhat = model.predict([row2])
print('Predicted: %s (class=%d)' % (yhat, argmax(yhat)))
# Creates a test sample (row1) with sepal & petal measurements.
# Converts it to a NumPy array (row2).
# model.predict([row2]): Predicts probabilities for each class.
# argmax(yhat): Returns the class with the highest probability.

# Example Output
# Test Accuracy: 0.960
# Predicted: [[9.9e-01 1.0e-02 2.3e-04]] (class=0)

# ✅ Loads and preprocesses the Iris dataset
# ✅ Splits it into training and test sets
# ✅ Builds a feedforward neural network using the Functional API
# ✅ Trains the model using the Adam optimizer
# ✅ Evaluates accuracy and predicts new inputs