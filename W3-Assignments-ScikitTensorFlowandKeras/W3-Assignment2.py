# Given
# from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_regression
# from sklearn.model_selection import train_test_split
# X, y = make_regression((n_samples=200, random_state=1))
#
#
# Develop a classification model using Multi Perceptron class MLPRegression in
# Scikit_Learn and display prediction results, and scores, loss function, activation function


from sklearn.neural_network import MLPRegressor #A multi-layer perceptron (MLP) for regression.
from sklearn.datasets import make_regression #Generates a synthetic regression dataset.
from sklearn.model_selection import train_test_split #Splits data into training and testing sets

X, y = make_regression(n_samples=200, random_state=1) #Generates 200 samples with random input features (X) and corresponding target values (y)
# random_state=1 ensures reproducibility.

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# Splits the data into:
# Training set (X_train, y_train) – used to train the model.
# Testing set (X_test, y_test) – used to evaluate model performance.
# random_state=1 ensures consistency in data splitting.


# print(X)
# print(y)

regr = MLPRegressor(random_state=1, max_iter=500)
regr.fit(X_train, y_train)
# MLPRegressor(random_state=1, max_iter=500):
# A neural network model with default parameters.
# max_iter=500: Trains for 500 iterations at most.
# regr.fit(X_train, y_train): Trains the model using the training data.


print(regr.loss_)
# This line should be print(regr.loss_) (corrected version).
# regr.loss_ gives the final loss after training, indicating how well the model fits the data.

print(regr.activation)
# This prints the activation function used in the hidden layers (default: 'relu')

print(regr.predict(X_test[:2]))
# Predicts values for the first two test samples.

print(regr.score(X_test, y_test))
# Computes the R² score (coefficient of determination):
# A value closer to 1 means a good fit.
# A value near 0 or negative means a poor fit.