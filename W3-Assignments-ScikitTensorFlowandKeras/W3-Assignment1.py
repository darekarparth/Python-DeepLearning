# Given
# from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# X, y = make_classification((n_samples=100, random_state=1))

# Develop a classification model using Multi Perceptron class MLPClassifier in Scikit_Learn
# and display the probabilities P(y|x), prediction results, and scores, loss function,
# activation function

from sklearn.neural_network import MLPClassifier # Implements a neural network for classification.
from sklearn.datasets import make_classification #Generates a synthetic classification dataset.
from sklearn.model_selection import train_test_split #Splits data into training and testing sets.

# load data from Scikit-Learn
X, y = make_classification(n_samples=100, random_state=1)
# make_classification generates a binary classification dataset.
# n_samples=100: The dataset contains 100 samples (data points).
# random_state=1: Ensures the data is the same every time the program runs.


# split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
# Splits X (features) and y (labels) into training and testing sets.
# stratify=y: Ensures the class distribution in the train/test sets is similar to the original dataset.
# random_state=1: Ensures reproducibility.


# constructe a classification model
clf = MLPClassifier(random_state=1, max_iter=300)
# MLPClassifier creates a neural network model.
# random_state=1: Ensures the same weight initialization each time.
# max_iter=300: Limits training to 300 iterations (epochs).


# training
clf.fit(X_train, y_train) #Trains the neural network using the training data (X_train, y_train).

# display the probabilities P(y|x)
print(clf.predict_proba(X_test[:1])) #predict_proba(X_test[:1]) gives the probability of each class for the first test sample
# Output: [[0.3, 0.7]]
# The model predicts that the first test sample belongs to class 1 with a 70% probability and class 0 with 30% probability.


# predict
print(clf.predict(X_test[:5, :])) #Predicts the class labels for the first 5 test samples
# Output: [1 0 1 0 1]
# The model classifies the first sample as class 1, second as class 0, and so on.

# Evaluate
print(clf.score(X_test, y_test)) #score(X_test, y_test) calculates the accuracy (correct predictions / total predictions).
# Output: 0.85
# This means the model is 85% accurate on the test set.


# The program generates a synthetic dataset for binary classification.
# The dataset is split into training (75%) and testing (25%).
# A Multilayer Perceptron (MLP) is created and trained on the data.
# It predicts class probabilities and labels for test data.
# The model’s accuracy is displayed