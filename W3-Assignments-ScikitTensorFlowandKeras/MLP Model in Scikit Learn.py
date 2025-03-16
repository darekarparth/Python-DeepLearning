# A Multilayer Perceptron model (MLP) is a standard fully
# connected neural network model.
# Scikit-Learn implements the MLP algorithm with a Class
# called MLPClassifier that trains using Backpropagation
# To invoke this class to create a MLP model, we need to
# initiate the class with appropriate parameters
# MLP trains on two arrays: array X of size (n_samples,
# n_features), which holds the training samples represented as
# floating point feature vectors; and array y of size (n_samples,),
# which holds the target values (class labels) for the training
# samples:

from sklearn.neural_network import MLPClassifier # This imports MLPClassifier, which is a type of artificial neural network (ANN) used for classification tasks.
X = [[0., 0.], [1., 1.]] #
y = [0, 1]
#X is the input feature set (each row represents an input data point).
# y is the target label (or output class) corresponding to each row in X.
# Here, we have two training examples:
# Input [0., 0.] corresponds to class 0
# Input [1., 1.] corresponds to class 1

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
# MLPClassifier is instantiated with the following parameters:
# solver='lbfgs': The optimization algorithm used to minimize the loss function (good for small datasets).
# alpha=1e-5: A regularization parameter that helps prevent overfitting.
# hidden_layer_sizes=(5, 2): Defines the structure of the neural network:
# The first hidden layer has 5 neurons.
# The second hidden layer has 2 neurons.
# random_state=1: Ensures reproducibility by fixing the random seed.

clf.fit(X, y)
# This trains the neural network using the input data X and target labels y.
# The network learns to classify inputs based on the given data.

print(clf.predict([[2., 2.], [-1., -2.]]))
print(clf.predict_proba([[2., 2.], [1., 2.]]) )
# The trained model is used to predict the class labels for two new inputs:
# [2., 2.]
# [-1., -2.]
# The predict() function returns the predicted class labels