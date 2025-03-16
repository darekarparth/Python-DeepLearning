# MLPClassifier supports multi-class classification by
# applying Softmax as the output function.
# in which a sample can belong to more than one class. For
# each class, the raw output passes through the logistic
# function. Values larger or equal to 0.5 are rounded to 1,
# otherwise to 0.
# For a predicted output of a sample, the indices where the
# value is 1 represents the assigned classes of that sample:

from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
clf.fit(X, y)
print(clf.predict([[1., 2.]]))
print(clf.predict([[0., 0.]]))