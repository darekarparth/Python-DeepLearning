import matplotlib.pylab as plt
import numpy as np

def sigmoid(z):
    """Compute sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def neural_network_output(x, w, b):
    """Compute output of the perceptron using given inputs, weights, and bias."""
    z = np.dot(x, w) + b  # Linear combination of inputs and weights
    output = sigmoid(z)   # Applying sigmoid activation
    return output

# Example values (replace these with actual values from the question)
x = np.array([0.5, 0.3, 0.2])  # Inputs
w = np.array([0.8, -0.5, 0.3]) # Weights
b = 0.1  # Bias

# Compute the output
output = neural_network_output(x, w, b)
print(f"Neural Network Output: {output:.4f}")
plt.plot(x, w)
plt.xlabel('x')
plt.ylabel('h_w(x)')
plt.show()