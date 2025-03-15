import numpy as np
# NumPy is imported to handle array operations efficiently.

# The neural network has two layers, and we define the weight matrices and bias vectors accordingly.
#
# Weights (w1, w2)
w1 = np.array([[0.2, 0.2, 0.2],  # Weights for the first layer
               [0.4, 0.4, 0.4],
               [0.6, 0.6, 0.6]])

# w1 is a 3×3 weight matrix for the first layer.
# Each row represents the weights for a node in the first layer.

w2 = np.zeros((1,3))
w2[0, :] = np.array([0.5, 0.5, 0.5])  # Weights for the second layer
# w2 is a 1×3 weight matrix for the second layer.
# This means the output layer has 1 neuron, which takes 3 inputs from the first layer.

# Biases (b1, b2)
b1 = np.array([0.8, 0.8, 0.8])  # Bias for the first layer
b2 = np.array([0.2])  # Bias for the second layer
# b1 is a vector of 3 values, corresponding to the three neurons in the first layer.
# b2 is a single value, corresponding to the one neuron in the second layer

# Sigmoid Activation Function
def f(x):
    return 1/(1 + np.exp(-x))
# This is the sigmoid activation function.
# It takes an input x and returns a value between 0 and 1.


# Neural Network Calculation
def simple_looped_nn_calc(n_layers, x, w, b):
    global h  # h is a global variable to store intermediate outputs
    for l in range(n_layers - 1):  # Loop through each layer
        if l == 0:
            node_in = x  # First layer input is the given input 'x'
        else:
            node_in = h  # For subsequent layers, input is the previous layer's output

        h = np.zeros((w[l].shape[0],))  # Create an empty array to store neuron outputs

        for i in range((w[l].shape[0])):  # Loop through each neuron in the current layer
            f_sum = 0  # Initialize the sum for weighted input

            for j in range((w[l].shape[1])):  # Loop through each input of the neuron
                f_sum += w[l][i][j] * node_in[j]  # Compute weighted sum

            f_sum += b[l][i]  # Add bias to the sum

            h[i] = f(f_sum)  # Apply the sigmoid activation function

    return h  # Return the output of the last layer

# How It Works:
# The function loops through each layer and computes the neuron outputs.
# It initializes node_in as input 𝑥 for the first layer.
# Then, the sigmoid function is applied to get the activation.
# This process repeats for each layer until the final output is obtained.

#Running the Neural Network
w = [w1, w2]  # Store weights in a list
b = [b1, b2]  # Store biases in a list

x = [1.5, 2.0, 3.0]  # Input values
print(simple_looped_nn_calc(3, x, w, b))

# Inputs: 𝑥 =[1.5,2.0,3.0]
# Calls the function with 3 layers (n_layers=3).
# Final Output: The output of the last layer is printed.


