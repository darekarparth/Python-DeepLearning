# The sequential model API is the simplest and is the API
# developed by Keras, it is referred to as ‘sequential’
# We initiate a model by the Sequential class
# and adds layers to
# the model one by one in a linear manner, from input to output.
# The example below defines a Sequential MLP model that accepts
# eight inputs, has one hidden layer with 10 nodes and then an
# output layer with one node to predict a numerical value.

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# Sequential: A linear stack of layers, where each layer feeds into the next.
# Dense: A fully connected (or dense) layer in a neural network.

model = Sequential()
# This initializes a sequential model, meaning layers will be added one after another.

model.add(Dense(10, input_shape=(8,)))
# Adding the First Layer
# Creates a fully connected layer with 10 neurons.
# input_shape=(8,) means the input data must have 8 features (or dimensions).
# Each neuron in this layer is connected to all 8 input features
# ✅ This is the first hidden layer.

model.add(Dense(1))
# Adding the Output Layer
# Adds another fully connected layer with 1 neuron.
# Since there's no activation function specified, the default is linear activation.
# This layer outputs a single value, making it useful for regression tasks (or binary classification if followed by a sigmoid activation).
# ✅ This is the output layer.

# The network consists of:
# An input layer (implicitly defined by input_shape=(8,)).
# A hidden layer with 10 neurons.
# An output layer with 1 neuron.

#############################################################################################


# The visible layer of the network is defined by the
# “input_shape” argument on the first layer. The model expects
# the input for one sample to be a vector of eight numbers.
# The sequential API is easy to use because we keep
# calling model.add() until we have added all of layers required.

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(100, input_shape=(8,)))  # Input layer with 8 features, 100 neurons
model.add(Dense(80)) # Hidden layer with 80 neurons
model.add(Dense(30)) # Hidden layer with 30 neurons
model.add(Dense(10)) # Hidden layer with 10 neurons
model.add(Dense(5))  # Hidden layer with 5 neurons
# These are hidden layers, meaning they process the input features before passing them to the next layer.
# Each layer applies a weight transformation and activation function

model.add(Dense(1))  # Output layer with 1 neuron

# What Can This Model Be Used For?
# Regression problems (like predicting house prices, stock prices, etc.).
# Binary classification (if you add a sigmoid activation in the last layer).
# Multi-class classification (if the last layer had multiple neurons with softmax activation).