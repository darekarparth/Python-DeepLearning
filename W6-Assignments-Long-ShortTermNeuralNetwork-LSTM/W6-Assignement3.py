from sklearn.datasets import make_classification
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from matplotlib import pyplot
# # make_classification (from sklearn.datasets) creates a synthetic dataset for classification tasks.
# # Sequential is a linear stack of layers in Keras.
# # Dense defines fully connected layers in the neural network.
# # SGD is the Stochastic Gradient Descent optimizer.
# # matplotlib is used to plot learning curves.
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# create the dataset
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
# make_classification() generates a dataset for binary classification.
# n_samples=1000 → Creates 1000 samples (rows).
# n_classes=2 → Binary classification problem (labels: 0 or 1).
# random_state=1 → Ensures reproducibility.
# X contains the input features, and y contains the class labels.


# determine the number of input features
n_features = X.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(1, activation='sigmoid'))
# n_features = X.shape[1] → Retrieves the number of features from X.
# Sequential() → Initializes a sequential model (layer by layer).
# First layer:
# Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,))
# 10 neurons with ReLU activation.
# he_normal → Initializes weights using He initialization (good for ReLU).
# input_shape=(n_features,) → Specifies the input size.
# Output layer:
# Dense(1, activation='sigmoid')
# 1 neuron with Sigmoid activation (output is a probability between 0 and 1 for binary classification).

# compile the model
sgd = SGD(learning_rate=0.001, momentum=0.8)
model.compile(optimizer=sgd, loss='binary_crossentropy')
# SGD(learning_rate=0.001, momentum=0.8):
# Stochastic Gradient Descent optimizer with momentum.
# learning_rate=0.001 → Small learning rate for stable training.
# momentum=0.8 → Helps accelerate gradient updates.
# binary_crossentropy loss function is used for binary classification.

# fit the model
history = model.fit(X, y, epochs=100, batch_size=32, verbose=0, validation_split=0.3)
# epochs=100 → Trains the model for 100 epochs.
# batch_size=32 → Processes data in mini-batches of 32.
# verbose=0 → No output during training.
# validation_split=0.3 → 30% of the data is used for validation.
# history stores the loss values for both training and validation.


# plot learning curves
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Cross Entropy')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()

# This program:
#
# Generates synthetic data (make_classification).
# Defines a simple neural network with:
# One hidden layer (10 neurons, ReLU).
# One output layer (Sigmoid activation).
# Uses SGD optimizer with momentum.
# Trains the model with 100 epochs.
# Plots training & validation loss to monitor learning.