'''Given the following fragment code

from sklearn.datasets import make_classification
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np

# create the dataset
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2,
                           random_state=1)
# determine the number of input features
n_features = X.shape[1]
# define model

.
.
.

Follow 5 steps introduced in Lecture 3 and use the Sequential model to develop a binary
classification model with the following tasks:

a) Construct 1 hidden layer, consisting of 10 neurons with “relu” activation function
b) Add the final layer with “sigmoid” activation function (think about why?)
c) Add optimizer with “SGD” and loss function with “binary_crossentropy”
d) Save the trained model as “model”
e) Load the model.Given an array[1.91518414, 1.14995454, -1.52847073, 0.79430654], use the loaded model to predict the result
'''

from sklearn.datasets import make_classification
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np
# make_classification (from sklearn): Generates a synthetic dataset for classification.
# Sequential (from tensorflow.keras): A linear stack of layers to define the neural network.
# Dense (from tensorflow.keras.layers): A fully connected (dense) layer in the neural network.
# SGD (from tensorflow.keras.optimizers): Stochastic Gradient Descent optimizer.
# numpy (as np): For handling arrays and numerical computations.

# create the dataset
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=1)
# Generates a dataset with:
# 1000 samples.
# 4 features (input variables).
# 2 classes (binary classification: 0 or 1).
# random_state=1: Ensures reproducibility.
# determine the number of input features


n_features = X.shape[1] # Determines the number of input features (4 in this case)
# define model
model = Sequential()
# n_features stores the number of input features (4).
# Sequential() initializes a sequential neural network.

model.add(Dense(10, activation='relu', kernel_initializer='he_normal',input_shape=(n_features,)))
# Adds a Dense (fully connected) layer with:
# 10 neurons
# ReLU activation function (helps with non-linearity)
# He-normal initialization (good for ReLU)
# Input shape = 4 features

model.add(Dense(1, activation='sigmoid'))
# Adds an output layer with:
# 1 neuron (since it's binary classification).
# Sigmoid activation (outputs probability between 0 and 1).

# compile the model
sgd = SGD(learning_rate=0.001, momentum=0.8)
model.compile(optimizer=sgd, loss='binary_crossentropy')
# SGD(learning_rate=0.001, momentum=0.8): Uses Stochastic Gradient Descent with:
# Learning rate 0.001 (small step size for optimization).
# Momentum 0.8 (helps accelerate learning).
# Binary cross-entropy loss (since it's binary classification).

# fit the model
model.fit(X, y, epochs=100, batch_size=32, verbose=0, validation_split=0.3)
# Trains the model on X (inputs) and y (labels).
# Epochs = 100: The dataset is passed through the model 100 times.
# Batch size = 32: Trains with mini-batches of 32 samples.
# Validation split = 0.3: 30% of data is used for validation/Testing.
# verbose=0: Hides training output.

# save model to file
model.save('model.h5')
# Saves the trained model to a file named "model.h5".

from tensorflow.keras.models import load_model
# load the model from file
model = load_model('model.h5')
# Loads the saved model back into memory.

# make a prediction
row = [1.91518414, 1.14995454, -1.52847073, 0.79430654]
row2 = np.array([[1.91518414, 1.14995454, -1.52847073, 0.79430654]])
yhat = model.predict([row2])
print('Predicted: %.3f' % yhat[0])

# row2 is a new data point with 4 features.
# It is converted into a NumPy array (np.array) to match input shape.
# model.predict([row2]): Predicts the probability of class 1 (since sigmoid outputs probabilities).
# Prints the predicted probability rounded to 3 decimal places.

# Expected Output
# Predicted: 0.XXX

# A probability close to 1.000 → Strongly classified as class 1.
# A probability close to 0.000 → Strongly classified as class 0.

# ✅ Creates a synthetic dataset for classification.
# ✅ Builds a neural network with one hidden layer (10 neurons).
# ✅ Uses ReLU and Sigmoid activations for hidden and output layers.
# ✅ Trains the model using SGD optimizer with binary cross-entropy loss.
# ✅ Saves and loads the trained model from a file.
# ✅ Makes a prediction on a new data point

