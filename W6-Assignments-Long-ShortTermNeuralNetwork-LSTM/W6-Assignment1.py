'''
Week 6: Practicals
1. Given one input sample with 3 time steps t1 = 0.1, t2 = 0.2, t3 = 0.3 and one feature
observed at each time step. Make the following code working and returns a sequence of
3 values, one hidden state output for each input time step for the single LSTM cell in the
layer.

'''

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from numpy import array
# Model is used to define the Keras functional API model.
# Input is used to define the input layer.
# LSTM is the Long Short-Term Memory layer for handling sequential data.
# array from NumPy is used for handling numerical data.

# define model/Defining the Input Layer
inputs1 = Input(shape=(3, 1))
# This defines an input layer that expects a 3D tensor with shape:
# 3 time steps (sequence length = 3)
# 1 feature per time step

#output a sngle sequence
#lstm1 = LSTM(1)(inputs1)
#output three sequences
# lstm1 = LSTM(1)(inputs1)
# The LSTM(1) layer processes the input sequence.
# It has 1 hidden unit (1 neuron).
# However, the issue here is that by default, LSTM in Keras only returns the last hidden state, not the full sequence.
# To return all 3 hidden states (one for each time step), we must modify it as follows:
lstm1 = LSTM(1, return_sequences=True)(inputs1) #Fixed Code
# return_sequences=True ensures that the LSTM returns a sequence of 3 outputs (one for each time step).

# Creating a model
model = Model(inputs=inputs1, outputs=lstm1)
# This defines a Keras Model that takes inputs1 as input and produces lstm1 as output.

# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# This creates a NumPy array with values [0.1, 0.2, 0.3].
# It is reshaped to match the required input shape:
# 1 sample
# 3 time steps
# 1 feature per time step

# make and show prediction
print(model.predict(data))
# The trained model takes data as input and predicts the hidden states at each time step.
# If return_sequences=True, it returns 3 hidden state outputs.