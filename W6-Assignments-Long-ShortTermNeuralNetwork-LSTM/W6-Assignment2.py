'''
Given one input sample with 3 time steps and one feature observed at each time step: t1
= 0.1, t2 = 0.2, t3 = 0.3. Make the following code working and returns:
a. A LSTM hidden state output for the last time step.
b. A LSTM hidden state output for the last time step (again).
c. A LSTM cell state for the last time step.
'''

from tensorflow.keras import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from numpy import array
# define model
inputs1 = Input(shape=(3, 1))
# lstm1, state_h, state_c = LSTM(1, )(inputs1) - Incorrect code
# model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c]) -  - Incorrect code
lstm1, state_h, state_c = LSTM(1, return_state=True)(inputs1)
model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
# inputs1 = Input(shape=(3,1))
# Defines the input shape as (3, 1), meaning:
# 3 time steps
# 1 feature per time step
# LSTM(1, return_state=True)
# Creates an LSTM layer with 1 unit.
# The return_state=True argument ensures that we return:
# lstm1 → The hidden state (output) of the last time step.
# state_h → The hidden state at the last time step.
# state_c → The cell state at the last time step.
# model = Model(inputs=inputs1, outputs=[lstm1, state_h, state_c])
# Creates a Keras model that takes inputs1 as input and produces three outputs.

# define input data
data = array([0.1, 0.2, 0.3]).reshape((1,3,1))
# Creates a NumPy array with values [0.1, 0.2, 0.3], representing one input sample with three time steps.
# Reshaped into the required LSTM input shape (1, 3, 1), where:
# 1 → Number of samples (batch size)
# 3 → Time steps
# 1 → Features per time step

# make and show prediction
print(model.predict(data))
# Passes data to the trained LSTM model.
# Outputs three values:
# Last time step’s LSTM output (lstm1)
# Last time step’s hidden state (state_h)
# Last time step’s cell state (state_c)

# Expected Output
# The output consists of three values:
#
# LSTM Output at Last Time Step (lstm1)
# This is the hidden state at the last time step.
# Hidden State at Last Time Step (state_h)
# Same as lstm1 (because the LSTM output at the last time step is just the hidden state).
# Cell State at Last Time Step (state_c)
# This is the internal memory state of the LSTM.