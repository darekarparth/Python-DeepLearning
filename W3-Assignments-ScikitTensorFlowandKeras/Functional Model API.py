'''Functional Model API (Advanced)
The functional API is more complex but is also more
flexible.
It involves explicitly connecting the output of one layer to
the input of another layer. Each connection is specified.
First, an input layer must be defined via the Input class, and
the shape of an input sample is specified. We must retain a
reference to the input layer when defining the model.

 # define the layers
 x_in = Input(shape=(8,))
Next, a fully connected layer can be connected to the input
by calling the layer and passing the input layer. This will
return a reference to the output connection in this new
layer.
'''

from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
# define the layers
x_in = Input(shape=(8,))
x = Dense(10)(x_in)
x_out = Dense(1)(x)
# define the model
model = Model(inputs=x_in, outputs=x_out)