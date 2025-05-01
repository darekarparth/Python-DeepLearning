# Define a one-dimensional input that has eight elements all with the value of 0, with a
# two element bump in the middle with the values 1 below
# [0, 0, 0, 1, 1, 0, 0, 0]
# Define a single kernel below
# [0, 1, 0]
from numpy import asarray
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D

# Define 1D input: bump in the middle
data = asarray([0, 0, 0, 1, 1, 0, 0, 0])
print(data.shape)  # Output: (8,)
data = data.reshape(1, 8, 1)  # (batch_size, input_length, channels) #is used to reshape your 1D input data into a format that the Keras Conv1D layer expects.
print(data.shape)  # Output: (1, 8, 1) #Now it fits the Conv1D input format.
#is used to reshape your 1D input data into a format that the Keras Conv1D layer expects.
# Here's what each dimension means:
# Dimension	                 Value	           Meaning
# 1 (1st dim)	Batch size =   1	We are passing one sample (one signal)
# 8 (2nd dim)	Input length = 8	The signal has 8 time steps or elements
# 1 (3rd dim)	Channels =     1	Each time step has 1 feature/channel

# Create the model with 1 filter (kernel), size 3
model = Sequential()
model.add(Conv1D(filters=1, kernel_size=3, input_shape=(8, 1)))
#adds a 1D Convolutional Layer to your Keras Sequential model. Let’s break it down part by part:
# Parameter	              Value	                               Meaning
# filters=1	            1 filter	                        The layer will learn 1 kernel (filter) during training. This means the output will have 1 feature map.
# kernel_size=3	        3 units wide	                    The filter will look at 3 time steps at a time as it slides over the input (like a sliding window of size 3).
# input_shape=(8, 1)	input has 8 time steps, 1 feature	This defines the shape of the input data: a sequence of 8 values (length = 8), each with 1 feature/channel (like grayscale pixels or 1D amplitude values).
# Slides a 3-step kernel over that 8-length signal.
# Applies the kernel at each position (5 total positions: 8 - 3 + 1 = 6).
# For each position, it performs a dot product (sum of element-wise multiplications).
# Since filters=1, it produces 1 output per position, resulting in a 1D output of length 6:

# Define a handcrafted kernel: [0, 1, 0]
# This means we are detecting the middle value in a 3-element window
weights = [asarray([[[0]],[[1]],[[0]]]), asarray([0.0])]  # weight, bias
# is setting custom weights for a 1D convolutional layer in Keras.
#Kernel weights – the values in the sliding filter (what we usually call the "kernel")
# Bias – a scalar added to each output after convolution
# 1️⃣ asarray([[[0]], [[1]], [[0]]])
# This defines a 1D kernel of size 3, with weights: [0, 1, 0]
# Why 3D?
# Keras expects kernel weights in shape: (kernel_size, input_channels, output_filters)
# In your case:
# kernel_size = 3
# input_channels = 1 (1 feature per time step)
# output_filters = 1 (1 filter in the layer)
# So shape is: (3, 1, 1)
# You’re telling Keras:
# “Use a 1D filter of size 3 that only looks at the center value in the 3-step window (because only the middle weight is 1).”

# 2️⃣ asarray([0.0])
# This is the bias value for the filter.
# Since you have filters=1, you need one bias value.
# Setting it to 0.0 ensures it doesn’t affect the convolution output.

# store the weights in the model # Set the weights into the Conv1D layer
model.set_weights(weights)
# a) Use ‘model’ to get the stored weights and print them out
# b) Input [0, 0, 0, 1, 1, 0, 0, 0] into the model and make a prediction
# c) Print the predicted result
# d) Modify the sizes of input and Kernel to see what will be predicted


# --- a) Print the stored weights ---
print("Kernel weights and bias:")
for w in model.get_weights():
    print(w)

# --- b) Feed the input into the model and get predictions ---
predicted = model.predict(data)

# --- c) Print the predicted result ---
print("\nPrediction result:")
print(predicted.reshape(-1))  # flatten for easy viewing

#Output Explanation
#For the input: [0, 0, 0, 1, 1, 0, 0, 0]
#And the kernel: [0, 1, 0]

# Only the center value in each 3-element window is
# considered (because of the kernel).
# So the model highlights places where the middle element is 1.

#[0. 0. 1. 1. 0. 0.]
#Output shape: (1, 6, 1)
# This is because the convolution layer slides across with stride = 1,
# and it outputs a value equal to the center of the 3-element window
# (due to the kernel [0, 1, 0]).
