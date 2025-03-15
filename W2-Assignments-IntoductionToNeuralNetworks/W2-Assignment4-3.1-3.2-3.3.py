# Let's start with the specific case of the three layer neural network below
# A feed-forward example
import numpy as np
w1 = np.array([[0.2, 0.2, 0.2],[0.4, 0.4, 0.4],[0.6, 0.6, 0.6]])
w2 = np.zeros((1,3))
w2[0, :] = np.array([0.5, 0.5, 0.5])
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])
def f(x):
    return 1/(1 + np.exp(-x))

# Our first attempt at a feed-forward function
# Below is a simple way of calculating the output of the neural network, using nested loops
# in python. We'll look at more efficient ways of calculating the output shortly.
# This function takes as input the number of layers in the neural network, the x input
# array/vector, then Python tuples or lists of the weights and bias weights of the network,
# with each element in the tuple/list representing a layer in the network. In other words,
# the inputs are setup in the following:

def simple_looped_nn_calc(n_layers, x, w, b):
    global h
    for l in range(n_layers-1):
        if l == 0:
            node_in = x
        else:
            node_in = h
        h = np.zeros((w[l].shape[0],))
        for i in range((w[l].shape[0])):
            f_sum = 0
            for j in range((w[l].shape[1])):
                f_sum += w[l][i][j]*node_in[j]
            f_sum += b[l][i]
            h[i] = f(f_sum)
    return h

w = [w1, w2]
b = [b1, b2]

x = [1.5, 2.0, 3.0]
print(simple_looped_nn_calc(3, x, w, b))
