# The sigmoid function is a common activation function as follows
#
#
#
# Given an arrange [-5, 5] and step size 0.1, implement this activation function in Python
# (refer to Section 2.1)

import matplotlib.pylab as plt
import numpy as np
x=np.arange(-5,5,0.1)
f = 1/(1+np.exp(-x))
plt.plot(x,f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()