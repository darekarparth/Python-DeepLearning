
'''1. Write a code in Python to create a two folder handout. The following code fragment is
used to generate a sequence of integers from 0 to 19,
a) Use the ‘KFold’ function in Scikit-Learn to create two folders for training and test
b) Print out the training and test data in these folders'''

from random import seed
from random import choice
import numpy as np
from sklearn.model_selection import KFold

# seed random number generator
seed(1)
# prepare a sequence
sequence = [i for i in range(20)]
print(sequence)

X = np.array(sequence)
kf = KFold(n_splits=2, random_state=None, shuffle=False)
# kf = KFold(n_splits=2, random_state=True, shuffle=True)

for train, test in kf.split(X):
    print('train: %s, test: %s' % (X[train], X[test]))