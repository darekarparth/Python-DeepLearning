#Understand matrix multiplication and implement the calculation
import numpy as np

# Define a 3x3 matrix
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# Define a 3x1 matrix (column vector)
B = np.array([[1],
              [2],
              [3]])

# Multiply matrices
C = np.dot(A, B)

print(C)
