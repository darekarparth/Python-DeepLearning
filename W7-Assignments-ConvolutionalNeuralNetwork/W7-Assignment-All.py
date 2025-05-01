#Assignment - 1
# Write a code in python, use a random way to generate a two dimensional matrix for grid
# processing. The matrix is composed of 10 rows and 10 columns, where the entries of the
# matrix is either 0 or 1.
import numpy as np

# Set random seed for reproducibility (optional)
np.random.seed(42)

# Generate 10x10 matrix with random 0s and 1s
grid_matrix = np.random.randint(0, 2, size=(10, 10))

# Display the generated matrix
print("Generated 10x10 Matrix:")
print(grid_matrix)

##Assignment - 2
#2. Below is a formula that is used to compute the convolution between two same
# dimension matrices:
#  [a b] * [w x]
#  [e f]   [𝑦 𝑧]  =[𝑎𝑤+𝑏𝑥+𝑒𝑦+𝑓𝑧]
# Given an input A = [[2, 3], [3, 4]], B = [[1, 2], [3, 3]], write a Python code to conduct their
# convolution calculation.

import numpy as np
A = np.array([[2,3],[3,4]])
B = np.array([[1,2],[3,3]])
# A is your input matrix (like an image).
# B is your kernel/filter, used to scan over A and compute weighted sums
print(B.dot(A))
print(A.dot(B))

def convolution2D(input_matrix, kernel):
    kernel = np.flipud(np.fliplr(kernel)) #This flips the kernel both vertically and horizontally.
    # Required in true mathematical convolution (as opposed to cross-correlation)
    # np.flipud: flips up-down
    # np.fliplr: flips left-right
    #[[3, 3],  Keral B fliped
    #[2, 1]]
    output_shape = (input_matrix.shape[0] - kernel.shape[0] + 1,
                    input_matrix.shape[1] - kernel.shape[1] + 1)
    #Calculates the dimensions of the output matrix
    #Since it's valid convolution:output size=input size−kernel size+1
    #In your case: 2 - 2 + 1 = 1, so output is 1×1.
    output = np.zeros(output_shape) #Creates an output matrix filled with zeros to store results.

    for i in range(output_shape[0]): #Loops over each position in the input where the kernel can be applied.
        for j in range(output_shape[1]):
            sub_matrix = input_matrix[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i, j] = np.sum(sub_matrix * kernel)
            #sub_matrix: part of the input that aligns with the kernel.
            #sub_matrix * kernel: element-wise multiplication.
            #np.sum(...): sums the result to get a single scalar (this goes in the output)

    return output

result = convolution2D(A,B)
print("Convolution Result:", result)
#This flips the kernel both vertically and horizontally.
#This is mathematical convolution, which involves flipping the kernel before computing the sum of element-wise multiplications.
#So, you’re applying A * flipped(B), not A * B.

##Assignment - 3
#3. Below is a formula that is used to compute the convolution between two same
# dimension matrices:
#  [a b] * [w x]
#  [e f]   [𝑦 𝑧]  =[𝑎𝑤+𝑏𝑥+𝑒𝑦+𝑓𝑧]�
#Given an input A = [[2, 3], [3, 4]], B = [[1, 2], [3, 3]], write a Python code to conduct their
#convolution calculation using the dot product function of the array.

A = np.array([[2,3],[3,4]])
B = np.array([[1,2],[3,3]])
result = np.dot(A.flatten(), B.flatten())
print("Convolution Result using dot product:", result)
result = np.sum(A * B)
print("Same result using element-wise multiplication and sum:", result)
#This does not flip the kernel, it directly flattens both matrices and calculates the dot product.
#It is equivalent to summing A[i][j] * B[i][j] element-wise.

#🧠 So, why the difference?
#Convolution: Uses flipped kernel.
#Dot product: Uses the kernel as-is, no flipping.
#To make the dot product simulate convolution:
flipped_B = np.flipud(np.fliplr(B))
result = np.dot(A.flatten(), flipped_B.flatten())
print("Dot product simulating convolution:", result)