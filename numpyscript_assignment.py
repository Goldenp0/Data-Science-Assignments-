import numpy as np

print(np.__version__)
print(np.show_config())

#Create a null vector of size 10
null_vector = np.zeros(10)
print(null_vector)

#How to find the memory size of any array
print(null_vector.size * null_vector.itemsize)

#How to get the documentation of the numpy add function from the command line?
print(np.info(np.add))

# Create a null vector of size 10 but the fifth value which is 1
null_vector[4] = 1
print(null_vector)

# Create a vector with values ranging from 10 to 49 
vector_10_to_49 = np.arange(10, 50)
print(vector_10_to_49)

# Reverse a vector (first element becomes last)
reversed_vector = np.flip(vector_10_to_49)
print(reversed_vector)

# Create a 3x3 matrix with values ranging from 0 to 8
matrix_3x3 = np.arange(9).reshape(3, 3)
print(matrix_3x3)

# Find indices of non-zero elements from [1,2,0,0,4,0]
nonzero_indices = np.nonzero([1, 2, 0, 0, 4, 0])
print(nonzero_indices)

# Create a 3x3 identity matrix
identity_matrix = np.eye(3)
print(identity_matrix)

# Create a 3x3x3 array with random values
random_matrix = np.random.random((3, 3, 3))
print(random_matrix)

# Create a 10x10 array with random values and find the minimum and maximum values
random_10x10 = np.random.random((10, 10))
print("Minimum:", np.min(random_10x10))
print("Maximum:", np.max(random_10x10))

# Create a random vector of size 30 and find the mean value
random_vector_30 = np.random.random(30)
print("Mean:", np.mean(random_vector_30))

#  Create a 2d array with 1 on the border and 0 inside
border_array = np.ones((10, 10))
border_array[1:-1, 1:-1] = 0
print(border_array)

# How to add a border (filled with 0's) around an existing array
array_to_border = np.ones((5, 5))
array_with_border = np.pad(array_to_border, pad_width=1, mode='constant', constant_values=0)
print(array_with_border)

# What is the result of the following expression
expression_results = [
    0 * np.nan,
    np.nan == np.nan,
    np.inf > np.nan,
    np.nan - np.nan,
    np.nan in set([np.nan]),
    0.3 == 3 * 0.1
]
print(expression_results)

# Create a 5x5 matrix with values 1,2,3,4 just below the diagonal 
diagonal_matrix = np.diag([1, 2, 3, 4], k=-1)
print(diagonal_matrix)

# Create a 8x8 matrix and fill it with a checkerboard pattern
checkerboard_matrix = np.zeros((8, 8), dtype=int)
checkerboard_matrix[1::2, ::2] = 1
checkerboard_matrix[::2, 1::2] = 1
print(checkerboard_matrix)

# Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element
shape = (6, 7, 8)
index_100 = np.unravel_index(100, shape)
print(index_100)

# Create a checkerboard 8x8 matrix using the tile function 
checkerboard_tile = np.tile(np.array([[1, 0], [0, 1]]), (4, 4))
print(checkerboard_tile)

# Normalize a 5x5 random matrix
normalized_matrix = np.random.random((5, 5))
normalized_matrix = (normalized_matrix - np.mean(normalized_matrix)) / np.std(normalized_matrix)
print(normalized_matrix)

# Create a custom dtype that describes a color as four unsigned bytes (RGBA) 
color_dtype = np.dtype([('R', np.ubyte, 1),
                        ('G', np.ubyte, 1),
                        ('B', np.ubyte, 1),
                        ('A', np.ubyte, 1)])
color_array = np.zeros((5, 5), dtype=color_dtype)
print(color_array)

# Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
matrix_A = np.random.random((5, 3))
matrix_B = np.random.random((3, 2))
matrix_product = np.dot(matrix_A, matrix_B)
print(matrix_product)

# Given a 1D array, negate all elements which are between 3 and 8, in place
array_1d = np.arange(11)
array_1d[(array_1d >= 3) & (array_1d <= 8)] *= -1
print(array_1d)

# What is the output of the following script? 
print(sum(range(5), -1))
from numpy import *
print(sum(range(5), -1))

