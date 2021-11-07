import numpy as np
from scipy.io import loadmat
import sys
mat = loadmat('mnist_all.mat')
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

print(x[3:])
print(x[:3])
