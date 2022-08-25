import numpy as np

a = np.array([1, 2, 3, 4])
# print(a)

a = np.random.randn(5, 1)
print(a)
print(a.shape)
print(a.T)
print(np.dot(a, a.T))
