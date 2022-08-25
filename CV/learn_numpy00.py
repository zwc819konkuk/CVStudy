import matplotlib.pyplot as plt
import numpy as np

# x = np.array([[1, 2, 3], [4, 5, 6]])
# print(x)
#
# a = []
# for i in range(100000000):
#     a.append(random.random())
# m1 = time.time()
# sum1 = sum(a)
# m2 = time.time()
# print(m2 - m1)
# b = np.array(a)
#
# m3 = time.time()
# sum2 = np.sum(b)
# m4 = time.time()
# print(m4 - m3)
# 正态分布
# x1 = np.random.normal(1.75, 1, 100000000)
# plt.figure(figsize=(20, 8), dpi=100)
#
# plt.hist(x1, 1000)

# 均匀分布
x2  = np.random.uniform(-1,1,100000000)
plt.figure(figsize=(20, 8), dpi=100)
plt.hist(x2,1000)
plt.show()
