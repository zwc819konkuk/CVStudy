import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))
print("=====================")
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print("=====================")
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])
for row in X:
    print(row)
X = X.flatten()
print(X)
print(X[np.array([0, 2, 4])])
print(X > 15)
print(X[X > 15])
print("=====================")
x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos")  # 用虚线绘制
plt.xlabel("x")  # x轴标签
plt.ylabel("y")  # y轴标签
plt.title('sin & cos')  # 标题
plt.legend()
plt.show()
print("=====================")
img = imread('../dataset/1.jpeg')  # 读入图像（设定合适的路径！）
plt.imshow(img)
plt.show()
