import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.datasets import mnist
import sys, os

sys.path.append(os.pardir)
from PIL import Image

# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.8
#     tmp = x1 * w1 + x2 * w2
#     if tmp <= theta:
#         return 0
#     elif tmp > theta:
#         return 1
#
#
# print(AND(0, 0))
# print(AND(1, 0))
# print(AND(0, 1))
# print(AND(1, 1))

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 仅权重和偏置与AND不同！
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # 仅权重和偏置与AND不同！
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)  # 指定y轴的范围


# plt.show()


def relu(x):
    return np.maximum(0, x)


A = np.array([1, 2, 3, 4])
print(A)
print(A.ndim)

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(B.ndim)
print(B.shape)
print(B.shape[0])

A = np.array([[1, 2, 3], [3, 4, 5]])
print(A.shape)
B = np.array([[5, 6], [7, 8], [9, 10]])
print(B.shape)
print(np.dot(A, B))

print("===================================")
X = np.array([1, 2])
print(X.shape)
W = np.array([[1, 3, 5], [2, 4, 6]])
print(W.shape)
print(np.dot(X, W))
print("===================================")
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
print(W1.shape)  # (2, 3)
print(X.shape)  # (2,)
print(B1.shape)  # (3,)
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print(A1)
print(Z1)
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
print(Z1.shape)  # (3,)
print(W2.shape)  # (3, 2)
print(B2.shape)  # (2,)
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)


def identity_function(x):
    return x


W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)  # 或者Y = A3
print("=======================")


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y


network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)  # [ 0.31682708 0.69627909]


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)  # 溢出对策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


print("==========================================")
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)  # (60000,)
print(X_test.shape)  # (10000, 784)
print(y_test.shape)  # (10000,)


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


img = X_train[0]
label = y_train[0]
print(label)  # 5
print(img.shape)  # (784,)
img = img.reshape(28, 28)  # 把图像的形状变成原来的尺寸
print(img.shape)  # (28, 28)
img_show(img)
