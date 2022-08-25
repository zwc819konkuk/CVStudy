import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np

#Sigmoid
x = np.linspace(-10,10,1000)
y = tf.nn.sigmoid(x)
# plt.plot(x,y)
# plt.grid()
# plt.show()

#tanh
x = np.linspace(-10,10,1000)
y = tf.nn.tanh(x)
# plt.plot(x,y)
# plt.grid()
# plt.show()

#relu
x = np.linspace(-10,10,1000)
y = tf.nn.relu(x)
# plt.plot(x,y)
# plt.grid()
# plt.show()

#leakyrelu
x = np.linspace(-10,10,1000)
y = tf.nn.leaky_relu(x)
# plt.plot(x,y)
# plt.grid()
# plt.show()

#softmax
x = tf.constant([0.2,0.02,0.15,1.3,0.5,0.06,1.1,0.05,3.75])
y = tf.nn.softmax(x)
print(y)

#参数初始化

#xavier正态分布
initializer = tf.keras.initializers.glorot_normal()
values = initializer(shape=(9,1))
# print(values)
#xavier标准化均匀分布
initializer = tf.keras.initializers.glorot_uniform()
values = initializer(shape=(9,1))
# print(values)

#he正态初始化
initializer = tf.keras.initializers.he_normal()
values = initializer(shape=(9,1))
print(values)
#he标准初始化均匀分布取0到1之间的数据的概率相等
initializer = tf.keras.initializers.he_uniform()
values = initializer(shape=(9,1))
print(values)