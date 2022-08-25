import tensorflow as tf
import numpy as np

# 创建0维张量
print(tf.constant(3))

# 创建1维张量
print(tf.constant([1.0, 2.0, 3.0]))

# 创建2维张量
print(tf.constant([[1.0, 2.0, 3.0],
                   [1.0, 2.0, 3.0]]))

# 创建3维张量
print(tf.constant([
    [[1.0, 2.0, 3.0],
     [3.0, 4.0, 3.0]],
    [[1.0, 2.0, 3.0],
     [1.0, 2.0, 3.0]]
]))

t1 = tf.constant([1, 2, 3, 4, 5])
print(t1.numpy())
# 定义张量a和b
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]])

print(tf.add(a, b), "\n") # 计算张量的和
print(tf.multiply(a, b), "\n") # 计算张量的元素乘法
print(tf.matmul(a, b), "\n") # 计算乘法
print('a')