import tensorflow as tf

# 梯度下降算法
# SGD
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
# 定义要更新的参数
var = tf.Variable(1.0)
# 定义损失函数
loss = lambda: (var ** 2) / 2.0
# 计算损失梯度 并进行参数更新
opt.minimize(loss, [var]).numpy()
# 参数更新结果
print(var.numpy())

# 动量梯度下降优化 [对梯度优化]
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
var = tf.Variable(1.0)
val0 = var.value()
# 定义损失函数
loss = lambda: (var ** 2) / 2.0
# 第一次更新
opt.minimize(loss, [var]).numpy()
var1 = var.value()
# 第二次更新
opt.minimize(loss, [var]).numpy()
var2 = var.value()
print(val0)
print(var1 - val0)
print(var2 - var1)
print(1e-02)

# Adagrad[对learning rate进行优化]
opt = tf.keras.optimizers.Adagrad(learning_rate=0.1, initial_accumulator_value=0.1, epsilon=1e-06)
# 定义要更新的参数
var = tf.Variable(1.0)


# 损失函数
def loss():
    return (var ** 2) / 2.0


# 进行更新
opt.minimize(loss, [var]).numpy()
print(var)

# RMSprop[对learning rate进行优化]
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1, epsilon=1e-06, rho=0.1)
var = tf.Variable(1.0)


def loss():
    return (var ** 2) / 2.0


opt.minimize(loss, [var]).numpy()
print(var.numpy())

#Adam
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
var = tf.Variable(1.0)
def loss():
    return (var**2)/2.0
opt.minimize(loss,[var]).numpy()
print(var.numpy())






























