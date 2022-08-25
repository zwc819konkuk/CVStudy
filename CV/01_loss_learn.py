import tensorflow as tf

# 交叉熵损失函数[多分类]
# 设置真实值和预测值
y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.9, 0.05], [0.01, 0.02, 0.97]]
# 实例化交叉熵损失
cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true, y_pred))

# 二分类
y_true = [[0], [1]]
y_pred = [[0.1], [0.9]]
bce = tf.keras.losses.BinaryCrossentropy()
print(bce(y_true, y_pred))

# 回归问题
# MAE[L1 loss]
y_true = [[0.], [1.]]
y_pred = [[0.], [1.]]
mae = tf.keras.losses.MeanAbsoluteError()
print(mae(y_true, y_pred))
# MSE[L2 loss 欧式距离]
y_true = [[0.], [1.]]
y_pred = [[1.], [1.]]
mse = tf.keras.losses.MeanSquaredError()
print(mse(y_true, y_pred))
# smooth L1 loss
y_true = [[0.], [1.]]
y_pred = [[0.2], [0.8]]
smooth = tf.keras.losses.Huber()
print(smooth(y_true, y_pred))