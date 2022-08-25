# 导入相应的工具包
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger
import tensorflow as tf
# 数据集
from tensorflow.keras.datasets import mnist
# 构建序列模型
from tensorflow.keras.models import Sequential
# 导入需要的层
from tensorflow.keras.layers import Dense, Dropout, Activation,BatchNormalization
# 导入辅助工具包
from tensorflow.keras import utils
# 正则化
from tensorflow.keras import regularizers

# 数据集加载
(x_train,y_train),(x_test,y_test) = mnist.load_data()
print(x_train.shape)
print(x_test.shape)
print(y_train)
# 显示数据
plt.figure()
plt.imshow(x_train[10],cmap="gray")
# plt.show()
# 数据展示：将数据集的前九个数据集进行展示
# for i in range(9):
#     plt.subplot(3,3,i+1)
#     # 以灰度图显示，不进行插值
#     plt.imshow(x_train[i], cmap='gray', interpolation='none')
#     # 设置图片的标题：对应的类别
#     plt.title("数字{}".format(y_train[i]))
# 一 数据处理
# 样本数据
# 1. 数据维度的调整
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)
# 2. 数据类型调整
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# 3. 数据归一化
x_train = x_train/255
x_test = x_test/255
# 目标数据
# 目标值转化为热编码
y_train = utils.to_categorical(y_train,10)
y_test = utils.to_categorical(y_test,10)
# print(y_test)
# 二 模型构建
model = Sequential()
# 全连接层：2个隐层 一个输出层
# 第一个隐层
model.add(Dense(512,activation="relu",input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# 第二个隐层
model.add(Dense(512,kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))
# 输出层
model.add(Dense(10,activation="softmax"))
# print(model.summary())
# 三 模型编译
# 指明损失函数和优化器，评估指标
model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.Adam(),metrics=tf.keras.metrics.categorical_accuracy)
# 四 模型训练 指定训练集 epoch batch_size val verbose
history = model.fit(x_train,y_train,epochs=4,batch_size=128,validation_data=(x_test,y_test),verbose=1)
# 绘制损失函数
print(history.history)
plt.figure()
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')
plt.legend()
plt.grid()
plt.show()
# 准确率
plt.figure()
plt.plot(history.history['categorical_accuracy'],label="train")
plt.plot(history.history["val_categorical_accuracy"],label="val")
plt.legend()
plt.grid()
plt.show()

# #添加tensoboard观察
# tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./graph")
# #训练
# history = model.fit(x_train,y_train,epochs=4,validation_data=(x_test,y_test),batch_size=128,verbose=1,
#                     callbacks=[tensorboard])

# 模型评估
score = model.evaluate(x_test,y_test,verbose=1)
print(score)
# 模型保存
model.save("model.h5")
#加载模型
# loadmodel = tf.keras.models.load_model("model.h5")








