import tensorflow as  tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

# 图像增强
cat = plt.imread('./cat.jpg')
plt.imshow(cat)
# plt.show()
cat1 = tf.image.random_flip_left_right(cat)
cat2 = tf.image.random_flip_up_down(cat)
cat3 = tf.image.random_crop(cat,(200,200,3))
cat4 = tf.image.random_brightness(cat,0.5)
cat5 = tf.image.random_hue(cat,0.5)
plt.imshow(cat5)
# plt.show()
# 获取数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# 将数据转换为4维的形式
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
# 设置图像增强方式：水平翻转
datagen = tf.keras.preprocessing.image.ImageDataGenerator(height_shift_range=10)
# 查看增强后的结果
for X_batch,y_batch in datagen.flow(x_train,y_train,batch_size=9):
    plt.figure(figsize=(8,8)) # 设定每个图像显示的大小
    # 产生一个3*3网格的图像
    for i in range(0,9):
        plt.subplot(330+1+i)
        plt.title(y_batch[i])
        plt.axis('off')
        plt.imshow(X_batch[i].reshape(28,28),cmap='gray')
    plt.show()
    break