import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 1. 模型构建
net = tf.keras.models.Sequential([
    # 卷积层 96个卷积核 11*11 步长4
    tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation="relu"),
    # 池化：3*3 步长2
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    # 卷积 256个 5*5 relu
    tf.keras.layers.Conv2D(filters=256, kernel_size=5, padding='same', activation="relu"),
    # 池化
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    # 卷积 384 3*3 1
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation="relu"),
    tf.keras.layers.Conv2D(filters=384, kernel_size=3, padding='same', activation="relu"),
    tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same', activation="relu"),
    # 池化
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
    # 展开一维向量
    tf.keras.layers.Flatten(),
    # FC：4096 relu
    tf.keras.layers.Dense(4096, activation="relu"),
    # 随机失活
    tf.keras.layers.Dropout(0.5),
    # FC：4096 relu
    tf.keras.layers.Dense(4096, activation="relu"),
    # 随机失活
    tf.keras.layers.Dropout(0.5),
    # 输出层
    tf.keras.layers.Dense(10, activation="softmax")
])

# 2. 数据读取
# 获取手写数字数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 维度调整
train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))


# 对训练数据进行抽样
def get_train(size):
    # 随机生成index
    index = np.random.randint(0, train_images.shape[0], size)
    # 选择图像并进行resize
    resized_image = tf.image.resize_with_pad(train_images[index], 227, 227, )
    return resized_image.numpy(), train_labels[index]


# 对测试数据进行抽样
def get_test(size):
    # 随机生成index
    index = np.random.randint(0, test_images.shape[0], size)
    # 选择图像并进行resize
    resized_image = tf.image.resize_with_pad(test_images[index], 227, 227, )
    return resized_image.numpy(), test_labels[index]


# 抽样结果
train_images, train_labels = get_train(256)
test_images, test_labels = get_test(128)

# plt.imshow(train_images[4].astype(np.int8).squeeze(),cmap='gray')
# plt.show()
# 3. 模型编译
net.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
            optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
            metrics=tf.keras.metrics.sparse_categorical_accuracy,)
# 4. 模型训练
# 模型训练：指定训练数据，batchsize,epoch,验证集
net.fit(train_images,train_labels,batch_size=128,epochs=3,verbose=1,validation_split=0.1)
# 5. 模型评估
net.evaluate(test_images, test_labels, verbose=1)
