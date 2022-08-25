import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


# 1. 模型构建
## VGG块的构建
def vgg_block(num_conv, num_filters):
    # 构建序列模型
    blk = tf.keras.models.Sequential()
    # 遍历卷积层
    for _ in range(num_conv):
        # 设置卷积层
        blk.add(tf.keras.layers.Conv2D(num_filters, kernel_size=3, padding='same', activation='relu'))
    # 池化层
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return blk


# 使用vgg块构建模型
def vgg(conv_arch):
    # 序列模型
    net = tf.keras.models.Sequential()
    # 生成卷积部分
    for (num_convs, num_filters) in conv_arch:
        net.add(vgg_block(num_convs, num_filters))
    # 全连接层
    net.add(tf.keras.models.Sequential([
        # 展平特征图
        tf.keras.layers.Flatten(),
        # FC
        tf.keras.layers.Dense(4096, activation='relu'),
        # 随机失活
        tf.keras.layers.Dropout(0.5),
        # FC
        tf.keras.layers.Dense(4096, activation='relu'),
        # 随机失活
        tf.keras.layers.Dropout(0.5),
        # 输出
        tf.keras.layers.Dense(10, activation='softmax')
    ]))
    return net


# 卷积块的参数
conv_arch = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
net = vgg(conv_arch)
x = tf.random.uniform((1, 224, 224, 1))
y = net(x)
net.summary()

# 2. 数据读取
# 获取手写数字数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 维度调整
train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
test_images = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))


# 定义两个方法随机抽取部分样本演示
# 获取训练集数据
def get_train(size):
    # 随机生成要抽样的样本的索引
    index = np.random.randint(0, np.shape(train_images)[0], size)
    # 将这些数据resize成22*227大小
    resized_images = tf.image.resize_with_pad(train_images[index], 224, 224, )
    # 返回抽取的
    return resized_images.numpy(), train_labels[index]


# 获取测试集数据
def get_test(size):
    # 随机生成要抽样的样本的索引
    index = np.random.randint(0, np.shape(test_images)[0], size)
    # 将这些数据resize成224*224大小
    resized_images = tf.image.resize_with_pad(test_images[index], 224, 224, )
    # 返回抽样的测试样本
    return resized_images.numpy(), test_labels[index]

# 获取训练样本和测试样本
train_images,train_labels = get_train(2560)
test_images,test_labels = get_test(1280)
# 指定优化器，损失函数和评价指标
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

net.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 模型训练：指定训练数据，batchsize,epoch,验证集
net.fit(train_images,train_labels,batch_size=128,epochs=3,verbose=1,validation_split=0.1)
# 指定测试数据
net.evaluate(test_images,test_labels,verbose=1)