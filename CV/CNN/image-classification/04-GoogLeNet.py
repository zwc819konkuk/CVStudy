import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist


# 定义Inception模块
class Inception(tf.keras.layers.Layer):
    # 输入参数为各个卷积的卷积核个数
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # 线路1：1 x 1卷积层，激活函数是RELU，padding是same
        self.p1_1 = tf.keras.layers.Conv2D(
            c1, kernel_size=1, activation='relu', padding='same')
        # 线路2，1 x 1卷积层后接3 x 3卷积层,激活函数是RELU，padding是same
        self.p2_1 = tf.keras.layers.Conv2D(
            c2[0], kernel_size=1, padding='same', activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], kernel_size=3, padding='same',
                                           activation='relu')
        # 线路3，1 x 1卷积层后接5 x 5卷积层,激活函数是RELU，padding是same
        self.p3_1 = tf.keras.layers.Conv2D(
            c3[0], kernel_size=1, padding='same', activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], kernel_size=5, padding='same',
                                           activation='relu')
        # 线路4，3 x 3最大池化层后接1 x 1卷积层,激活函数是RELU，padding是same
        self.p4_1 = tf.keras.layers.MaxPool2D(
            pool_size=3, padding='same', strides=1)
        self.p4_2 = tf.keras.layers.Conv2D(
            c4, kernel_size=1, padding='same', activation='relu')

    # 完成前向传播过程
    def call(self, x):
        # 线路1
        p1 = self.p1_1(x)
        # 线路2
        p2 = self.p2_2(self.p2_1(x))
        # 线路3
        p3 = self.p3_2(self.p3_1(x))
        # 线路4
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维上concat输出
        outputs = tf.concat([p1, p2, p3, p4], axis=-1)
        return outputs


# GoogLeNet构建
# 定义模型的输入
inputs = tf.keras.Input(shape=(224, 224, 3), name="input")
# b1 模块
# 卷积层7*7的卷积核，步长为2，pad是same，激活函数RELU
x = tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same', activation='relu')(inputs)
# 最大池化：窗口大小为3*3，步长为2，pad是same
x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
# b2 模块
# 卷积层1*1的卷积核，步长为2，pad是same，激活函数RELU
x = tf.keras.layers.Conv2D(64, kernel_size=1, padding='same', activation='relu')(x)
# 卷积层3*3的卷积核，步长为2，pad是same，激活函数RELU
x = tf.keras.layers.Conv2D(192, kernel_size=3, padding='same', activation='relu')(x)
# 最大池化：窗口大小为3*3，步长为2，pad是same
x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
# b3 模块
# Inception
x = Inception(64, (96, 128), (16, 32), 32)(x)
# Inception
x = Inception(128, (128, 192), (32, 96), 64)(x)
# 最大池化：窗口大小为3*3，步长为2，pad是same
x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)


# 辅助分类器
def aux_classifier(x, filter_size):
    # x:输入数据，filter_size:卷积层卷积核个数，全连接层神经元个数
    # 池化层
    x = tf.keras.layers.AveragePooling2D(
        pool_size=5, strides=3, padding='same')(x)
    # 1x1 卷积层
    x = tf.keras.layers.Conv2D(filters=filter_size[0], kernel_size=1, strides=1,
                               padding='valid', activation='relu')(x)
    # 展平
    x = tf.keras.layers.Flatten()(x)
    # 全连接层1
    x = tf.keras.layers.Dense(units=filter_size[1], activation='relu')(x)
    # softmax输出层
    x = tf.keras.layers.Dense(units=10, activation='softmax')(x)
    return x


# b4 模块
# Inception
x = Inception(192, (96, 208), (16, 48), 64)(x)
# 辅助输出1
aux_output_1 = aux_classifier(x, [128, 1024])
# Inception
x = Inception(160, (112, 224), (24, 64), 64)(x)
# Inception
x = Inception(128, (128, 256), (24, 64), 64)(x)
# Inception
x = Inception(112, (144, 288), (32, 64), 64)(x)
# 辅助输出2
aux_output_2 = aux_classifier(x, [128, 1024])
# Inception
x = Inception(256, (160, 320), (32, 128), 128)(x)
# 最大池化
x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)

# b5 模块
# Inception
x = Inception(256, (160, 320), (32, 128), 128)(x)
# Inception
x = Inception(384, (192, 384), (48, 128), 128)(x)
# GAP
x = tf.keras.layers.GlobalAvgPool2D()(x)
# 输出层
main_outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
# 使用Model来创建模型，指明输入和输出
model = tf.keras.Model(inputs=inputs, outputs=[main_outputs, aux_output_1, aux_output_2])
# model.summary()

# 获取手写数字数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 训练集数据维度的调整：N H W C
train_images = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
# 测试集数据维度的调整：N H W C
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
train_images, train_labels = get_train(256)
test_images, test_labels = get_test(128)
# 指定优化器，损失函数和评价指标
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)
# 模型有3个输出，所以指定损失函数对应的权重系数
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'], loss_weights=[1, 0.3, 0.3])
# 模型训练：指定训练数据，batchsize,epoch,验证集
model.fit(train_images, train_labels, batch_size=128, epochs=3, verbose=1, validation_split=0.1)
