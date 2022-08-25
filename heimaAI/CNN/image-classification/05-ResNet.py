import tensorflow as tf
from tensorflow.keras import layers, activations
from tensorflow.keras.datasets import mnist
import numpy as np

# 残差块
class Residual(tf.keras.Model):
    # 定义网络结构
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super(Residual, self).__init__()
        # 卷积层
        self.conv1 = layers.Conv2D(num_channels, padding='same', kernel_size=3, strides=strides)
        # 卷积层
        self.conv2 = layers.Conv2D(num_channels, kernel_size=3, strides=strides)
        # 是否使用1*1卷积
        if use_1x1conv:
            self.conv3 = layers.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None
        # BN
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()

    # 定义前向传播过程
    def call(self, x):
        Y = activations.relu(self.bn1(self.conv1(x)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            x = self.conv3(x)
        outputs = activations.relu(Y + x)
        return outputs


# ResNet模块的构成
class ResNetBlock(tf.keras.layers.Layer):
    # 定义所需的网络结构
    def __init__(self, num_channels, num_res, first_block=False):
        super(ResNetBlock, self).__init__()
        # 存储残差块
        self.listLayers = []
        # 遍历残差生成模块
        for i in range(num_res):
            # 如果是第一个残差块，不是第一个模块时
            if i == 0 and not first_block:
                self.listLayers.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.listLayers.append(Residual(num_channels))

    # 定义前向传播
    def call(self, x):
        for layer in self.listLayers.layers:
            x = layer(x)
        return x


# 构建ResNet
class ResNet(tf.keras.Model):
    # 定义网络的构成
    def __init__(self, num_blocks):
        super(ResNet, self).__init__()
        # 输入层
        self.conv = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')
        # BN
        self.bn = layers.BatchNormalization()
        # activation
        self.relu = layers.Activation('relu')
        # 池化
        self.mp = layers.MaxPool2D(pool_size=3, strides=2, padding="same")
        # 残差模块
        self.res_block1 = ResNetBlock(64, num_blocks[0], first_block=True)
        self.res_block2 = ResNetBlock(128, num_blocks[1])
        self.res_block3 = ResNetBlock(256, num_blocks[2])
        self.res_block4 = ResNetBlock(512, num_blocks[3])
        # GAP
        self.gap = layers.GlobalAvgPool2D()
        # fc
        self.fc = layers.Dense(units=10, activation='softmax')

    # 前向传播
    def call(self, x):
        # 输入部分的传输过程
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp(x)
        # block
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        # 输出部分的传输
        x = self.gap(x)
        x = self.fc(x)
        return x


# 实例化
mynet = ResNet([2, 2, 2, 2])
# X = tf.random.uniform((1, 224, 224, 1))
# y = mynet(X)
# # mynet.summary()
# 获取手写数字数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 训练集数据维度的调整：N H W C
train_images = np.reshape(train_images,(train_images.shape[0],train_images.shape[1],train_images.shape[2],1))
# 测试集数据维度的调整：N H W C
test_images = np.reshape(test_images,(test_images.shape[0],test_images.shape[1],test_images.shape[2],1))
# 定义两个方法随机抽取部分样本演示
# 获取训练集数据
def get_train(size):
    # 随机生成要抽样的样本的索引
    index = np.random.randint(0, np.shape(train_images)[0], size)
    # 将这些数据resize成22*227大小
    resized_images = tf.image.resize_with_pad(train_images[index],224,224,)
    # 返回抽取的
    return resized_images.numpy(), train_labels[index]
# 获取测试集数据
def get_test(size):
    # 随机生成要抽样的样本的索引
    index = np.random.randint(0, np.shape(test_images)[0], size)
    # 将这些数据resize成224*224大小
    resized_images = tf.image.resize_with_pad(test_images[index],224,224,)
    # 返回抽样的测试样本
    return resized_images.numpy(), test_labels[index]
# 获取训练样本和测试样本
train_images,train_labels = get_train(256)
test_images,test_labels = get_test(128)
# 指定优化器，损失函数和评价指标
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0)

mynet.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 模型训练：指定训练数据，batchsize,epoch,验证集
mynet.fit(train_images,train_labels,batch_size=128,epochs=3,verbose=1,validation_split=0.1)