import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 1. 数据集加载
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 2. 数据处理
# 维度调整
train_images = tf.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
# print(train_images.shape)
test_images = tf.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))
# print(test_images.shape)
# 3. 模型构建
net = tf.keras.models.Sequential([
    # 卷积层01：6个5*5的卷积 sigmoid
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, strides=1, activation="sigmoid", input_shape=(28, 28, 1)),
    # 最大池化01
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    # 卷积层02：16个5*5
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation="sigmoid"),
    # 最大池化02
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    # 维度调整
    tf.keras.layers.Flatten(),
    # 全连接层
    tf.keras.layers.Dense(120, activation="sigmoid"),
    tf.keras.layers.Dense(84, activation="sigmoid"),
    tf.keras.layers.Dense(10, activation="softmax")
])
# print(net.summary())
# 4. 模型编译
# 优化器 损失函数 评价指标
optimizer = tf.keras.optimizers.SGD(learning_rate=0.9)
net.compile(optimizer=optimizer,
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=tf.keras.metrics.sparse_categorical_accuracy)
# 5. 模型训练
net.fit(train_images, train_labels, epochs=5, batch_size=128, verbose=1)
# 6. 模型评估
net.evaluate(test_images, test_labels, verbose=1)
