import tensorflow as  tf
from tensorflow import keras
from tensorflow.keras import layers

# 神经网络的搭建

# 定义一个sequential模型，包含3层
model = keras.Sequential(
    [
        # 第一个隐藏层
        layers.Dense(3, activation="relu", kernel_initializer="he_normal", name="layer01", input_shape=(3,)),
        # 第二个隐藏层
        layers.Dense(2, activation="relu", kernel_initializer="he_normal", name="layer02"),
        # 输出层
        layers.Dense(2, activation="sigmoid", kernel_initializer="he_normal", name="layer03")
    ],
    name="sequential_model"
)
# model.summary()

# functional API
# 输入层
inputs = tf.keras.Input(shape=(3,), name="input")
# 第一个隐藏层
x = tf.keras.layers.Dense(3, activation="relu", name="layer01")(inputs)
# 第二个隐藏层
x = tf.keras.layers.Dense(2, activation="relu", name="layer02")(x)
# 输出层
outputs = tf.keras.layers.Dense(2, activation="sigmoid", name="output")(x)
# 创建模型
model = keras.Model(inputs=inputs, outputs=outputs, name="functional_API_model")
# model.summary()


# model子类构建
class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer1 = tf.keras.layers.Dense(3, activation="relu", kernel_initializer="he_normal", name="layer1",
                                            input_shape=(3,))
        self.layer2 = tf.keras.layers.Dense(2, activation="relu", kernel_initializer="he_normal", name="layer2")
        self.layer3 = tf.keras.layers.Dense(2, activation="sigmoid", kernel_initializer="he_normal", name="layer3")

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        outputs = self.layer3(x)
        return outputs


model = MyModel()
x = tf.ones((1, 3))
y = model(x)
model.summary()
