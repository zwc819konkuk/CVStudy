import numpy as np
import tensorflow as tf

# dropout层
layer = tf.keras.layers.Dropout(0.2, input_shape=(2,))
# 定义输入数据
data = np.arange(1, 11).reshape(5, 2).astype(np.float32)
# print(data)
outputs = layer(data, training=True)
# print(outputs)

#early-stopping
callback = tf.keras.callbacks.EarlyStopping(monitor="loss",patience=3)
#定义一层的网络
model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
#模型编译
model.compile(tf.keras.optimizers.SGD(),loss="mse")
#模型训练
history = model.fit(np.arange(100).reshape(5,20),np.array([0,1,0,1,1]),epochs=10,callbacks=[callback],batch_size=1,verbose=1)
print(len(history.history['loss']))































