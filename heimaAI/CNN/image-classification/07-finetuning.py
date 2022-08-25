import tensorflow as tf
import numpy as np
import pathlib
import matplotlib.pyplot as plt

traindir = 'hotdog/train'
testdir = 'hotdog/test'

image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
train_data_gen = image_gen.flow_from_directory(traindir,batch_size=32,target_size=(224,224),shuffle=True)
test_data_gen = image_gen.flow_from_directory(testdir,batch_size=32,target_size=(224,224),shuffle=True)
image,label = next(train_data_gen)
plt.figure(figsize=(10,10))
for n in range(15):
    ax = plt.subplot(5,5,n+1)
    plt.imshow(image[n])
    plt.axis('off')
ResNet50 = tf.keras.applications.ResNet50(weights='imagenet',input_shape=(224,224,3))
for layer in ResNet50.layers:
    layer.trainable=False
net = tf.keras.models.Sequential()
net.add(ResNet50)
net.add(tf.keras.layers.Flatten())
net.add(tf.keras.layers.Dense(2,activation='sigmoid'))
# 模型编译：指定优化器，损失函数和评价指标
net.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])
# 模型训练：指定数据，每一个epoch中只运行10个迭代，指定验证数据集
history = net.fit(
                    train_data_gen,
                    steps_per_epoch=10,
                    epochs=3,
                    validation_data=test_data_gen,
                    validation_steps=10
                    )