from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
(train_images,train_labels),(test_images,test_labels) = cifar10.load_data()
print(test_images.shape)
print(train_images.shape)
plt.figure(figsize=(1,1))
plt.imshow(train_images[4])
plt.show()