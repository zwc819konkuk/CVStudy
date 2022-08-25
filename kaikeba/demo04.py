import tensorflow as tf

rank_0_tensor = tf.constant(4)
rank_1_tensor = tf.constant([1.0, 2.0, 3.0])
# print(rank_0_tensor)
# print(rank_1_tensor)


if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

