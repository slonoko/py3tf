import tensorflow as tf
from tensorflow.python.client import device_lib

print(f'Tensorflow version {tf.__version__}\n')
print(device_lib.list_local_devices())

W = tf.Variable(tf.ones(shape=(2, 2)), name="W")
b = tf.Variable(tf.zeros(shape=(2, 2)), name="b")


@tf.function
def model(x):
    return W * x + b


out_a = model([1, 0])
print(out_a)
