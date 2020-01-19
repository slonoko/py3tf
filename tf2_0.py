#%%
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.client import device_lib

print(f'Tensorflow version {tf.__version__}\n')
print(device_lib.list_local_devices())

# %%
