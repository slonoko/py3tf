#%%
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, optimizers
import cv2

gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus)>0:
    print("Using a GPU ...")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    #tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

cat = plt.imread('C:\\Users\\elie\\Documents\\Projects\\py3tf\\cat.png')
plt.imshow(cat)
#cat = cv2.resize(cat, (224, 224)).astype(np.float32)


cat.shape
#%%
model = models.Sequential()
model.add(layers.Convolution2D(3,3,3,input_shape=cat.shape))

cat_batch = np.expand_dims(cat, axis=0)
conv_cat = model.predict(cat_batch)
conv_cat = np.squeeze(conv_cat, axis=0)
print(conv_cat.shape)
plt.imshow(conv_cat)

# %%
model = models.Sequential()
model.add(layers.Convolution2D(3,5,5,input_shape=cat.shape))

cat_batch = np.expand_dims(cat, axis=0)
conv_cat = model.predict(cat_batch)
conv_cat = np.squeeze(conv_cat, axis=0)
print(conv_cat.shape)
plt.imshow(conv_cat)

# %%
model = models.Sequential()
model.add(layers.Convolution2D(3,3,3,activation='relu',input_shape=cat.shape))
cat_batch = np.expand_dims(cat, axis=0)
conv_cat = model.predict(cat_batch)
conv_cat = np.squeeze(conv_cat, axis=0)
print(conv_cat.shape)
plt.imshow(conv_cat)

# %%
model = models.Sequential()
model.add(layers.Convolution2D(3,3,3,activation='relu',input_shape=cat.shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

cat_batch = np.expand_dims(cat, axis=0)
conv_cat = model.predict(cat_batch)
conv_cat = np.squeeze(conv_cat, axis=0)
print(conv_cat.shape)
plt.imshow(conv_cat)

# %%
