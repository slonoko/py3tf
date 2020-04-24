#%%
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plt
import numpy as np
import cv2


gpus = tf.config.experimental.list_physical_devices('GPU')

if len(gpus)>0:
    print("Using a GPU ...")
    tf.config.experimental.set_memory_growth(gpus[0], True)

# prebuild model with pre-trained weights on imagenet
model = VGG16(weights='imagenet', include_top=True)
model.compile(optimizer='sgd', loss='categorical_crossentropy')
# resize into VGG16 trained images' format
im = cv2.resize(cv2.imread('cat.png'), (224, 224))
im = np.expand_dims(im, axis=0)
im.astype(np.float32)
# predict
out = model.predict(im)
index = np.argmax(out)
print(index)
plt.plot(out.ravel())
plt.show()
# this should print 820 for steaming train


# %%
