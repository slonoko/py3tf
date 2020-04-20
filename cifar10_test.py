#%% Testing CIFAR10

import numpy as np
from skimage.transform import resize
from imageio import imread
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import SGD

# load model
model_architecture = 'c:\\Users\\elie\\Documents\\Projects\\py3tf\\cifar10_architecture.json'
model_weights = 'c:\\Users\\elie\\Documents\\Projects\\py3tf\\cifar10_weights.h5'
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
# load images
img_names = ['cat.jpg']
imgs = [resize(imread(img_name), (32, 32, 3)).astype("float32") for img_name in img_names]
imgs = np.array(imgs) / 255
print("imgs.shape:", imgs.shape)
# train
optim = SGD()
model.compile(loss='categorical_crossentropy', optimizer=optim,
              metrics=['accuracy'])
# predict
predictions = model.predict_classes(imgs)
print("predictions:", predictions)



# %%
